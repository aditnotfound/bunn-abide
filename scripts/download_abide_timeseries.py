"""Download the frozen ABIDE-I ROI time-series manifest with resumable checks.

The script deliberately has no third-party dependencies so it can run on a
fresh AWS Ubuntu instance. It never modifies the manifest and stores each
public PCP derivative under a deterministic local path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="data/processed/abide_i_manifest_cpac_filt_noglobal_rois_aal.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/abide_i/cpac/filt_noglobal/rois_aal",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--limit", type=int, default=None, help="Download only the first N rows for a smoke test."
    )
    parser.add_argument(
        "--summary-output",
        default="data/processed/abide_i_timeseries_download_summary.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"file_id", "timeseries_url"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Manifest lacks required columns: {sorted(missing)}")
    return rows


def destination_for(row: dict[str, str], output_dir: Path) -> Path:
    return output_dir / f"{row['file_id']}_rois_aal.1D"


def download_one(
    row: dict[str, str], output_dir: Path, timeout: float, retries: int
) -> tuple[str, str, int]:
    destination = destination_for(row, output_dir)
    if destination.exists() and destination.stat().st_size > 0:
        return row["file_id"], "skipped", destination.stat().st_size

    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(row["timeseries_url"])
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with partial.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
            size = partial.stat().st_size
            if size == 0:
                raise OSError("Downloaded file is empty")
            os.replace(partial, destination)
            return row["file_id"], "downloaded", size
        except (OSError, urllib.error.URLError, TimeoutError) as error:
            partial.unlink(missing_ok=True)
            if attempt == retries:
                return row["file_id"], f"failed: {error}", 0
            time.sleep(attempt)
    raise AssertionError("unreachable")


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    summary_path = Path(args.summary_output)
    rows = read_manifest(manifest_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No manifest rows selected")

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, str, int]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(download_one, row, output_dir, args.timeout, args.retries)
            for row in rows
        ]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort()
    failed = [result for result in results if result[1].startswith("failed:")]
    summary = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "output_dir": str(output_dir),
        "requested": len(rows),
        "downloaded": sum(status == "downloaded" for _, status, _ in results),
        "skipped_existing": sum(status == "skipped" for _, status, _ in results),
        "failed": [{"file_id": file_id, "error": status} for file_id, status, _ in failed],
        "total_bytes_this_run": sum(size for _, status, size in results if status == "downloaded"),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
