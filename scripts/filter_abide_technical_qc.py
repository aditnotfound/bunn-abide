"""Derive the analysis manifest after non-negotiable connectome eligibility checks.

The parent Step 5 manifest is never changed. A participant is retained only if
the downloaded 116-region AAL time series can yield a finite Pearson/Fisher-z
connectome. Every rejected row and reason is written to a separate CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from scripts.build_abide_connectomes import fisher_z_connectome, read_aal_timeseries
except ModuleNotFoundError:  # Allows `python scripts/filter_abide_technical_qc.py`.
    from build_abide_connectomes import fisher_z_connectome, read_aal_timeseries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="data/processed/abide_i_manifest_cpac_filt_noglobal_rois_aal.csv",
    )
    parser.add_argument(
        "--timeseries-dir",
        default="data/raw/abide_i/cpac/filt_noglobal/rois_aal",
    )
    parser.add_argument(
        "--output",
        default="data/processed/abide_i_analysis_manifest.csv",
    )
    parser.add_argument(
        "--exclusions-output",
        default="data/processed/abide_i_technical_exclusions.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="data/processed/abide_i_technical_qc_summary.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def site_class_counts(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[row["site_id"]][row["label_asd"]] += 1
    return {
        site: {
            "asd": count.get("1", 0),
            "control": count.get("0", 0),
            "total": sum(count.values()),
        }
        for site, count in sorted(counts.items())
    }


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    exclusions_path = Path(args.exclusions_output)
    summary_path = Path(args.summary_output)
    fieldnames, rows = read_manifest(manifest_path)
    required = {"subject_id", "site_id", "file_id", "label_asd"}
    if missing := required - set(fieldnames):
        raise ValueError(f"Manifest lacks columns: {sorted(missing)}")

    retained: list[dict[str, str]] = []
    exclusions: list[dict[str, str]] = []
    for row in rows:
        path = Path(args.timeseries_dir) / f"{row['file_id']}_rois_aal.1D"
        try:
            _, timeseries = read_aal_timeseries(path)
            fisher_z_connectome(timeseries)
            retained.append(row)
        except (OSError, ValueError) as error:
            exclusions.append(
                {
                    "subject_id": row["subject_id"],
                    "site_id": row["site_id"],
                    "file_id": row["file_id"],
                    "label_asd": row["label_asd"],
                    "reason": str(error),
                }
            )

    per_site = site_class_counts(retained)
    invalid_sites = [
        site for site, counts in per_site.items() if not counts["asd"] or not counts["control"]
    ]
    if invalid_sites:
        raise ValueError(
            "Technical QC would make held-out-site balanced accuracy undefined at: "
            + ", ".join(invalid_sites)
        )

    write_csv(output_path, fieldnames, retained)
    write_csv(
        exclusions_path,
        ["subject_id", "site_id", "file_id", "label_asd", "reason"],
        exclusions,
    )
    summary = {
        "parent_manifest": str(manifest_path),
        "parent_manifest_sha256": sha256_file(manifest_path),
        "eligibility_rule": (
            "A downloaded AAL-116 time series must have a valid header, finite numeric "
            "values, non-zero variance at every ROI, and yield a finite Pearson/Fisher-z connectome."
        ),
        "parent_rows": len(rows),
        "retained_rows": len(retained),
        "excluded_rows": len(exclusions),
        "analysis_manifest": str(output_path),
        "analysis_manifest_sha256": sha256_file(output_path),
        "technical_exclusions": str(exclusions_path),
        "technical_exclusions_sha256": sha256_file(exclusions_path),
        "class_counts": {
            "asd": sum(row["label_asd"] == "1" for row in retained),
            "control": sum(row["label_asd"] == "0" for row in retained),
        },
        "site_counts": per_site,
        "exclusion_site_counts": dict(Counter(row["site_id"] for row in exclusions)),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
