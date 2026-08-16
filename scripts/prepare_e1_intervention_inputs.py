"""Selectively extract accepted E1 checkpoint inputs from the sealed Study 1 archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path


CHECKPOINT_RE = re.compile(
    r"^outputs/runs/neural-full-parallel/step10_neural_full_parallel_v1/"
    r"workers/worker_(?P<worker>\d{2})/work/(?P<fold>\d{2})_(?P<site>[^/]+)/"
    r"learned_bunn_density_(?P<density>0\.(?:01|05|10|20))/final/"
    r"seed_(?P<seed>\d+)\.pt$"
)
ROOT_PREFIX = "outputs/runs/neural-full-parallel/step10_neural_full_parallel_v1/"
SOURCE_FILES = (
    "metadata.json",
    "predictions.csv",
    "training_curves.csv",
    "tuning_scores.csv",
    "fit_runtime.csv",
    "fit_warnings.csv",
    "diagnostics.csv",
)


class PreparationError(ValueError):
    """Raised when the frozen archive or selected E1 input set is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_copy_member(archive: tarfile.TarFile, member: tarfile.TarInfo, destination: Path) -> str:
    source = archive.extractfile(member)
    if source is None or not member.isfile():
        raise PreparationError(f"Archive member is not a regular file: {member.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as output:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            output.write(chunk)
            digest.update(chunk)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, destination)
    return digest.hexdigest()


def prepare(archive_path: Path, contract_path: Path, output_dir: Path) -> dict[str, object]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected_archive = contract["input_hashes"]["sealed_archive"]
    observed_archive = sha256_file(archive_path)
    if observed_archive != expected_archive:
        raise PreparationError("Sealed Study 1 archive hash differs from the frozen E1 contract")

    # Stream once. Random access inside a gzip tar repeatedly decompresses the
    # archive and is prohibitively slow for hundreds of selected members.
    checkpoint_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    observed_cells: set[tuple[int, str, float, int]] = set()
    required_sources = {ROOT_PREFIX + name: name for name in SOURCE_FILES}
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            match = CHECKPOINT_RE.match(member.name)
            if match is not None:
                fold = int(match.group("fold"))
                site = match.group("site")
                density = float(match.group("density"))
                seed = int(match.group("seed"))
                cell = (fold, site, density, seed)
                if cell in observed_cells:
                    raise PreparationError(f"Duplicate accepted checkpoint cell: {cell}")
                observed_cells.add(cell)
                relative = Path("checkpoints") / f"{fold:02d}_{site}" / f"density_{density:.2f}" / f"seed_{seed}.pt"
                digest = atomic_copy_member(archive, member, output_dir / relative)
                checkpoint_rows.append(
                    {
                        "fold": fold,
                        "site": site,
                        "density": density,
                        "seed": seed,
                        "worker": int(match.group("worker")),
                        "archive_member": member.name,
                        "path": relative.as_posix(),
                        "sha256": digest,
                        "bytes": member.size,
                    }
                )
            elif member.name in required_sources:
                name = required_sources[member.name]
                relative = Path("source") / name
                digest = atomic_copy_member(archive, member, output_dir / relative)
                source_rows.append(
                    {
                        "archive_member": member.name,
                        "path": relative.as_posix(),
                        "sha256": digest,
                        "bytes": member.size,
                    }
                )

    checkpoint_rows.sort(key=lambda row: str(row["archive_member"]))
    source_rows.sort(key=lambda row: str(row["archive_member"]))
    expected_count = int(contract["cohort"]["accepted_final_learned_bunn_checkpoints"])
    if len(checkpoint_rows) != expected_count:
        raise PreparationError(
            f"Expected {expected_count} accepted learned-BuNN checkpoints, found {len(checkpoint_rows)}"
        )
    found_sources = {str(row["archive_member"]) for row in source_rows}
    missing_sources = set(required_sources) - found_sources
    if missing_sources:
        raise PreparationError(f"Sealed archive lacks canonical source files: {sorted(missing_sources)}")

    folds = sorted({row["fold"] for row in checkpoint_rows})
    sites = sorted({str(row["site"]) for row in checkpoint_rows})
    densities = sorted({float(row["density"]) for row in checkpoint_rows})
    seeds = sorted({int(row["seed"]) for row in checkpoint_rows})
    if folds != list(range(18)) or len(sites) != 18:
        raise PreparationError("Accepted checkpoint set does not cover all 18 frozen folds/sites")
    if densities != [0.01, 0.05, 0.1, 0.2]:
        raise PreparationError("Accepted checkpoint set has an unexpected density grid")
    if seeds != [20260803, 20260804, 20260805, 20260806, 20260807]:
        raise PreparationError("Accepted checkpoint set has an unexpected final-seed grid")

    manifest: dict[str, object] = {
        "state": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive_path": str(archive_path),
        "archive_sha256": observed_archive,
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "checkpoint_count": len(checkpoint_rows),
        "folds": folds,
        "sites": sites,
        "densities": densities,
        "seeds": seeds,
        "checkpoints": checkpoint_rows,
        "canonical_source_files": source_rows,
        "notice": "Input preparation only; no checkpoint inference or result analysis was performed.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "input_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_dir / "input_manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive", type=Path,
        default=Path("outputs/archives/step10_neural_full_parallel_v1.sealed.tar.gz"),
    )
    parser.add_argument(
        "--contract", type=Path,
        default=Path("configs/extensions/e1_checkpoint_interventions_v1.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/extensions/e1_interventions_v1/inputs"),
    )
    args = parser.parse_args()
    manifest = prepare(args.archive, args.contract, args.output_dir)
    print(json.dumps({
        "state": manifest["state"],
        "checkpoint_count": manifest["checkpoint_count"],
        "site_count": len(manifest["sites"]),
        "result_values_opened": False,
    }, indent=2))


if __name__ == "__main__":
    main()
