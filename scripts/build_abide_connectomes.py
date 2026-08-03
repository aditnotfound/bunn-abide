"""Build validated Fisher-z ABIDE-I connectomes from the frozen ROI manifest.

The input is the Step 5 participant manifest plus PCP AAL ROI time-series
files.  This script performs no diagnosis-, site-, or model-dependent data
selection.  It stops if any selected file is malformed rather than silently
changing the cohort.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_REGIONS = 116


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
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=None)
    selection.add_argument(
        "--site-diverse-limit",
        type=int,
        default=None,
        help="For a smoke test, select the first subject from each site in manifest order.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/abide_i_connectomes_fisher_z.npz",
    )
    parser.add_argument(
        "--qc-output",
        default="data/processed/abide_i_connectome_qc.csv",
    )
    parser.add_argument(
        "--metadata-output",
        default="data/processed/abide_i_connectome_metadata.json",
    )
    parser.add_argument(
        "--code-version",
        default="unknown",
        help="Git commit or other immutable code identifier recorded in metadata.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"subject_id", "site_id", "file_id", "dx_group", "label_asd"}
    if not rows:
        raise ValueError("Manifest has no rows")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Manifest lacks columns: {sorted(missing)}")
    return rows


def select_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        return rows[: args.limit]
    if args.site_diverse_limit is not None:
        if args.site_diverse_limit <= 0:
            raise ValueError("--site-diverse-limit must be positive")
        selected: list[dict[str, str]] = []
        seen_sites: set[str] = set()
        for row in rows:
            if row["site_id"] not in seen_sites:
                selected.append(row)
                seen_sites.add(row["site_id"])
            if len(selected) == args.site_diverse_limit:
                break
        return selected
    return rows


def read_aal_timeseries(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    """Return the PCP AAL labels and T-by-116 numeric matrix."""
    labels: tuple[str, ...] | None = None
    values: list[list[float]] = []

    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                if labels is None:
                    labels = tuple(stripped.split())
                continue
            fields = stripped.split()
            if len(fields) != EXPECTED_REGIONS:
                raise ValueError(
                    f"line {line_number}: expected {EXPECTED_REGIONS} values, got {len(fields)}"
                )
            try:
                values.append([float(value) for value in fields])
            except ValueError as error:
                raise ValueError(f"line {line_number}: non-numeric value") from error

    if labels is None:
        raise ValueError("missing AAL header")
    if len(labels) != EXPECTED_REGIONS:
        raise ValueError(
            f"header has {len(labels)} regions, expected {EXPECTED_REGIONS}"
        )
    if len(set(labels)) != EXPECTED_REGIONS:
        raise ValueError("AAL header contains duplicate region labels")
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 3:
        raise ValueError("fewer than three numeric time points")
    if not np.isfinite(matrix).all():
        raise ValueError("contains NaN or infinite time-series values")
    if np.any(np.std(matrix, axis=0) == 0):
        raise ValueError("contains a zero-variance ROI")
    return labels, matrix


def fisher_z_connectome(timeseries: np.ndarray) -> tuple[np.ndarray, int]:
    correlation = np.corrcoef(timeseries, rowvar=False)
    if correlation.shape != (EXPECTED_REGIONS, EXPECTED_REGIONS):
        raise ValueError(f"unexpected correlation shape: {correlation.shape}")
    if not np.isfinite(correlation).all():
        raise ValueError("correlation matrix contains NaN or infinite values")

    np.fill_diagonal(correlation, 0.0)
    off_diagonal = ~np.eye(EXPECTED_REGIONS, dtype=bool)
    clip_mask = off_diagonal & (np.abs(correlation) >= 1.0)
    clipped_count = int(clip_mask.sum())
    if clipped_count:
        epsilon = np.finfo(np.float64).eps
        correlation[clip_mask] = np.sign(correlation[clip_mask]) * (1.0 - epsilon)

    fisher_z = np.zeros_like(correlation, dtype=np.float64)
    fisher_z[off_diagonal] = np.arctanh(correlation[off_diagonal])
    np.fill_diagonal(fisher_z, 0.0)
    if not np.isfinite(fisher_z).all():
        raise ValueError("Fisher-z matrix contains NaN or infinite values")
    if not np.allclose(fisher_z, fisher_z.T, atol=1e-6):
        raise ValueError("Fisher-z matrix is not symmetric")
    return fisher_z.astype(np.float32), clipped_count


def process_rows(
    rows: list[dict[str, str]], timeseries_dir: Path
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], tuple[str, ...], list[dict[str, str]]]:
    connectomes: list[np.ndarray] = []
    edge_features: list[np.ndarray] = []
    qcs: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    expected_labels: tuple[str, ...] | None = None
    edge_i, edge_j = np.tril_indices(EXPECTED_REGIONS, k=-1)

    for row in rows:
        path = timeseries_dir / f"{row['file_id']}_rois_aal.1D"
        try:
            labels, timeseries = read_aal_timeseries(path)
            if expected_labels is None:
                expected_labels = labels
            elif labels != expected_labels:
                raise ValueError("AAL region order differs from the first selected file")
            connectome, clipped_count = fisher_z_connectome(timeseries)
            connectomes.append(connectome)
            edge_features.append(connectome[edge_i, edge_j])
            qcs.append(
                {
                    "subject_id": row["subject_id"],
                    "site_id": row["site_id"],
                    "file_id": row["file_id"],
                    "timepoints": int(timeseries.shape[0]),
                    "regions": int(timeseries.shape[1]),
                    "timeseries_min": float(timeseries.min()),
                    "timeseries_max": float(timeseries.max()),
                    "timeseries_mean": float(timeseries.mean()),
                    "timeseries_std": float(timeseries.std()),
                    "fisher_z_min": float(connectome.min()),
                    "fisher_z_max": float(connectome.max()),
                    "perfect_correlation_values_clipped": clipped_count,
                }
            )
        except (OSError, ValueError) as error:
            failures.append(
                {
                    "subject_id": row["subject_id"],
                    "site_id": row["site_id"],
                    "file_id": row["file_id"],
                    "error": str(error),
                }
            )

    if failures:
        return {}, qcs, tuple(), failures
    assert expected_labels is not None
    arrays = {
        "subject_id": np.asarray([row["subject_id"] for row in rows]),
        "site_id": np.asarray([row["site_id"] for row in rows]),
        "label_asd": np.asarray([int(row["label_asd"]) for row in rows], dtype=np.int8),
        "region_labels": np.asarray(expected_labels),
        "connectomes_fisher_z": np.stack(connectomes),
        "edge_i": edge_i.astype(np.int16),
        "edge_j": edge_j.astype(np.int16),
        "edge_features_fisher_z": np.stack(edge_features),
    }
    return arrays, qcs, expected_labels, failures


def write_qc(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    qc_path = Path(args.qc_output)
    metadata_path = Path(args.metadata_output)
    rows = select_rows(load_manifest(manifest_path), args)
    arrays, qc_rows, region_labels, failures = process_rows(rows, Path(args.timeseries_dir))

    metadata: dict[str, Any] = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "timeseries_dir": str(args.timeseries_dir),
        "selected_rows": len(rows),
        "selection": {
            "limit": args.limit,
            "site_diverse_limit": args.site_diverse_limit,
        },
        "transform": {
            "atlas": "AAL-116",
            "connectivity": "Pearson ROI time-series correlation",
            "feature_transform": "Fisher-z on off-diagonal entries; diagonal set to zero",
            "edge_vector": "strict lower triangle, 6,670 unique undirected edges",
        },
        "code_version": args.code_version,
        "script_sha256": sha256_file(Path(__file__)),
        "failures": failures,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if failures:
        metadata["status"] = "failed_validation"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        print(json.dumps(metadata, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    write_qc(qc_path, qc_rows)
    metadata.update(
        {
            "status": "complete",
            "output": str(output_path),
            "output_sha256": sha256_file(output_path),
            "qc_output": str(qc_path),
            "qc_output_sha256": sha256_file(qc_path),
            "regions": len(region_labels),
            "edge_feature_count": int(arrays["edge_features_fisher_z"].shape[1]),
            "array_shapes": {name: list(array.shape) for name, array in arrays.items()},
            "site_counts": dict(Counter(row["site_id"] for row in rows)),
            "class_counts": {
                "asd": int(arrays["label_asd"].sum()),
                "control": int((arrays["label_asd"] == 0).sum()),
            },
            "timepoints": {
                "min": min(row["timepoints"] for row in qc_rows),
                "max": max(row["timepoints"] for row in qc_rows),
            },
            "perfect_correlation_values_clipped": sum(
                row["perfect_correlation_values_clipped"] for row in qc_rows
            ),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
