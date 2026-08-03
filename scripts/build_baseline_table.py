"""Create and validate the subject-level table for Step 7 baseline models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="data/processed/abide_i_analysis_manifest.csv"
    )
    parser.add_argument(
        "--connectomes", default="data/processed/abide_i_connectomes_fisher_z.npz"
    )
    parser.add_argument(
        "--connectome-metadata",
        default="data/processed/abide_i_connectome_metadata.json",
    )
    parser.add_argument(
        "--connectome-qc", default="data/processed/abide_i_connectome_qc.csv"
    )
    parser.add_argument(
        "--output", default="data/processed/abide_i_baseline_table.csv"
    )
    parser.add_argument(
        "--summary-output", default="data/processed/abide_i_baseline_table_summary.json"
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_optional_float(value: str) -> float | None:
    text = value.strip().lower()
    if not text or text in {"na", "nan", "n/a"}:
        return None
    return float(value)


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    connectome_path = Path(args.connectomes)
    metadata_path = Path(args.connectome_metadata)
    qc_path = Path(args.connectome_qc)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)

    manifest = read_csv(manifest_path)
    qc_by_subject = {row["subject_id"]: row for row in read_csv(qc_path)}
    if len(qc_by_subject) != len(manifest):
        raise ValueError("QC table does not contain exactly one row per manifest participant")
    data = np.load(connectome_path, allow_pickle=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    subject_ids = [str(value) for value in data["subject_id"]]
    site_ids = [str(value) for value in data["site_id"]]
    labels = [int(value) for value in data["label_asd"]]
    if len(subject_ids) != len(manifest):
        raise ValueError("Connectome and manifest row counts differ")
    if metadata.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Connectome metadata does not reference this analysis manifest")

    table_rows: list[dict[str, object]] = []
    for index, (row, subject_id, site_id, label) in enumerate(
        zip(manifest, subject_ids, site_ids, labels, strict=True)
    ):
        if row["subject_id"] != subject_id:
            raise ValueError(f"Subject order mismatch at connectome row {index}")
        if row["site_id"] != site_id or int(row["label_asd"]) != label:
            raise ValueError(f"Site or label mismatch at connectome row {index}")
        qc = qc_by_subject.get(subject_id)
        if qc is None:
            raise ValueError(f"Missing QC row for {subject_id}")
        table_rows.append(
            {
                "connectome_row": index,
                "subject_id": subject_id,
                "site_id": site_id,
                "label_asd": label,
                "age_at_scan": parse_optional_float(row["age_at_scan"]),
                "sex_code": row["sex"].strip() or None,
                "mean_framewise_displacement": parse_optional_float(row["func_mean_fd"]),
                "scan_length_timepoints": int(qc["timepoints"]),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(table_rows[0])
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)

    missing_counts = {
        field: sum(row[field] in {None, ""} for row in table_rows)
        for field in ["age_at_scan", "sex_code", "mean_framewise_displacement", "scan_length_timepoints"]
    }
    summary = {
        "analysis_manifest": str(manifest_path),
        "analysis_manifest_sha256": sha256_file(manifest_path),
        "connectome_artifact": str(connectome_path),
        "connectome_sha256": sha256_file(connectome_path),
        "connectome_metadata_sha256": sha256_file(metadata_path),
        "connectome_qc_sha256": sha256_file(qc_path),
        "baseline_table": str(output_path),
        "baseline_table_sha256": sha256_file(output_path),
        "rows": len(table_rows),
        "connectome_feature_count": int(data["edge_features_fisher_z"].shape[1]),
        "missing_counts": missing_counts,
        "sex_counts": {
            code: sum(row["sex_code"] == code for row in table_rows)
            for code in sorted({str(row["sex_code"]) for row in table_rows if row["sex_code"]})
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
