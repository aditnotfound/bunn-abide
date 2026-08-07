from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.audit_baseline_run import AuditFailure, audit_run
from scripts.run_baselines import (
    FOLD_ARTIFACT_FIELDS,
    INNER_SITE_FIELDS,
    METRIC_FIELDS,
    PREDICTION_FIELDS,
    TUNING_FIELDS,
    WARNING_FIELDS,
    sha256_file,
    write_csv,
    write_fold_artifacts,
)


class BaselineAuditTests(unittest.TestCase):
    @staticmethod
    def read_rows(path: Path) -> list[dict[str, str]]:
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def row_count(path: Path) -> int:
        with path.open(newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))

    def make_valid_run(self, root: Path) -> tuple[Path, Path, Path]:
        table = root / "baseline_table.csv"
        protocol = root / "protocol.json"
        run_dir = root / "run"
        table_rows = [
            {
                "connectome_row": 0, "subject_id": "SUBJECT_A", "site_id": "SITE_A", "label_asd": 0,
                "age_at_scan": 10, "sex_code": "1", "mean_framewise_displacement": 0.1,
                "scan_length_timepoints": 100,
            },
            {
                "connectome_row": 1, "subject_id": "SUBJECT_B", "site_id": "SITE_A", "label_asd": 1,
                "age_at_scan": 11, "sex_code": "2", "mean_framewise_displacement": 0.2,
                "scan_length_timepoints": 100,
            },
        ]
        write_csv(table, table_rows, [
            "connectome_row", "subject_id", "site_id", "label_asd", "age_at_scan", "sex_code",
            "mean_framewise_displacement", "scan_length_timepoints",
        ])
        model = "covariates_l2_logistic"
        protocol.write_text(json.dumps({"models": {model: {"C_grid": [1.0]}}}))

        prediction_rows = [
            {
                "model": model, "outer_fold": 0, "held_out_site": "SITE_A",
                "subject_id": row["subject_id"], "site_id": "SITE_A",
                "label_asd": row["label_asd"], "probability_asd": 0.5,
                "predicted_asd": row["label_asd"], "C": 1.0, "l1_ratio": "",
            }
            for row in table_rows
        ]
        metric_rows = [{
            "model": model, "outer_fold": 0, "held_out_site": "SITE_A",
            "participants": 2, "asd": 1, "control": 1, "balanced_accuracy": 0.5,
            "auroc": 0.5, "sensitivity": 0.5, "specificity": 0.5, "C": 1.0, "l1_ratio": "",
        }]
        tuning_rows = [{
            "model": model, "outer_fold": 0, "held_out_site": "SITE_A", "C": 1.0,
            "l1_ratio": "", "inner_mean_site_balanced_accuracy": 0.5,
            "inner_sites_scored": 0, "selected": 1,
        }]
        fold = write_fold_artifacts(
            run_dir, 0, "SITE_A", prediction_rows, metric_rows, tuning_rows, [], []
        )
        aggregate = {
            "predictions.csv": prediction_rows,
            "test_metrics.csv": metric_rows,
            "tuning_scores.csv": tuning_rows,
            "inner_site_scores.csv": [],
            "fit_warnings.csv": [],
        }
        for filename, fields in FOLD_ARTIFACT_FIELDS.items():
            write_csv(run_dir / filename, aggregate[filename], fields)
        metadata = {
            "run_id": "run", "status": "complete", "participants_in_dataset": 2,
            "held_out_sites": ["SITE_A"], "models": [model],
            "site_to_outer_fold": {"0": "SITE_A"},
            "sources": {"baseline_table": str(table), "outer_splits": str(table), "inner_splits": str(table)},
            "frozen_input_hashes": {"expected": {
                "baseline table": sha256_file(table), "outer assignments": sha256_file(table),
                "inner assignments": sha256_file(table),
            }},
            "artifact_hashes": {name: sha256_file(run_dir / name) for name in aggregate},
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata))
        (run_dir / "status.json").write_text(json.dumps({
            "state": "complete", "completed_site_count": 1, "completed_sites": ["SITE_A"],
        }))
        self.assertTrue(fold.exists())
        return run_dir, protocol, table

    def refresh_hashes(self, run_dir: Path) -> None:
        fold_dir = run_dir / "folds" / "00_SITE_A"
        completion_path = fold_dir / "complete.json"
        completion = json.loads(completion_path.read_text())
        completion["artifact_hashes"] = {
            name: sha256_file(fold_dir / name) for name in FOLD_ARTIFACT_FIELDS
        }
        completion["row_counts"] = {
            name: self.row_count(fold_dir / name) for name in FOLD_ARTIFACT_FIELDS
        }
        completion_path.write_text(json.dumps(completion))
        metadata_path = run_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["artifact_hashes"] = {
            name: sha256_file(run_dir / name) for name in FOLD_ARTIFACT_FIELDS
        }
        metadata_path.write_text(json.dumps(metadata))

    def audit(self, run_dir: Path, protocol: Path, table: Path) -> dict[str, object]:
        return audit_run(run_dir, protocol, table, run_dir.parent)

    def test_valid_synthetic_run_passes_without_metric_values_in_report(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir, protocol, table = self.make_valid_run(Path(directory))
            report = self.audit(run_dir, protocol, table)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["row_counts"]["predictions.csv"], 2)
        self.assertNotIn("balanced_accuracy", json.dumps(report))

    def test_corrupted_fold_hash_fails(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir, protocol, table = self.make_valid_run(Path(directory))
            with (run_dir / "folds" / "00_SITE_A" / "predictions.csv").open("a") as handle:
                handle.write("corruption\n")
            with self.assertRaises(AuditFailure) as raised:
                self.audit(run_dir, protocol, table)
        self.assertIn("fold_manifest_or_hash", raised.exception.codes)

    def test_duplicate_prediction_fails_after_hashes_are_refreshed(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir, protocol, table = self.make_valid_run(Path(directory))
            for path in [
                run_dir / "predictions.csv",
                run_dir / "folds" / "00_SITE_A" / "predictions.csv",
            ]:
                rows = self.read_rows(path)
                rows[1]["subject_id"] = "SUBJECT_A"
                write_csv(path, rows, PREDICTION_FIELDS)
            self.refresh_hashes(run_dir)
            with self.assertRaises(AuditFailure) as raised:
                self.audit(run_dir, protocol, table)
        self.assertTrue({"prediction_duplicate", "prediction_subject_coverage"} & set(raised.exception.codes))

    def test_invalid_probability_fails_after_hashes_are_refreshed(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir, protocol, table = self.make_valid_run(Path(directory))
            for path in [
                run_dir / "predictions.csv",
                run_dir / "folds" / "00_SITE_A" / "predictions.csv",
            ]:
                rows = self.read_rows(path)
                rows[0]["probability_asd"] = "1.5"
                write_csv(path, rows, PREDICTION_FIELDS)
            self.refresh_hashes(run_dir)
            with self.assertRaises(AuditFailure) as raised:
                self.audit(run_dir, protocol, table)
        self.assertIn("prediction_range", raised.exception.codes)


if __name__ == "__main__":
    unittest.main()
