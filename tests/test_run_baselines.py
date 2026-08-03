from __future__ import annotations

import unittest
import json
import os
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from scripts.run_baselines import (
    COVARIATE_NUMERIC,
    apply_fast_smoke_overrides,
    build_pipeline,
    initialise_or_resume_run,
    mean_site_balanced_accuracy,
    select_candidate,
    verify_completed_fold,
    write_fold_artifacts,
)
from scripts.check_baseline_run import status_report


class BaselineRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = {
            "models": {
                "covariates_l2_logistic": {
                    "class_weight": "balanced",
                    "solver": "lbfgs",
                    "C_grid": [1.0],
                },
                "connectome_elastic_net_logistic": {
                    "class_weight": "balanced",
                    "solver": "saga",
                    "C_grid": [0.3],
                    "l1_ratio_grid": [0.5],
                    "max_iter": 100,
                    "retry_max_iter": 200,
                },
                "combined_elastic_net_logistic": {
                    "class_weight": "balanced",
                    "solver": "saga",
                    "C_grid": [0.3],
                    "l1_ratio_grid": [0.5],
                    "max_iter": 100,
                    "retry_max_iter": 200,
                },
            }
        }

    def test_covariate_imputer_is_fit_on_training_rows_only(self) -> None:
        # The final row represents a held-out participant. Its huge value must
        # not influence the training imputer's fitted median.
        X = pd.DataFrame(
            {
                "age_at_scan": [10.0, 20.0, 30.0, 999999.0],
                "mean_framewise_displacement": [0.1, np.nan, 0.3, 999999.0],
                "scan_length_timepoints": [100, 110, 120, 999999],
                "sex_code": ["1", "2", "1", "2"],
            }
        )
        estimator = build_pipeline(
            "covariates_l2_logistic", {"C": 1.0}, [], self.protocol, random_state=7
        )
        estimator.fit(X.iloc[:3], np.asarray([0, 1, 0]))
        imputer = (
            estimator.named_steps["preprocess"]
            .named_transformers_["covariate_numeric"]
            .named_steps["imputer"]
        )
        self.assertTrue(np.allclose(imputer.statistics_, [20.0, 0.2, 110.0]))

    def test_inner_score_weights_sites_equally_not_participants(self) -> None:
        labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
        predictions = np.asarray([0, 1, 1, 1, 1, 1, 1, 1])
        sites = np.asarray(["small", "small", "large", "large", "large", "large", "large", "large"])
        mean_score, rows = mean_site_balanced_accuracy(labels, predictions, sites)
        self.assertEqual([row["site_id"] for row in rows], ["large", "small"])
        self.assertAlmostEqual(mean_score, (0.5 + 1.0) / 2.0)

    def test_hyperparameter_tie_break_prefers_lower_regularization_values(self) -> None:
        selected = select_candidate(
            [
                {"C": 3.0, "l1_ratio": 0.9, "inner_mean_site_balanced_accuracy": 0.75},
                {"C": 0.3, "l1_ratio": 0.9, "inner_mean_site_balanced_accuracy": 0.75},
                {"C": 0.3, "l1_ratio": 0.1, "inner_mean_site_balanced_accuracy": 0.75},
            ],
            "connectome_elastic_net_logistic",
        )
        self.assertEqual(selected["C"], 0.3)
        self.assertEqual(selected["l1_ratio"], 0.1)

    def test_covariate_selection_accepts_blank_saved_l1_ratio(self) -> None:
        selected = select_candidate(
            [
                {"C": 0.1, "l1_ratio": "", "inner_mean_site_balanced_accuracy": 0.75},
                {"C": 1.0, "l1_ratio": "", "inner_mean_site_balanced_accuracy": 0.75},
            ],
            "covariates_l2_logistic",
        )
        self.assertEqual(selected["C"], 0.1)

    def test_numeric_columns_are_explicit_and_stable(self) -> None:
        self.assertEqual(
            COVARIATE_NUMERIC,
            ["age_at_scan", "mean_framewise_displacement", "scan_length_timepoints"],
        )

    def test_fast_smoke_override_does_not_mutate_frozen_protocol(self) -> None:
        effective, override = apply_fast_smoke_overrides(self.protocol)
        self.assertIsNotNone(override)
        self.assertEqual(effective["models"]["connectome_elastic_net_logistic"]["C_grid"], [0.3])
        self.assertEqual(effective["models"]["connectome_elastic_net_logistic"]["max_iter"], 100)
        self.assertNotIn("retry_max_iter", effective["models"]["connectome_elastic_net_logistic"])
        self.assertEqual(self.protocol["models"]["connectome_elastic_net_logistic"]["max_iter"], 100)
        self.assertEqual(self.protocol["models"]["connectome_elastic_net_logistic"]["retry_max_iter"], 200)

    def test_fold_checkpoint_is_verified_and_resume_refuses_changed_contract(self) -> None:
        metadata = {
            "run_id": "test-run",
            "run_kind": "smoke",
            "status": "running",
            "started_utc": "2026-08-03T00:00:00+00:00",
            "code_version": "test-commit",
            "protocol_sha256": "protocol-hash",
            "frozen_input_hashes": {"table": "hash"},
            "sources": {"table": "table.csv"},
            "models": ["covariates_l2_logistic"],
            "smoke_override": None,
            "held_out_sites": ["SITE_A"],
            "site_to_outer_fold": {"0": "SITE_A"},
            "participants_in_dataset": 2,
            "edge_features": 1,
        }
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "test-run"
            self.assertEqual(initialise_or_resume_run(run_dir, metadata, resume=False), [])
            checkpoint = write_fold_artifacts(
                run_dir, 0, "SITE_A", [], [], [], [], []
            )
            self.assertTrue(verify_completed_fold(checkpoint, 0, "SITE_A"))
            self.assertEqual(initialise_or_resume_run(run_dir, metadata, resume=True), ["SITE_A"])
            resumed_status = json.loads((run_dir / "status.json").read_text())
            self.assertEqual(resumed_status["pid"], os.getpid())
            changed = deepcopy(metadata)
            changed["code_version"] = "different-commit"
            with self.assertRaises(ValueError):
                initialise_or_resume_run(run_dir, changed, resume=True)

    def test_status_report_reads_checkpoint_progress(self) -> None:
        metadata = {
            "run_id": "status-run",
            "status": "running",
        }
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "status-run"
            run_dir.mkdir()
            (run_dir / "metadata.json").write_text(json.dumps(metadata))
            (run_dir / "status.json").write_text(
                json.dumps(
                    {
                        "state": "running",
                        "last_updated_utc": "2026-08-03T00:00:00+00:00",
                        "completed_site_count": 1,
                        "total_sites": 2,
                        "completed_sites": ["SITE_A"],
                        "current_site": "SITE_B",
                        "current_model": "connectome_elastic_net_logistic",
                        "current_stage": "inner_tuning",
                        "current_candidate": {"C": 0.03, "l1_ratio": 0.1},
                        "pid": None,
                    }
                )
            )
            report = status_report(run_dir, stale_minutes=10_000_000)
            self.assertEqual(report["state"], "running")
            self.assertEqual(report["completed_sites"], ["SITE_A"])
            self.assertEqual(report["current_site"], "SITE_B")


if __name__ == "__main__":
    unittest.main()
