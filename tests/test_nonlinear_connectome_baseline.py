from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.audit_nonlinear_connectome_baseline import audit
from scripts.run_baselines import sha256_file, write_json_atomic
from scripts.run_nonlinear_connectome_baseline import (
    MODEL_NAME,
    build_estimator,
    candidates,
    seal_fold,
    select_candidate,
    verified_completed_fold,
)


class NonlinearConnectomeBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = {
            "cohort": {"participants": 2, "sites": 1},
            "model": {
                "feature_count": 4,
                "kernel": "rbf",
                "class_weight": "balanced",
                "C_grid": [0.1, 1.0, 10.0],
                "gamma_multipliers_over_feature_count": [0.25, 1.0, 4.0],
            },
            "expected_counts": {
                "candidates_per_outer_site": 9,
                "inner_fits": 36,
                "held_out_prediction_rows": 2,
                "held_out_metric_rows": 1,
            },
        }

    def test_grid_has_exact_declared_gamma_values(self) -> None:
        rows = candidates(self.protocol)
        self.assertEqual(len(rows), 9)
        self.assertEqual(
            sorted({row["gamma"] for row in rows}),
            [0.25 / 4, 1.0 / 4, 4.0 / 4],
        )

    def test_scaler_is_fit_only_on_rows_passed_to_fit(self) -> None:
        estimator = build_estimator(candidates(self.protocol)[0], self.protocol)
        features = np.asarray(
            [[0.0, 1.0, 2.0, 3.0], [2.0, 3.0, 4.0, 5.0], [999.0] * 4]
        )
        estimator.fit(features[:2], np.asarray([0, 1]))
        self.assertTrue(np.allclose(estimator.named_steps["scaler"].mean_, [1, 2, 3, 4]))

    def test_tie_break_prefers_lower_c_then_lower_gamma(self) -> None:
        rows = [
            {"inner_mean_site_balanced_accuracy": 0.6, "C": 1.0, "gamma_multiplier": 1.0},
            {"inner_mean_site_balanced_accuracy": 0.6, "C": 0.1, "gamma_multiplier": 4.0},
            {"inner_mean_site_balanced_accuracy": 0.6, "C": 0.1, "gamma_multiplier": 0.25},
        ]
        self.assertEqual(select_candidate(rows)["gamma_multiplier"], 0.25)

    def test_seal_and_score_blind_audit(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(self.protocol), encoding="utf-8")
            run_dir = root / "run"
            (run_dir / "folds").mkdir(parents=True)
            metadata = {
                "run_id": "smoke",
                "run_kind": "timing_smoke",
                "protocol_sha256": sha256_file(protocol_path),
                "held_out_sites": ["SITE_A"],
                "site_to_outer_fold": {"SITE_A": 0},
            }
            write_json_atomic(run_dir / "metadata.json", metadata)
            candidate = {"C": 0.1, "gamma_multiplier": 0.25, "gamma": 0.0625}
            tuning = []
            for index, row in enumerate(candidates(self.protocol)):
                tuning.append(
                    {
                        "model": MODEL_NAME,
                        "outer_fold": 0,
                        "held_out_site": "SITE_A",
                        **row,
                        "inner_mean_site_balanced_accuracy": 0.5,
                        "inner_sites_scored": 4,
                        "selected": int(index == 0),
                    }
                )
            inner = []
            for inner_fold in range(4):
                inner.append(
                    {
                        "model": MODEL_NAME,
                        "outer_fold": 0,
                        "held_out_site": "SITE_A",
                        "inner_validation_fold": inner_fold,
                        "site_id": f"INNER_{inner_fold}",
                        "participants": 2,
                        "balanced_accuracy": 0.5,
                        **candidate,
                    }
                )
            predictions = [
                {
                    "model": MODEL_NAME, "outer_fold": 0, "held_out_site": "SITE_A",
                    "subject_id": "A", "site_id": "SITE_A", "label_asd": 0,
                    "decision_score": -1.0, "predicted_asd": 0, **candidate,
                },
                {
                    "model": MODEL_NAME, "outer_fold": 0, "held_out_site": "SITE_A",
                    "subject_id": "B", "site_id": "SITE_A", "label_asd": 1,
                    "decision_score": 1.0, "predicted_asd": 1, **candidate,
                },
            ]
            metrics = [
                {
                    "model": MODEL_NAME, "outer_fold": 0, "held_out_site": "SITE_A",
                    "participants": 2, "asd": 1, "control": 1,
                    "balanced_accuracy": 1.0, "auroc": 1.0,
                    "sensitivity": 1.0, "specificity": 1.0, **candidate,
                }
            ]
            seal_fold(
                run_dir,
                0,
                "SITE_A",
                {
                    "predictions.csv": predictions,
                    "test_metrics.csv": metrics,
                    "tuning_scores.csv": tuning,
                    "inner_site_scores.csv": inner,
                },
            )
            fold = run_dir / "folds" / "00_SITE_A"
            self.assertTrue(verified_completed_fold(fold, 0, "SITE_A"))
            result = audit(run_dir, protocol_path)
            self.assertTrue(result["passed"])
            self.assertFalse(result["metric_values_disclosed"])


if __name__ == "__main__":
    unittest.main()
