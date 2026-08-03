from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.run_baselines import (
    COVARIATE_NUMERIC,
    build_pipeline,
    mean_site_balanced_accuracy,
    select_candidate,
)


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


if __name__ == "__main__":
    unittest.main()
