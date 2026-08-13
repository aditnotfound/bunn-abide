from __future__ import annotations

import unittest

import numpy as np

from src.extensions.e2_analysis import binary_metrics, exact_sign_flip_p, paired_bootstrap


class E2AnalysisTests(unittest.TestCase):
    def test_binary_metrics_hand_calculation(self) -> None:
        result = binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.7, 0.6, 0.9]))
        self.assertAlmostEqual(result["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(result["auroc"], 0.75)
        self.assertAlmostEqual(result["brier_score"], (0.01 + 0.49 + 0.16 + 0.01) / 4)

    def test_bootstrap_is_deterministic_and_paired(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(paired_bootstrap(values, draws=1000, seed=7), paired_bootstrap(values, draws=1000, seed=7))

    def test_exact_sign_flip_has_expected_resolution_for_ten_positive_pairs(self) -> None:
        self.assertAlmostEqual(exact_sign_flip_p(np.ones(10)), 2 / 1024)


if __name__ == "__main__":
    unittest.main()
