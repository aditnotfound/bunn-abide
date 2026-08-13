from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.extensions.e1_analysis import (
    DIAGNOSTICS,
    INTERVENTIONS,
    SECONDARY_ENDPOINTS,
    E1AnalysisError,
    aggregate_tables,
    binary_metrics,
    exact_sign_flip_p,
    holm_adjust,
    site_density_rows,
)


class E1AnalysisTests(unittest.TestCase):
    def test_binary_metrics_are_hand_calculated(self) -> None:
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([[0.1, 0.7, 0.4, 0.9]])
        metrics = binary_metrics(labels, probabilities)
        self.assertAlmostEqual(float(metrics["sensitivity"][0]), 0.5)
        self.assertAlmostEqual(float(metrics["specificity"][0]), 0.5)
        self.assertAlmostEqual(float(metrics["balanced_accuracy"][0]), 0.5)
        self.assertAlmostEqual(float(metrics["auroc"][0]), 0.75)

    def test_site_density_aggregation_uses_seed_and_permutation_contract(self) -> None:
        names = [
            "unaltered", "identity_maps", "node_map_shuffle", "random_orthogonal_maps",
            "degree_preserving_topology", "encoded_node_permutation_equivariance",
        ]
        labels = np.array([0, 0, 1, 1])
        probabilities = np.full((6, 2, 3, 4), np.nan, dtype=float)
        diagnostics = np.full((6, 2, 3, 4, 4), np.nan, dtype=float)
        reference = np.array([0.1, 0.4, 0.6, 0.9])
        damaged = np.array([0.1, 0.6, 0.4, 0.9])
        for seed in range(2):
            probabilities[0, seed, 0] = reference
            probabilities[1, seed, 0] = damaged
            diagnostics[0, seed, 0] = 0.0
            diagnostics[1, seed, 0] = 1.0
            for intervention in (2, 3, 4):
                probabilities[intervention, seed, :] = reference
                diagnostics[intervention, seed, :] = 0.0
            probabilities[5, seed, 0] = reference
            diagnostics[5, seed, 0] = 0.0
        rows = site_density_rows(
            site="A", density=0.01, labels=labels,
            probabilities=probabilities, diagnostics=diagnostics,
            intervention_names=names, diagnostic_names=list(DIAGNOSTICS),
        )
        identity = next(row for row in rows if row["intervention"] == "identity_maps")
        shuffled = next(row for row in rows if row["intervention"] == "node_map_shuffle")
        self.assertAlmostEqual(identity["balanced_accuracy_change_pp"], -50.0)
        self.assertAlmostEqual(identity["classification_flip_fraction"], 0.5)
        self.assertAlmostEqual(identity["normalized_effective_rank_change"], 1.0)
        self.assertAlmostEqual(shuffled["balanced_accuracy_change_pp"], 0.0)

    def test_equal_site_aggregation_ignores_participant_count(self) -> None:
        sites = ["A", "B"]
        densities = [0.01, 0.05]
        rows = []
        for site, participants, effect in (("A", 1, 10.0), ("B", 100, 0.0)):
            for density in densities:
                for intervention in INTERVENTIONS:
                    row = {
                        "site": site, "density": density, "intervention": intervention,
                        "participants": participants,
                        "reference_balanced_accuracy": 0.5,
                        "intervention_balanced_accuracy": 0.5 + effect / 100,
                        "balanced_accuracy_change_pp": effect,
                    }
                    for endpoint in SECONDARY_ENDPOINTS:
                        row[endpoint] = effect
                    rows.append(row)
        tables = aggregate_tables(
            pd.DataFrame(rows), sites=sites, densities=densities,
            bootstrap_resamples=100, bootstrap_seed=7,
        )
        self.assertTrue(np.allclose(tables["primary_contrasts"].estimate_pp, 5.0))
        self.assertEqual(len(tables["site_density_effects"]), 16)
        self.assertEqual(len(tables["secondary_endpoints"]), 36)

    def test_exact_sign_flip_and_holm_are_known(self) -> None:
        self.assertAlmostEqual(exact_sign_flip_p(np.array([1.0, 1.0])), 0.5)
        self.assertTrue(np.allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]))

    def test_missing_cell_is_rejected(self) -> None:
        rows = []
        for intervention in INTERVENTIONS:
            row = {
                "site": "A", "density": 0.01, "intervention": intervention,
                "participants": 4, "balanced_accuracy_change_pp": 0.0,
            }
            for endpoint in SECONDARY_ENDPOINTS:
                row[endpoint] = 0.0
            rows.append(row)
        with self.assertRaises(E1AnalysisError):
            aggregate_tables(
                pd.DataFrame(rows[:-1]), sites=["A"], densities=[0.01],
                bootstrap_resamples=10, bootstrap_seed=1,
            )


if __name__ == "__main__":
    unittest.main()
