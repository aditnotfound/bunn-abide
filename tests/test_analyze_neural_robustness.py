from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.analyze_neural_robustness import (
    BASELINE_SITE_METRIC_FIELDS,
    CURVE_FIELDS,
    NeuralAnalysisError,
    REPRESENTATION_FIELDS,
    RUNTIME_SUMMARY_FIELDS,
    SITE_CONFIGURATION_FIELDS,
    SITE_SEED_FIELDS,
    analyze_robustness,
    main,
    normalized_auc,
    robustness_decision,
    sha256_file,
)


SITES = [f"SITE_{index:02d}" for index in range(18)]
SEEDS = [20260803, 20260804, 20260805, 20260806, 20260807]
DENSITIES = [0.0, 0.01, 0.05, 0.1, 0.2]
CONFIGURATIONS = [
    ("identity", 0.0), ("learned_local", 0.0),
    *[(operator, density) for operator in ("gcn", "trivial_bundle", "learned_bunn")
      for density in DENSITIES[1:]],
]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class NeuralRobustnessAnalysisTests(unittest.TestCase):
    def make_step11(self, root: Path) -> tuple[Path, Path, Path]:
        step11_dir = root / "step11"
        step11_dir.mkdir()
        site_seed = []
        site_configuration = []
        representation = []
        runtime = []
        curve_rows = []

        def configuration_value(site_index: int, operator: str, density: float, seed_index: int) -> float:
            site_offset = (site_index - 8.5) * 0.001
            seed_offset = (seed_index - 2) * 0.002
            if operator == "identity":
                base = 0.600
            elif operator == "learned_local":
                base = 0.605
            elif operator == "gcn":
                base = 0.620 - density * 0.20
            elif operator == "trivial_bundle":
                base = 0.612 - density * 0.18
            else:
                base = 0.615 - density * 0.24
            return base + site_offset + seed_offset

        for site_index, site in enumerate(SITES):
            for operator, density in CONFIGURATIONS:
                seed_values = []
                for seed_index, seed in enumerate(SEEDS):
                    value = configuration_value(site_index, operator, density, seed_index)
                    seed_values.append(value)
                    site_seed.append({
                        "operator": operator, "density": density, "seed": seed,
                        "held_out_site": site, "participants": 20 + site_index,
                        "asd": 10, "control": 10 + site_index, "balanced_accuracy": value,
                        "auroc": min(0.99, value + 0.04), "sensitivity": value,
                        "specificity": value, "parameter_count": 1200 if "learned" in operator else 1000,
                    })
                mean_value = sum(seed_values) / len(seed_values)
                site_configuration.append({
                    "operator": operator, "density": density, "held_out_site": site,
                    "seed_count": 5, "participants": 20 + site_index,
                    "mean_balanced_accuracy": mean_value, "sd_balanced_accuracy": 0.003,
                    "mean_auroc": mean_value + 0.04, "mean_sensitivity": mean_value,
                    "mean_specificity": mean_value, "parameter_count": 1200 if "learned" in operator else 1000,
                })
                for layer_index, layer in enumerate(("encoder", "layer_1", "layer_2")):
                    collapse = density * (0.25 if operator == "gcn" else 0.12 if operator == "learned_bunn" else 0.18)
                    representation.append({
                        "operator": operator, "density": density, "held_out_site": site,
                        "layer": layer, "seed_count": 5, "participants_per_seed": 20 + site_index,
                        "normalized_effective_rank": 0.8 - layer_index * collapse,
                        "normalized_dispersion": 0.7 - layer_index * collapse,
                        "mean_pairwise_cosine": 0.2 + layer_index * collapse,
                        "invariant_edge_transport_distance": density * 5,
                    })
            identity = configuration_value(site_index, "identity", 0.0, 2)
            learned_local = configuration_value(site_index, "learned_local", 0.0, 2)
            for operator, anchor, anchor_value in (
                ("gcn", "identity", identity),
                ("trivial_bundle", "identity", identity),
                ("learned_bunn", "learned_local", learned_local),
            ):
                values = [anchor_value] + [configuration_value(site_index, operator, density, 2) for density in DENSITIES[1:]]
                area = normalized_auc(DENSITIES, values)
                curve_rows.append({
                    "curve_operator": operator, "anchor_operator": anchor,
                    "held_out_site": site, "normalized_auc_balanced_accuracy": area,
                    "anchor_balanced_accuracy": anchor_value,
                    "mean_nonzero_density_balanced_accuracy": sum(values[1:]) / 4,
                    "change_from_anchor": area - anchor_value,
                })
        for operator, density in CONFIGURATIONS:
            runtime.append({
                "fit_scope": "final_refit", "operator": operator, "density": density,
                "fits": 90, "total_runtime_seconds": 900.0, "mean_runtime_seconds": 10.0,
                "median_runtime_seconds": 9.0, "maximum_peak_gpu_memory_bytes": 1024 ** 3,
                "resumed_fits": 0,
            })

        write_csv(step11_dir / "site_predictive_curves.csv", CURVE_FIELDS, curve_rows)
        write_csv(step11_dir / "site_seed_metrics.csv", SITE_SEED_FIELDS, site_seed)
        write_csv(step11_dir / "site_configuration_metrics.csv", SITE_CONFIGURATION_FIELDS, site_configuration)
        write_csv(step11_dir / "site_representation_metrics.csv", REPRESENTATION_FIELDS, representation)
        write_csv(step11_dir / "runtime_summary.csv", RUNTIME_SUMMARY_FIELDS, runtime)
        (step11_dir / "decision_summary.json").write_text(
            json.dumps({"all_three_conditions": False}), encoding="utf-8"
        )
        generated_names = [
            "site_predictive_curves.csv", "site_seed_metrics.csv", "site_configuration_metrics.csv",
            "site_representation_metrics.csv", "runtime_summary.csv", "decision_summary.json",
        ]
        manifest = {
            "run_id": "synthetic_run",
            "generated_sha256": {name: sha256_file(step11_dir / name) for name in generated_names},
        }
        (step11_dir / "analysis_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        baseline = root / "baseline.csv"
        write_csv(baseline, BASELINE_SITE_METRIC_FIELDS, [{
            "model": "connectome_elastic_net_logistic", "held_out_site": site,
            "participants": 20 + index, "asd": 10, "control": 10 + index,
            "balanced_accuracy": 0.64 + (index - 8.5) * 0.001,
            "auroc": 0.68, "sensitivity": 0.64, "specificity": 0.64,
        } for index, site in enumerate(SITES)])

        contract = json.loads(
            Path("configs/neural_robustness_analysis_v1.json").read_text(encoding="utf-8")
        )
        contract["input_run_id"] = "synthetic_run"
        contract["input_step11_manifest_sha256"] = sha256_file(step11_dir / "analysis_manifest.json")
        contract["input_baseline_per_site_sha256"] = sha256_file(baseline)
        contract["input_step11_archive_sha256"] = "synthetic_not_archived"
        contract_path = root / "robustness_contract.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        return step11_dir, baseline, contract_path

    def test_synthetic_robustness_generates_fixed_outputs(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            step11, baseline, contract = self.make_step11(root)
            output = root / "robustness"
            def placeholder(_rows: object, path: Path) -> None:
                path.write_bytes(b"synthetic-png-placeholder")
            with (
                patch("scripts.analyze_neural_robustness.plot_site_influence", side_effect=placeholder),
                patch("scripts.analyze_neural_robustness.plot_seed_stability", side_effect=placeholder),
            ):
                result = analyze_robustness(step11, baseline, contract, output)
            expected = {
                "alternative_summaries.csv", "analysis_manifest.json",
                "exhaustive_curve_contrasts.csv", "leave_one_site_out.csv",
                "operator_efficiency.csv", "robustness_decision.json",
                "seed_configuration_rankings.csv", "seed_specific_curves.csv",
                "seed_stability.png", "site_influence.png",
            }
            self.assertEqual(set(result["generated_files"]), expected)
            with (output / "leave_one_site_out.csv").open(newline="", encoding="utf-8") as handle:
                loo = list(csv.DictReader(handle))
            self.assertEqual(len(loo), 54)
            with (output / "seed_specific_curves.csv").open(newline="", encoding="utf-8") as handle:
                seeds = list(csv.DictReader(handle))
            self.assertEqual(len(seeds), 10)
            with (output / "exhaustive_curve_contrasts.csv").open(newline="", encoding="utf-8") as handle:
                exhaustive = list(csv.DictReader(handle))
            self.assertEqual(len(exhaustive), 9)

    def test_changed_step11_artifact_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            step11, baseline, contract = self.make_step11(root)
            path = step11 / "site_predictive_curves.csv"
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(NeuralAnalysisError, "artifact hash mismatch"):
                analyze_robustness(step11, baseline, contract, root / "robustness")

    def test_classification_detects_mixed_sensitivity_without_override(self) -> None:
        loo = [
            {"contrast": "learned_bunn_curve_minus_gcn_curve", "leave_one_out_estimate": -0.01,
             "bootstrap_ci_low": -0.03},
            {"contrast": "learned_bunn_curve_minus_gcn_curve", "leave_one_out_estimate": 0.002,
             "bootstrap_ci_low": -0.02},
        ]
        seeds = [
            {"contrast": "learned_bunn_curve_minus_gcn_curve", "observed_mean_difference": -0.02,
             "bootstrap_ci_low": -0.04},
            {"contrast": "learned_bunn_curve_minus_gcn_curve", "observed_mean_difference": 0.01,
             "bootstrap_ci_low": -0.01},
        ]
        decision = robustness_decision(loo, seeds)
        self.assertEqual(decision["robustness_category"], "mixed_site_and_seed_sensitive_null")
        self.assertFalse(decision["positive_override_allowed"])
        self.assertTrue(decision["confirmatory_step11_result_unchanged"])

    def test_cli_requires_exact_step11_acknowledgement(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            step11, baseline, contract = self.make_step11(root)
            original = sys.argv
            try:
                sys.argv = [
                    "analyze_neural_robustness.py", "--step11-dir", str(step11),
                    "--baseline-per-site", str(baseline), "--contract", str(contract),
                    "--output-dir", str(root / "robustness"),
                ]
                with self.assertRaisesRegex(SystemExit, "exact archived Step 11 run ID"):
                    main()
            finally:
                sys.argv = original


if __name__ == "__main__":
    unittest.main()
