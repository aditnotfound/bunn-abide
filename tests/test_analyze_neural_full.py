from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from scripts.analyze_baselines import SITE_METRIC_FIELDS as BASELINE_SITE_METRIC_FIELDS
from scripts.analyze_neural_full import (
    NeuralAnalysisError,
    analyze_run,
    holm_adjust,
    main,
    normalized_auc,
    sha256_file,
)
from scripts.run_neural_full import (
    DIAGNOSTIC_FIELDS,
    METRIC_FIELDS,
    PREDICTION_FIELDS,
    RUNTIME_FIELDS,
    TUNING_FIELDS,
    WARNING_FIELDS,
)


SITES = [f"SITE_{index:02d}" for index in range(18)]
SEEDS = [20260803, 20260804, 20260805, 20260806, 20260807]
CONFIGURATIONS = [
    ("identity", 0.0), ("learned_local", 0.0),
    *[(operator, density) for operator in ("gcn", "trivial_bundle", "learned_bunn")
      for density in (0.01, 0.05, 0.1, 0.2)],
]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class NeuralFullAnalysisTests(unittest.TestCase):
    def make_run(self, root: Path) -> tuple[Path, Path, Path, Path]:
        run_dir = root / "synthetic_run"
        run_dir.mkdir()
        predictions: list[dict[str, object]] = []
        metrics: list[dict[str, object]] = []
        diagnostics: list[dict[str, object]] = []
        tuning: list[dict[str, object]] = []
        runtimes: list[dict[str, object]] = []
        labels = np.asarray([0, 0, 1, 1], dtype=int)

        for site_index, site in enumerate(SITES):
            for operator, density in CONFIGURATIONS:
                tuning.append({
                    "operator": operator, "density": density, "outer_fold": site_index,
                    "held_out_site": site, "candidate_index": 0, "learning_rate": 0.001,
                    "weight_decay": 0.0001, "inner_mean_site_balanced_accuracy": 0.6,
                    "inner_site_score_rows": 8, "selected": 1, "selected_final_epoch": 12,
                })
                runtimes.append({
                    "fit_scope": "final_refit", "operator": operator, "density": density,
                    "outer_fold": site_index, "held_out_site": site, "candidate_index": 0,
                    "inner_validation_fold": -1, "seed": SEEDS[0], "epochs_completed": 12,
                    "runtime_seconds": 1.0 + density, "peak_gpu_memory_bytes": 1024,
                    "resumed": 0,
                })
                for seed_index, seed in enumerate(SEEDS):
                    base = np.asarray([0.20, 0.55, 0.45, 0.80], dtype=float)
                    if operator == "gcn":
                        adjustment = np.asarray([0.0, 0.05, -0.05, 0.0]) * (density / 0.2)
                    elif operator == "trivial_bundle":
                        adjustment = np.asarray([0.0, 0.02, -0.02, 0.0]) * (density / 0.2)
                    elif operator == "learned_bunn":
                        adjustment = np.asarray([0.0, -0.12, 0.12, 0.0]) * (density / 0.2)
                    elif operator == "learned_local":
                        adjustment = np.asarray([0.0, -0.03, 0.03, 0.0])
                    else:
                        adjustment = np.zeros(4)
                    jitter = (site_index - 8.5) * 0.001 + (seed_index - 2) * 0.0005
                    probabilities = np.clip(base + adjustment + jitter, 0.01, 0.99)
                    predicted = probabilities >= 0.5
                    positives = labels == 1
                    negatives = ~positives
                    metric_values = {
                        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
                        "auroc": float(roc_auc_score(labels, probabilities)),
                        "sensitivity": float(predicted[positives].mean()),
                        "specificity": float((~predicted[negatives]).mean()),
                    }
                    metrics.append({
                        "operator": operator, "density": density, "seed": seed,
                        "outer_fold": site_index, "held_out_site": site, "participants": 4,
                        "asd": 2, "control": 2, **metric_values, "learning_rate": 0.001,
                        "weight_decay": 0.0001, "final_epochs": 12,
                        "parameter_count": 1000 + (200 if operator in {"learned_local", "learned_bunn"} else 0),
                    })
                    for subject_index, (label, probability) in enumerate(zip(labels, probabilities, strict=True)):
                        subject_id = f"{site}_{subject_index}"
                        predictions.append({
                            "operator": operator, "density": density, "seed": seed,
                            "outer_fold": site_index, "held_out_site": site, "subject_id": subject_id,
                            "site_id": site, "label_asd": int(label), "probability_asd": probability,
                            "predicted_asd": int(probability >= 0.5),
                        })
                        for layer_index, layer in enumerate(("encoder", "layer_1", "layer_2")):
                            collapse = density * (0.30 if operator == "gcn" else 0.08 if operator == "learned_bunn" else 0.18)
                            diagnostics.append({
                                "operator": operator, "density": density, "seed": seed,
                                "outer_fold": site_index, "held_out_site": site,
                                "subject_id": subject_id, "site_id": site, "layer": layer,
                                "normalized_effective_rank": 0.9 - layer_index * collapse + subject_index * 0.001,
                                "normalized_dispersion": 0.8 - layer_index * collapse,
                                "mean_pairwise_cosine": 0.1 + layer_index * collapse,
                                "invariant_edge_transport_distance": 0.5 - layer_index * collapse / 2,
                            })

        write_csv(run_dir / "predictions.csv", PREDICTION_FIELDS, predictions)
        write_csv(run_dir / "test_metrics.csv", METRIC_FIELDS, metrics)
        write_csv(run_dir / "diagnostics.csv", DIAGNOSTIC_FIELDS, diagnostics)
        write_csv(run_dir / "fit_runtime.csv", RUNTIME_FIELDS, runtimes)
        write_csv(run_dir / "tuning_scores.csv", TUNING_FIELDS, tuning)
        write_csv(run_dir / "fit_warnings.csv", WARNING_FIELDS, [])
        artifact_names = [
            "predictions.csv", "test_metrics.csv", "diagnostics.csv", "fit_runtime.csv",
            "tuning_scores.csv", "fit_warnings.csv",
        ]
        metadata = {
            "run_id": "synthetic_run", "status": "complete", "run_kind": "full",
            "results_embargoed": True, "held_out_sites": SITES,
            "artifact_hashes": {name: sha256_file(run_dir / name) for name in artifact_names},
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

        certificate_path = root / "integrity.json"
        certificate_path.write_text(json.dumps({
            "state": "passed", "run_id": "synthetic_run", "results_remain_embargoed": True,
        }), encoding="utf-8")
        source_contract = json.loads(
            Path("configs/neural_confirmatory_analysis_v1.json").read_text(encoding="utf-8")
        )
        source_contract["input_run_id"] = "synthetic_run"
        contract_path = root / "analysis_contract.json"
        contract_path.write_text(json.dumps(source_contract), encoding="utf-8")
        baseline_path = root / "baseline.csv"
        baseline_rows = []
        for site in SITES:
            baseline_rows.append({
                "model": "connectome_elastic_net_logistic", "held_out_site": site,
                "participants": 4, "asd": 2, "control": 2, "balanced_accuracy": 0.65,
                "auroc": 0.7, "sensitivity": 0.65, "specificity": 0.65,
            })
        write_csv(baseline_path, BASELINE_SITE_METRIC_FIELDS, baseline_rows)
        return run_dir, certificate_path, contract_path, baseline_path

    def test_frozen_curve_and_holm_math(self) -> None:
        self.assertAlmostEqual(normalized_auc([0.0, 0.1, 0.2], [0.5, 0.6, 0.7]), 0.6)
        self.assertEqual(holm_adjust([0.01, 0.04, 0.03, 0.2]), [0.04, 0.09, 0.09, 0.2])

    def test_synthetic_full_analysis_generates_frozen_outputs(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, certificate, contract, baseline = self.make_run(root)
            output = root / "analysis"
            result = analyze_run(run_dir, certificate, contract, baseline, output)
            expected = {
                "analysis_manifest.json", "confirmatory_predictive_contrasts.csv",
                "confirmatory_predictive_contrasts.png", "decision_summary.json",
                "density_specific_contrasts.csv", "fit_warning_summary.csv",
                "predictive_density_curves.png", "representation_contrasts.csv",
                "representation_density_curves.png", "runtime_summary.csv",
                "seed_stability.csv", "selected_hyperparameters.csv",
                "site_configuration_metrics.csv", "site_predictive_curves.csv",
                "site_representation_metrics.csv", "site_seed_metrics.csv",
            }
            self.assertEqual(set(result["generated_files"]), expected)
            with (output / "confirmatory_predictive_contrasts.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5)
            self.assertEqual(sum(int(row["primary"]) for row in rows), 1)
            with (output / "density_specific_contrasts.csv").open(newline="", encoding="utf-8") as handle:
                density_rows = list(csv.DictReader(handle))
            self.assertEqual(len(density_rows), 20)

    def test_changed_metric_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, certificate, contract, baseline = self.make_run(root)
            path = run_dir / "test_metrics.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["balanced_accuracy"] = "0.123"
            write_csv(path, METRIC_FIELDS, rows)
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            metadata["artifact_hashes"]["test_metrics.csv"] = sha256_file(path)
            (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(NeuralAnalysisError, "Runner balanced_accuracy mismatch"):
                analyze_run(run_dir, certificate, contract, baseline, root / "analysis")

    def test_cli_requires_exact_unblind_acknowledgement(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, certificate, contract, baseline = self.make_run(root)
            original = sys.argv
            try:
                sys.argv = [
                    "analyze_neural_full.py", "--run-dir", str(run_dir),
                    "--integrity-certificate", str(certificate), "--analysis-contract", str(contract),
                    "--baseline-per-site", str(baseline), "--output-dir", str(root / "analysis"),
                ]
                with self.assertRaisesRegex(SystemExit, "explicit acknowledgement"):
                    main()
            finally:
                sys.argv = original


if __name__ == "__main__":
    unittest.main()
