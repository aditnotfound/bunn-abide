from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.analyze_baselines import AnalysisError, analyze_run, main, paired_bootstrap, recompute_site_metrics, validate_predictions
from scripts.run_baselines import METRIC_FIELDS, PREDICTION_FIELDS, TUNING_FIELDS, WARNING_FIELDS


MODELS = (
    "covariates_l2_logistic",
    "connectome_elastic_net_logistic",
    "combined_elastic_net_logistic",
)
SITES = ("SITE_A", "SITE_B", "SITE_C")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class BaselineAnalysisTests(unittest.TestCase):
    def make_run(self, directory: Path) -> tuple[Path, Path]:
        run_dir = directory / "synthetic_run"
        run_dir.mkdir()
        protocol_path = directory / "protocol.json"
        protocol_path.write_text(
            json.dumps(
                {"evaluation": {"decision_threshold": 0.5, "bootstrap_resamples": 10000, "seed": 17}}
            ),
            encoding="utf-8",
        )
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "run_id": "synthetic_run",
                    "status": "complete",
                    "held_out_sites": list(SITES),
                    "models": list(MODELS),
                }
            ),
            encoding="utf-8",
        )

        probabilities = {
            "covariates_l2_logistic": {
                "SITE_A": [0.1, 0.7, 0.6, 0.9],
                "SITE_B": [0.2, 0.4, 0.4, 0.8],
                "SITE_C": [0.1, 0.8, 0.3, 0.7],
            },
            "connectome_elastic_net_logistic": {
                "SITE_A": [0.1, 0.2, 0.8, 0.9],
                "SITE_B": [0.1, 0.3, 0.7, 0.8],
                "SITE_C": [0.2, 0.3, 0.6, 0.9],
            },
            "combined_elastic_net_logistic": {
                "SITE_A": [0.1, 0.4, 0.7, 0.8],
                "SITE_B": [0.3, 0.4, 0.6, 0.7],
                "SITE_C": [0.1, 0.2, 0.8, 0.9],
            },
        }
        labels = [0, 0, 1, 1]
        predictions: list[dict[str, object]] = []
        for model in MODELS:
            for site in SITES:
                for participant, (label, probability) in enumerate(zip(labels, probabilities[model][site], strict=True)):
                    predictions.append(
                        {
                            "model": model,
                            "outer_fold": site,
                            "held_out_site": site,
                            "subject_id": f"{site}_{participant}",
                            "site_id": site,
                            "label_asd": label,
                            "probability_asd": probability,
                            "predicted_asd": int(probability >= 0.5),
                            "C": 1.0,
                            "l1_ratio": "",
                        }
                    )
        write_csv(run_dir / "predictions.csv", PREDICTION_FIELDS, predictions)
        computed = recompute_site_metrics(validate_predictions(predictions, 0.5))
        metrics = []
        for row in computed:
            metrics.append(
                {
                    "model": row["model"], "outer_fold": row["held_out_site"], "held_out_site": row["held_out_site"],
                    "participants": row["participants"], "asd": row["asd"], "control": row["control"],
                    "balanced_accuracy": row["balanced_accuracy"], "auroc": row["auroc"],
                    "sensitivity": row["sensitivity"], "specificity": row["specificity"], "C": 1.0, "l1_ratio": "",
                }
            )
        write_csv(run_dir / "test_metrics.csv", METRIC_FIELDS, metrics)
        tuning = []
        for model in MODELS:
            for site in SITES:
                tuning.extend(
                    [
                        {"model": model, "outer_fold": site, "held_out_site": site, "C": 0.1, "l1_ratio": "", "inner_mean_site_balanced_accuracy": 0.5, "inner_sites_scored": 2, "selected": "0"},
                        {"model": model, "outer_fold": site, "held_out_site": site, "C": 1.0, "l1_ratio": "", "inner_mean_site_balanced_accuracy": 0.6, "inner_sites_scored": 2, "selected": "1"},
                    ]
                )
        write_csv(run_dir / "tuning_scores.csv", TUNING_FIELDS, tuning)
        write_csv(run_dir / "fit_warnings.csv", WARNING_FIELDS, [])
        return run_dir, protocol_path

    def test_synthetic_analysis_generates_frozen_outputs(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir, protocol_path = self.make_run(Path(temporary))
            output = Path(temporary) / "analysis"
            result = analyze_run(run_dir, protocol_path, output)
            expected = {
                "analysis_manifest.json", "fit_warning_summary.csv", "model_summary.csv",
                "paired_balanced_accuracy_differences.csv", "paired_balanced_accuracy_differences.png",
                "per_site_balanced_accuracy.png", "per_site_metrics.csv", "selected_hyperparameters.csv",
            }
            self.assertEqual(set(result["generated_files"]), expected)
            with (output / "model_summary.csv").open(newline="", encoding="utf-8") as handle:
                summaries = list(csv.DictReader(handle))
            self.assertEqual([row["model"] for row in summaries], list(MODELS))
            with (output / "paired_balanced_accuracy_differences.csv").open(newline="", encoding="utf-8") as handle:
                contrasts = list(csv.DictReader(handle))
            self.assertEqual(len(contrasts), 3)
            self.assertTrue(all(row["bootstrap_resamples"] == "10000" for row in contrasts))
            self.assertTrue((output / "per_site_balanced_accuracy.png").is_file())
            self.assertTrue((output / "paired_balanced_accuracy_differences.png").is_file())

    def test_metric_mismatch_refuses_analysis(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir, protocol_path = self.make_run(Path(temporary))
            metric_path = run_dir / "test_metrics.csv"
            with metric_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["balanced_accuracy"] = "0.123"
            write_csv(metric_path, METRIC_FIELDS, rows)
            with self.assertRaisesRegex(AnalysisError, "differs from recomputed"):
                analyze_run(run_dir, protocol_path, Path(temporary) / "analysis")

    def test_paired_bootstrap_is_deterministic_and_site_paired(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir, _protocol_path = self.make_run(Path(temporary))
            with (run_dir / "predictions.csv").open(newline="", encoding="utf-8") as handle:
                predictions = list(csv.DictReader(handle))
            site_rows = recompute_site_metrics(validate_predictions(predictions, 0.5))
            first = paired_bootstrap(site_rows, 10000, 17)
            second = paired_bootstrap(site_rows, 10000, 17)
            self.assertEqual(first, second)
            self.assertEqual(first[0]["held_out_sites"], 3)
            self.assertEqual(first[0]["left_model"], MODELS[0])
            self.assertEqual(first[0]["right_model"], MODELS[1])

    def test_cli_requires_exact_unblind_acknowledgement(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir, protocol_path = self.make_run(Path(temporary))
            original_argv = sys.argv
            try:
                sys.argv = [
                    "analyze_baselines.py", "--run-dir", str(run_dir),
                    "--protocol", str(protocol_path), "--output-dir", str(Path(temporary) / "analysis"),
                ]
                with self.assertRaisesRegex(SystemExit, "explicit acknowledgement"):
                    main()
            finally:
                sys.argv = original_argv


if __name__ == "__main__":
    unittest.main()
