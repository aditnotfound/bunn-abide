from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.analyze_nonlinear_connectome_baseline import analyze, sha256_file


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class NonlinearBaselineAnalysisTests(unittest.TestCase):
    def test_paired_analysis_uses_all_sites_and_fixed_inputs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            for fold, site, label, score in (
                (0, "A", 0, -1.0), (1, "B", 1, 1.0)
            ):
                fold_dir = run / "folds" / f"{fold:02d}_{site}"
                write_csv(
                    fold_dir / "test_metrics.csv",
                    [{"held_out_site": site, "participants": 1, "balanced_accuracy": 1.0}],
                )
                write_csv(
                    fold_dir / "predictions.csv",
                    [{
                        "held_out_site": site, "subject_id": site, "label_asd": label,
                        "decision_score": score, "predicted_asd": label,
                    }],
                )
            (run / "score_blind_audit.json").write_text(
                json.dumps({
                    "passed": True, "metric_values_disclosed": False,
                    "run_id": "fixture", "sealed_sites": 2,
                }), encoding="utf-8"
            )
            baseline = root / "baseline.csv"
            neural = root / "neural.csv"
            archive = root / "run.tar.gz"
            write_csv(baseline, [
                {"model": "connectome_elastic_net_logistic", "held_out_site": "A", "balanced_accuracy": 0.5},
                {"model": "connectome_elastic_net_logistic", "held_out_site": "B", "balanced_accuracy": 0.75},
            ])
            write_csv(neural, [
                {"held_out_site": site, "curve_operator": operator, "normalized_auc_balanced_accuracy": value}
                for site, value in (("A", 0.6), ("B", 0.7))
                for operator in ("gcn", "learned_bunn")
            ])
            archive.write_bytes(b"fixture")
            contract = root / "contract.json"
            contract.write_text(json.dumps({
                "analysis_version": "fixture_v1",
                "input_run_id": "fixture",
                "expected_sites": 2,
                "expected_participants": 2,
                "source_sha256": {
                    "run_archive": sha256_file(archive),
                    "classical_per_site_metrics": sha256_file(baseline),
                    "neural_site_predictive_curves": sha256_file(neural),
                },
                "contrasts": [
                    {"name": "rbf_svm_minus_connectome_elastic_net"},
                    {"name": "rbf_svm_minus_gcn_curve"},
                    {"name": "rbf_svm_minus_learned_bunn_curve"},
                ],
                "inference": {"bootstrap_seed": 7, "bootstrap_resamples": 100},
            }), encoding="utf-8")
            output = root / "analysis"
            analyze(run, baseline, neural, archive, contract, output)
            with (output / "paired_contrasts.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertAlmostEqual(
                float(rows[0]["equal_site_mean_difference"]), 0.375
            )
            self.assertEqual(int(rows[0]["held_out_sites"]), 2)

    def test_changed_source_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            contract = root / "contract.json"
            contract.write_text(json.dumps({
                "source_sha256": {
                    "run_archive": "bad", "classical_per_site_metrics": "bad",
                    "neural_site_predictive_curves": "bad",
                }
            }), encoding="utf-8")
            for name in ("baseline.csv", "neural.csv", "archive.tar.gz"):
                (root / name).write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                analyze(
                    root / "run", root / "baseline.csv", root / "neural.csv",
                    root / "archive.tar.gz", contract, root / "output",
                )


if __name__ == "__main__":
    unittest.main()
