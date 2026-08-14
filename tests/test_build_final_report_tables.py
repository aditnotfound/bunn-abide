from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.build_final_report_tables import build_tables


class FinalReportTableTests(unittest.TestCase):
    def test_tables_are_generated_from_frozen_records(self) -> None:
        with TemporaryDirectory() as directory:
            outputs = build_tables(
                Path("configs/baseline_protocol.json"),
                Path("configs/neural_full_protocol.json"),
                Path("configs/neural_operator_contract_v2.json"),
                Path("reproducibility/result_snapshot.json"),
                Path(directory),
                Path("reproducibility/nonlinear_baseline_result.json"),
            )
            self.assertEqual({path.name for path in outputs}, {
                "hyperparameters.tex", "weighting_robustness.tex",
                "nonlinear_baseline.tex",
            })
            hyperparameters = (Path(directory) / "hyperparameters.tex").read_text()
            weighting = (Path(directory) / "weighting_robustness.tex").read_text()
            self.assertIn("at most 150 epochs", hyperparameters)
            self.assertIn("2 tuning and 5 final seeds", hyperparameters)
            self.assertIn("+0.0030", weighting)
            self.assertIn("-0.0552", weighting)
            nonlinear = (Path(directory) / "nonlinear_baseline.tex").read_text()
            self.assertIn("+0.0406", nonlinear)
            self.assertIn("Post-hoc RBF-SVM", nonlinear)


if __name__ == "__main__":
    unittest.main()
