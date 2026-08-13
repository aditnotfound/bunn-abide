from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.extensions.e2_synthetic import generate_e2_dataset
from src.extensions.e2_training import audit_e2_cell, run_e2_cell


class E2TrainingTests(unittest.TestCase):
    def test_smoke_cell_never_materializes_test_predictions(self) -> None:
        dataset = generate_e2_dataset(
            family="S1_recoverable_geometry", seed=21, samples=40, nodes=12,
        )
        with tempfile.TemporaryDirectory() as temporary:
            cell = Path(temporary) / "cell"
            run_e2_cell(
                dataset=dataset, operator="learned_bunn", model_seed=99, cell_dir=cell,
                protocol_sha256="a" * 64, learning_rate=1e-3, weight_decay=1e-4,
                batch_size=8, maximum_epochs=2, minimum_epochs=2, patience=2,
                gradient_clip=5.0, device=torch.device("cpu"), evaluate_test=False, resume=False,
            )
            audit = audit_e2_cell(cell, require_predictions=False)
            self.assertEqual(audit["state"], "audit_passed_score_blind")
            self.assertFalse((cell / "test_predictions.npz").exists())

    def test_scientific_cell_predictions_are_aligned_and_audited(self) -> None:
        dataset = generate_e2_dataset(
            family="S0_no_geometry", seed=22, samples=40, nodes=12,
        )
        with tempfile.TemporaryDirectory() as temporary:
            cell = Path(temporary) / "cell"
            run_e2_cell(
                dataset=dataset, operator="gcn", model_seed=100, cell_dir=cell,
                protocol_sha256="b" * 64, learning_rate=1e-3, weight_decay=1e-4,
                batch_size=8, maximum_epochs=2, minimum_epochs=2, patience=2,
                gradient_clip=5.0, device=torch.device("cpu"), evaluate_test=True, resume=False,
            )
            audit = audit_e2_cell(cell, require_predictions=True)
            self.assertIsNotNone(audit["prediction_sha256"])
            manifest = json.loads((cell / "cell_manifest.json").read_text())
            self.assertEqual(manifest["state"], "sealed_score_blind")


if __name__ == "__main__":
    unittest.main()
