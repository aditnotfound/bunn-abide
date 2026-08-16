from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from src.neural_data import REGION_COUNT, NeuralCohort
from src.neural_full_training import (
    build_fit_tensors,
    selected_candidate_index,
    selected_final_epoch,
    train_early_stopped,
)
from src.neural_models import NeuralArchitecture


def cohort() -> NeuralCohort:
    generator = np.random.default_rng(11)
    matrices = []
    for _ in range(12):
        values = generator.normal(size=(REGION_COUNT, REGION_COUNT)).astype(np.float32)
        values = (values + values.T) / 2
        np.fill_diagonal(values, 0.0)
        matrices.append(values)
    return NeuralCohort(
        subject_ids=np.asarray([f"S{index}" for index in range(12)]),
        site_ids=np.asarray([f"SITE_{index // 2}" for index in range(12)]),
        labels=np.asarray([index % 2 for index in range(12)], dtype=np.int64),
        region_labels=np.asarray([f"ROI_{index}" for index in range(REGION_COUNT)]),
        connectomes=np.stack(matrices),
    )


class FullNeuralTrainingTests(unittest.TestCase):
    def test_fit_tensor_scaler_excludes_evaluation_rows(self) -> None:
        data = cohort()
        tensors = build_fit_tensors(data, np.arange(8), np.arange(8, 12), 0.0, torch.device("cpu"))
        altered = data.connectomes.copy()
        altered[8:] += 1000.0
        altered[:, np.arange(REGION_COUNT), np.arange(REGION_COUNT)] = 0.0
        modified = NeuralCohort(data.subject_ids, data.site_ids, data.labels, data.region_labels, altered)
        second = build_fit_tensors(modified, np.arange(8), np.arange(8, 12), 0.0, torch.device("cpu"))
        self.assertTrue(torch.equal(tensors.scaler_mean, second.scaler_mean))
        self.assertTrue(torch.equal(tensors.scaler_scale, second.scaler_scale))

    def test_candidate_tie_break_and_epoch_rule_are_deterministic(self) -> None:
        rows = [
            {"candidate_index": 0, "inner_mean_site_balanced_accuracy": 0.6, "weight_decay": 1e-5, "learning_rate": 3e-4},
            {"candidate_index": 1, "inner_mean_site_balanced_accuracy": 0.6, "weight_decay": 1e-4, "learning_rate": 1e-3},
            {"candidate_index": 2, "inner_mean_site_balanced_accuracy": 0.6, "weight_decay": 1e-4, "learning_rate": 3e-4},
        ]
        self.assertEqual(selected_candidate_index(rows), 2)
        self.assertEqual(selected_final_epoch([1, 2, 3, 4]), 3)

    def test_interrupted_early_stop_fit_resumes_to_uninterrupted_state(self) -> None:
        data = cohort()
        tensors = build_fit_tensors(data, np.arange(8), np.arange(8, 12), 0.0, torch.device("cpu"))
        architecture = NeuralArchitecture(
            input_dim=REGION_COUNT, hidden_dim=8, layers=1,
            bundles=2, channels=2, dropout=0.0,
        )
        common = dict(
            immutable_contract={"fit": "test"}, operator="identity", architecture=architecture,
            tensors=tensors, learning_rate=1e-3, weight_decay=1e-4, batch_size=4,
            maximum_epochs=3, minimum_epochs=3, patience=20, minimum_delta=1e-4,
            gradient_clip_norm=5.0, seed=17, device=torch.device("cpu"),
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            uninterrupted = train_early_stopped(checkpoint_path=root / "full.pt", resume=False, **common)

            def interrupt_first(row: dict[str, object]) -> None:
                if row["epoch"] == 0:
                    raise InterruptedError("test interruption")

            with self.assertRaises(InterruptedError):
                train_early_stopped(
                    checkpoint_path=root / "resumed.pt", resume=False,
                    progress=interrupt_first, **common,
                )
            resumed = train_early_stopped(checkpoint_path=root / "resumed.pt", resume=True, **common)
            self.assertEqual(uninterrupted["best_epoch"], resumed["best_epoch"])
            self.assertEqual(uninterrupted["history"], resumed["history"])
            for key in uninterrupted["best_model_state"]:
                self.assertTrue(torch.equal(uninterrupted["best_model_state"][key], resumed["best_model_state"][key]))


if __name__ == "__main__":
    unittest.main()
