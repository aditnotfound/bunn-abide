from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from src.neural_data import (
    NEURAL_DENSITIES,
    REGION_COUNT,
    NeuralDataError,
    TrainOnlyFeatureStandardizer,
    load_neural_cohort,
    positive_topk_adjacency,
)


def symmetric_connectomes(participants: int) -> torch.Tensor:
    values = torch.zeros((participants, REGION_COUNT, REGION_COUNT), dtype=torch.float32)
    for participant in range(participants):
        for node in range(REGION_COUNT - 1):
            values[participant, node, node + 1] = 1.0 + participant
            values[participant, node + 1, node] = 1.0 + participant
    return values


class NeuralDataTests(unittest.TestCase):
    def test_top_positive_density_has_exact_symmetric_edge_count(self) -> None:
        connectomes = torch.ones((1, REGION_COUNT, REGION_COUNT), dtype=torch.float32)
        connectomes.diagonal(dim1=-2, dim2=-1).zero_()
        for density in NEURAL_DENSITIES:
            adjacency = positive_topk_adjacency(connectomes, density)
            self.assertTrue(torch.equal(adjacency, adjacency.transpose(-1, -2)))
            self.assertTrue(torch.equal(adjacency.diagonal(dim1=-2, dim2=-1), torch.zeros((1, REGION_COUNT))))
            expected = int(np.ceil(density * (REGION_COUNT * (REGION_COUNT - 1) // 2)))
            self.assertEqual(int(adjacency.sum().item() // 2), expected)

    def test_ties_use_canonical_upper_triangle_order(self) -> None:
        connectomes = torch.ones((1, REGION_COUNT, REGION_COUNT), dtype=torch.float32)
        connectomes.diagonal(dim1=-2, dim2=-1).zero_()
        adjacency = positive_topk_adjacency(connectomes, 0.01)
        self.assertEqual(float(adjacency[0, 0, 1]), 1.0)
        self.assertEqual(float(adjacency[0, 0, 67]), 1.0)
        self.assertEqual(float(adjacency[0, 0, 68]), 0.0)

    def test_density_requires_enough_positive_edges(self) -> None:
        connectomes = torch.zeros((1, REGION_COUNT, REGION_COUNT), dtype=torch.float32)
        with self.assertRaisesRegex(NeuralDataError, "positive edges"):
            positive_topk_adjacency(connectomes, 0.01)

    def test_standardizer_is_fit_only_on_supplied_training_features(self) -> None:
        training = torch.zeros((2, REGION_COUNT, REGION_COUNT), dtype=torch.float32)
        training[1] = 2.0
        held_out = torch.full((1, REGION_COUNT, REGION_COUNT), 1000.0)
        scaler = TrainOnlyFeatureStandardizer().fit(training)
        self.assertTrue(torch.allclose(scaler.mean, torch.ones_like(scaler.mean)))
        self.assertTrue(torch.allclose(scaler.scale, torch.ones_like(scaler.scale)))
        self.assertTrue(torch.allclose(scaler.transform(held_out), torch.full_like(held_out, 999.0)))

    def test_npz_and_baseline_table_alignment_are_verified(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            connectomes = symmetric_connectomes(2).numpy()
            np.savez_compressed(
                root / "connectomes.npz",
                subject_id=np.asarray(["a", "b"]),
                site_id=np.asarray(["SITE_A", "SITE_B"]),
                label_asd=np.asarray([0, 1], dtype=np.int8),
                region_labels=np.asarray([f"ROI_{index}" for index in range(REGION_COUNT)]),
                connectomes_fisher_z=connectomes,
            )
            with (root / "baseline.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["connectome_row", "subject_id", "site_id", "label_asd"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"connectome_row": 0, "subject_id": "a", "site_id": "SITE_A", "label_asd": 0},
                        {"connectome_row": 1, "subject_id": "b", "site_id": "SITE_B", "label_asd": 1},
                    ]
                )
            cohort = load_neural_cohort(root / "connectomes.npz", root / "baseline.csv")
            self.assertEqual(cohort.participants, 2)
            self.assertEqual(cohort.connectomes.shape, (2, REGION_COUNT, REGION_COUNT))
            with (root / "baseline.csv").open("a", newline="", encoding="utf-8") as handle:
                handle.write("\n")
            # A malformed alignment is deliberately rejected rather than repaired.
            with (root / "baseline.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["connectome_row", "subject_id", "site_id", "label_asd"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"connectome_row": 0, "subject_id": "wrong", "site_id": "SITE_A", "label_asd": 0},
                        {"connectome_row": 1, "subject_id": "b", "site_id": "SITE_B", "label_asd": 1},
                    ]
                )
            with self.assertRaisesRegex(NeuralDataError, "alignment mismatch"):
                load_neural_cohort(root / "connectomes.npz", root / "baseline.csv")


if __name__ == "__main__":
    unittest.main()
