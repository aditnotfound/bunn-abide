from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from src.neural_data import REGION_COUNT, NeuralCohort
from src.neural_models import NeuralArchitecture
from src.neural_training import (
    PilotPartitions,
    PilotTrainingError,
    build_pilot_tensors,
    load_pilot_partitions,
    run_pilot_cell,
)


def synthetic_cohort(participants: int = 8) -> NeuralCohort:
    connectomes = np.ones((participants, REGION_COUNT, REGION_COUNT), dtype=np.float32)
    for index in range(participants):
        np.fill_diagonal(connectomes[index], 0.0)
    return NeuralCohort(
        subject_ids=np.asarray([f"S{index}" for index in range(participants)]),
        site_ids=np.asarray([f"SITE_{index // 2}" for index in range(participants)]),
        labels=np.asarray([index % 2 for index in range(participants)], dtype=np.int64),
        region_labels=np.asarray([f"ROI_{index}" for index in range(REGION_COUNT)]),
        connectomes=connectomes,
    )


class NeuralTrainingTests(unittest.TestCase):
    def test_pilot_partitions_exclude_outer_test_site(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            subject_ids = [f"S{index}" for index in range(8)]
            sites = ["SITE_A", "SITE_A", "SITE_B", "SITE_B", "SITE_C", "SITE_C", "SITE_D", "SITE_D"]
            labels = [0, 1, 0, 1, 0, 1, 0, 1]
            with (root / "table.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["connectome_row", "subject_id", "site_id", "label_asd"])
                writer.writeheader()
                writer.writerows(
                    {"connectome_row": index, "subject_id": subject_ids[index], "site_id": sites[index], "label_asd": labels[index]}
                    for index in range(8)
                )
            with (root / "outer.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["outer_fold", "held_out_site", "subject_id", "site_id", "label_asd", "role"])
                writer.writeheader()
                writer.writerows(
                    {"outer_fold": 0, "held_out_site": site, "subject_id": subject, "site_id": site, "label_asd": label, "role": "test"}
                    for subject, site, label in zip(subject_ids, sites, labels, strict=True)
                )
            with (root / "inner.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["outer_fold", "held_out_site", "subject_id", "site_id", "label_asd", "inner_validation_fold"])
                writer.writeheader()
                writer.writerows(
                    {"outer_fold": 0, "held_out_site": "SITE_A", "subject_id": subject_ids[index], "site_id": sites[index], "label_asd": labels[index], "inner_validation_fold": {"SITE_B": 0, "SITE_C": 1, "SITE_D": 2}[sites[index]]}
                    for index in range(2, 8)
                )
            partitions = load_pilot_partitions(root / "table.csv", root / "outer.csv", root / "inner.csv", "SITE_A", 0)
            self.assertEqual(set(partitions.test_indices.tolist()), {0, 1})
            self.assertEqual(set(partitions.validation_indices.tolist()), {2, 3})
            self.assertEqual(set(partitions.fitting_indices.tolist()), {4, 5, 6, 7})

    def test_pilot_tensors_fit_scaler_without_materializing_test_graphs(self) -> None:
        cohort = synthetic_cohort()
        partitions = PilotPartitions(
            outer_fold=0,
            held_out_site="SITE_0",
            fitting_indices=np.asarray([2, 3, 4, 5]),
            validation_indices=np.asarray([6, 7]),
            test_indices=np.asarray([0, 1]),
        )
        tensors = build_pilot_tensors(cohort, partitions, 0.01, torch.device("cpu"))
        self.assertEqual(tensors.fitting_features.shape[0], 4)
        self.assertEqual(tensors.validation_features.shape[0], 2)
        self.assertEqual(tensors.fitting_adjacency.shape[0], 4)
        self.assertEqual(tensors.validation_adjacency.shape[0], 2)

    def test_checkpoint_resume_matches_uninterrupted_cpu_training(self) -> None:
        cohort = synthetic_cohort()
        partitions = PilotPartitions(
            outer_fold=0,
            held_out_site="SITE_0",
            fitting_indices=np.asarray([2, 3, 4, 5]),
            validation_indices=np.asarray([6, 7]),
            test_indices=np.asarray([0, 1]),
        )
        tensors = build_pilot_tensors(cohort, partitions, 0.0, torch.device("cpu"))
        architecture = NeuralArchitecture(input_dim=REGION_COUNT, hidden_dim=8, layers=1, bundles=2, channels=2, dropout=0.0)
        contract = {"run_id": "test", "operator": "identity", "density": 0.0}
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            uninterrupted = run_pilot_cell(
                run_dir=root / "uninterrupted", operator="identity", density=0.0, tensors=tensors,
                immutable_contract=contract, architecture=architecture, learning_rate=1e-3,
                weight_decay=0.0, batch_size=2, max_epochs=2, seed=11, resume=False, device=torch.device("cpu"),
            )
            first_epoch = run_pilot_cell(
                run_dir=root / "resumed", operator="identity", density=0.0, tensors=tensors,
                immutable_contract=contract, architecture=architecture, learning_rate=1e-3,
                weight_decay=0.0, batch_size=2, max_epochs=1, seed=11, resume=False, device=torch.device("cpu"),
            )
            resumed = run_pilot_cell(
                run_dir=root / "resumed", operator="identity", density=0.0, tensors=tensors,
                immutable_contract=contract, architecture=architecture, learning_rate=1e-3,
                weight_decay=0.0, batch_size=2, max_epochs=2, seed=11, resume=True, device=torch.device("cpu"),
            )
            self.assertEqual(len(first_epoch), 1)
            self.assertEqual(uninterrupted, resumed)
            with self.assertRaises(PilotTrainingError):
                run_pilot_cell(
                    run_dir=root / "resumed", operator="identity", density=0.0, tensors=tensors,
                    immutable_contract={"different": "contract"}, architecture=architecture, learning_rate=1e-3,
                    weight_decay=0.0, batch_size=2, max_epochs=2, seed=11, resume=True, device=torch.device("cpu"),
                )

    def test_global_resume_can_start_an_untouched_cell_only_when_explicitly_allowed(self) -> None:
        cohort = synthetic_cohort()
        partitions = PilotPartitions(
            outer_fold=0,
            held_out_site="SITE_0",
            fitting_indices=np.asarray([2, 3, 4, 5]),
            validation_indices=np.asarray([6, 7]),
            test_indices=np.asarray([0, 1]),
        )
        tensors = build_pilot_tensors(cohort, partitions, 0.0, torch.device("cpu"))
        architecture = NeuralArchitecture(input_dim=REGION_COUNT, hidden_dim=8, layers=1, bundles=2, channels=2, dropout=0.0)
        with TemporaryDirectory() as temporary:
            history = run_pilot_cell(
                run_dir=Path(temporary), operator="gcn", density=0.0, tensors=tensors,
                immutable_contract={"run_id": "test", "operator": "gcn", "density": 0.0},
                architecture=architecture, learning_rate=1e-3, weight_decay=0.0, batch_size=2,
                max_epochs=1, seed=19, resume=True, resume_missing_ok=True, device=torch.device("cpu"),
            )
        self.assertEqual([row["epoch"] for row in history], [0])


if __name__ == "__main__":
    unittest.main()
