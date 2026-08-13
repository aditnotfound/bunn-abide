from __future__ import annotations

import unittest

import torch

from src.extensions.e2_synthetic import (
    SyntheticGraphClassifier,
    direct_o2_maps,
    generate_e2_dataset,
    permuted_topology,
    relative_transport_error,
    ring_adjacency,
)
from src.neural_models import NeuralArchitecture, SharedGraphClassifier
from src.neural_operators import orthogonality_error, to_global


class E2SyntheticTests(unittest.TestCase):
    def test_o2_maps_are_orthogonal_and_relative_error_is_zero_for_global_gauge(self) -> None:
        angles = torch.linspace(-2.0, 2.0, 16).reshape(2, 8)
        maps = direct_o2_maps(angles, bundles=8)
        self.assertLess(float(orthogonality_error(maps)), 1e-6)
        adjacency = ring_adjacency(8, degree=2)
        self.assertTrue(torch.allclose(relative_transport_error(maps, maps, adjacency), torch.zeros(2)))

    def test_family_splits_are_balanced_disjoint_and_exhaustive(self) -> None:
        dataset = generate_e2_dataset(
            family="S1_recoverable_geometry", seed=11, samples=40, nodes=12,
        )
        parts = [set(x.tolist()) for x in (dataset.train_indices, dataset.validation_indices, dataset.test_indices)]
        self.assertFalse(parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
        self.assertEqual(set.union(*parts), set(range(40)))
        for indices in (dataset.train_indices, dataset.validation_indices, dataset.test_indices):
            self.assertEqual(float(dataset.labels[indices].mean()), 0.5)

    def test_recoverable_frames_round_trip_to_constant_marker(self) -> None:
        dataset = generate_e2_dataset(
            family="S1_recoverable_geometry", seed=12, samples=20, nodes=12,
            noise_standard_deviation=0.0,
        )
        fields = dataset.features.reshape(20, 12, 8, 2, 2)
        common = to_global(fields, dataset.true_maps)
        self.assertTrue(torch.allclose(common[..., 0, 0], torch.ones_like(common[..., 0, 0]), atol=1e-5))
        self.assertTrue(torch.allclose(common[..., 0, 1], torch.zeros_like(common[..., 0, 1]), atol=1e-5))

    def test_S0_uses_true_identity_maps_in_every_bundle(self) -> None:
        dataset = generate_e2_dataset(
            family="S0_no_geometry", seed=120, samples=20, nodes=12,
        )
        identity = torch.eye(2).view(1, 1, 1, 2, 2).expand_as(dataset.true_maps)
        self.assertTrue(torch.equal(dataset.true_maps, identity))

    def test_incorrect_topology_preserves_degree_and_changes_edges(self) -> None:
        adjacency = ring_adjacency(20, degree=6)
        wrong = permuted_topology(adjacency, seed=19)
        self.assertFalse(torch.equal(adjacency, wrong))
        self.assertTrue(torch.equal(adjacency.sum(1).sort().values, wrong.sum(1).sort().values))

    def test_subject_specific_frames_change_across_subjects(self) -> None:
        dataset = generate_e2_dataset(
            family="S5_unlearnable_subject_frames", seed=14, samples=20, nodes=12,
        )
        self.assertFalse(torch.equal(dataset.true_maps[0], dataset.true_maps[1]))

    def test_all_operator_outputs_are_finite_with_expected_shape(self) -> None:
        dataset = generate_e2_dataset(
            family="S1_recoverable_geometry", seed=15, samples=20, nodes=12,
        )
        for operator in (
            "identity", "gcn", "trivial_bundle", "fixed_random_transport",
            "learned_local", "learned_bunn", "oracle_true_map",
        ):
            model = SyntheticGraphClassifier(operator, dataset.adjacency)
            logits, maps = model(
                dataset.features[:4], oracle_maps=dataset.true_maps[:4], return_maps=True
            )
            self.assertEqual(logits.shape, (4,))
            self.assertEqual(maps.shape, (4, 12, 8, 2, 2))
            self.assertTrue(torch.isfinite(logits).all())
            self.assertLess(float(orthogonality_error(maps).detach()), 1e-5)

    def test_cached_synthetic_operators_match_original_implementation(self) -> None:
        dataset = generate_e2_dataset(
            family="S1_recoverable_geometry", seed=16, samples=20, nodes=12,
        )
        architecture = NeuralArchitecture(
            input_dim=32, hidden_dim=32, layers=1, bundles=8, channels=2,
            dropout=0.0, diffusion_time=1.0,
        )
        for operator in ("identity", "gcn", "trivial_bundle", "learned_local", "learned_bunn"):
            torch.manual_seed(77)
            original = SharedGraphClassifier(operator, architecture).eval()
            with torch.no_grad():
                original.encoder.weight.copy_(torch.eye(32))
                original.encoder.bias.zero_()
            cached = SyntheticGraphClassifier(operator, dataset.adjacency).eval()
            cached.update.load_state_dict(original.propagation[0].update.state_dict())
            cached.classifier.load_state_dict(original.classifier.state_dict())
            if operator in {"learned_local", "learned_bunn"}:
                cached.map_generator.load_state_dict(original.propagation[0].map_generator.state_dict())
            features = dataset.features[:3]
            adjacency = dataset.adjacency.unsqueeze(0).expand(3, -1, -1)
            with torch.no_grad():
                expected = original(features, adjacency)
                actual = cached(features)
            self.assertTrue(torch.allclose(actual, expected, atol=2e-5, rtol=1e-5), operator)


if __name__ == "__main__":
    unittest.main()
