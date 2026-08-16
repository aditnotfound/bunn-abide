from __future__ import annotations

import unittest

import torch

from src.extensions.e1_interventions import (
    degree_preserving_rewire,
    deterministic_derangement,
    forward_intervened,
    random_o2_maps,
)
from src.neural_models import SharedGraphClassifier
from src.neural_operators import orthogonality_error


def cycle_adjacency(batch: int, nodes: int) -> torch.Tensor:
    adjacency = torch.zeros(batch, nodes, nodes)
    for node in range(nodes):
        adjacency[:, node, (node + 1) % nodes] = 1
        adjacency[:, (node + 1) % nodes, node] = 1
    return adjacency


class E1InterventionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(9)
        self.model = SharedGraphClassifier("learned_bunn")
        self.model.eval()
        self.features = torch.randn(3, 116, 116)
        self.adjacency = cycle_adjacency(3, 116)

    def test_unaltered_reproduces_standard_forward(self) -> None:
        expected = self.model(self.features, self.adjacency)
        actual = forward_intervened(self.model, self.features, self.adjacency, "unaltered").logits
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)

    def test_encoded_node_permutation_is_equivariant(self) -> None:
        reference = forward_intervened(self.model, self.features, self.adjacency, "unaltered")
        permuted = forward_intervened(
            self.model, self.features, self.adjacency,
            "encoded_node_permutation_equivariance", randomization_seed=71,
        )
        torch.testing.assert_close(permuted.logits, reference.logits, atol=2e-5, rtol=0)
        for left, right in zip(reference.diagnostics, permuted.diagnostics, strict=True):
            for metric in left:
                torch.testing.assert_close(right[metric], left[metric], atol=2e-5, rtol=0)

    def test_derangement_is_reproducible_and_has_no_fixed_points(self) -> None:
        first = deterministic_derangement(116, 4)
        second = deterministic_derangement(116, 4)
        self.assertTrue((first == second).all())
        self.assertTrue((first != torch.arange(116).numpy()).all())

    def test_random_maps_are_orthogonal(self) -> None:
        maps = random_o2_maps(2, 116, 8, seed=3, device=torch.device("cpu"), dtype=torch.float32)
        self.assertLess(float(orthogonality_error(maps)), 1e-5)

    def test_rewire_preserves_every_degree_and_is_deterministic(self) -> None:
        first = degree_preserving_rewire(self.adjacency, seed=17)
        second = degree_preserving_rewire(self.adjacency, seed=17)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first.sum(-1), self.adjacency.sum(-1)))
        self.assertTrue(torch.equal(first, first.transpose(-1, -2)))
        self.assertTrue(torch.equal(torch.diagonal(first, dim1=-2, dim2=-1), torch.zeros(3, 116)))
        self.assertFalse(torch.equal(first, self.adjacency))


if __name__ == "__main__":
    unittest.main()
