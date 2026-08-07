from __future__ import annotations

import unittest

import torch

from src.neural_models import NeuralArchitecture, SharedGraphClassifier
from src.neural_operators import (
    GCNPropagation,
    IdentityPropagation,
    LearnedOrthogonalBundleDiffusion,
    TrivialBundleDiffusion,
    apply_heat,
    fields_from_flat,
    gauge_aware_metrics,
    gauge_transform,
    graph_heat_kernel,
    identity_bundle_maps,
    invariant_edge_transport_distance,
    orthogonality_error,
    relative_transports,
    to_global,
    to_local,
)


NODES = 5
BUNDLES = 2
CHANNELS = 2
HIDDEN = BUNDLES * CHANNELS * 2


def adjacency(batch: int = 2) -> torch.Tensor:
    output = torch.zeros((batch, NODES, NODES), dtype=torch.float32)
    for index in range(NODES - 1):
        output[:, index, index + 1] = 1.0
        output[:, index + 1, index] = 1.0
    return output


class NeuralOperatorTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.features = torch.randn((2, NODES, HIDDEN), dtype=torch.float32)
        self.adjacency = adjacency()

    def test_zero_density_gcn_equals_pointwise_identity_update(self) -> None:
        identity = IdentityPropagation(HIDDEN, BUNDLES, CHANNELS)
        gcn = GCNPropagation(HIDDEN, BUNDLES, CHANNELS)
        gcn.update.load_state_dict(identity.update.state_dict())
        empty = torch.zeros_like(self.adjacency)
        identity_output, _ = identity(self.features, empty)
        gcn_output, _ = gcn(self.features, empty)
        self.assertTrue(torch.allclose(identity_output, gcn_output, atol=1e-6, rtol=0.0))

    def test_zero_density_trivial_bundle_has_no_inter_node_diffusion(self) -> None:
        identity = IdentityPropagation(HIDDEN, BUNDLES, CHANNELS)
        trivial = TrivialBundleDiffusion(HIDDEN, BUNDLES, CHANNELS, diffusion_time=1.0)
        trivial.update.load_state_dict(identity.update.state_dict())
        empty = torch.zeros_like(self.adjacency)
        identity_output, _ = identity(self.features, empty)
        trivial_output, _ = trivial(self.features, empty)
        self.assertTrue(torch.allclose(identity_output, trivial_output, atol=1e-6, rtol=0.0))
        self.assertTrue(torch.allclose(graph_heat_kernel(empty, 1.0), torch.eye(NODES).expand(2, -1, -1)))

    def test_common_frame_round_trip_and_trivial_heat_match(self) -> None:
        fields = fields_from_flat(self.features, BUNDLES, CHANNELS)
        maps = identity_bundle_maps(2, NODES, BUNDLES, device=fields.device, dtype=fields.dtype)
        self.assertTrue(torch.allclose(to_local(to_global(fields, maps), maps), fields, atol=1e-6, rtol=0.0))
        kernel = graph_heat_kernel(self.adjacency, 0.4)
        expected = torch.einsum("bnm,bmcd->bncd", kernel, self.features.reshape(2, NODES, -1, 2)).reshape_as(self.features)
        actual = apply_heat(kernel, fields).reshape_as(self.features)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=0.0))

    def test_learned_maps_are_orthogonal_and_backpropagate(self) -> None:
        layer = LearnedOrthogonalBundleDiffusion(HIDDEN, BUNDLES, CHANNELS, diffusion_time=0.5)
        output, maps = layer(self.features.clone().requires_grad_(True), self.adjacency)
        self.assertLess(float(orthogonality_error(maps).detach()), 1e-5)
        loss = output.square().mean()
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in layer.parameters()))
        self.assertTrue(torch.isfinite(output).all())

    def test_invariant_transport_distance_and_common_frame_metrics_are_gauge_invariant(self) -> None:
        layer = LearnedOrthogonalBundleDiffusion(HIDDEN, BUNDLES, CHANNELS, diffusion_time=0.5)
        fields = fields_from_flat(self.features, BUNDLES, CHANNELS)
        maps = layer.maps_for(self.features)
        gauge = layer.maps_for(torch.flip(self.features, dims=(1,)))
        transformed_fields, transformed_maps = gauge_transform(fields, maps, gauge)
        before = gauge_aware_metrics(fields, maps, self.adjacency)
        after = gauge_aware_metrics(transformed_fields, transformed_maps, self.adjacency)
        for key in before:
            self.assertTrue(torch.allclose(before[key], after[key], atol=1e-5, rtol=1e-5), key)

    def test_identity_transport_distance_has_expected_value(self) -> None:
        fields = torch.zeros((1, NODES, BUNDLES, CHANNELS, 2), dtype=torch.float32)
        fields[:, 0, :, :, 0] = 1.0
        graph = torch.zeros((1, NODES, NODES), dtype=torch.float32)
        graph[:, 0, 1] = graph[:, 1, 0] = 1.0
        maps = identity_bundle_maps(1, NODES, BUNDLES, device=fields.device, dtype=fields.dtype)
        # Each directed edge differs in BUNDLES * CHANNELS unit vector components.
        self.assertAlmostEqual(float(invariant_edge_transport_distance(fields, maps, graph)), float(BUNDLES * CHANNELS))

    def test_relative_transport_has_the_documented_v_to_u_orientation(self) -> None:
        maps = identity_bundle_maps(1, NODES, BUNDLES, device=self.features.device, dtype=self.features.dtype).clone()
        # O_v maps local e_0 to common-frame e_1. With O_u = I, transporting
        # from v=1 to u=0 must therefore yield local e_1 at node u.
        maps[0, 1, 0] = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
        transport = relative_transports(maps)[0, 0, 1, 0]
        self.assertTrue(torch.allclose(transport @ torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])))

    def test_shared_classifier_is_batch_isolated_finite_and_seed_deterministic(self) -> None:
        architecture = NeuralArchitecture(input_dim=HIDDEN, hidden_dim=HIDDEN, layers=2, bundles=BUNDLES, channels=CHANNELS, dropout=0.0)
        torch.manual_seed(42)
        first = SharedGraphClassifier("learned_bunn", architecture).eval()
        torch.manual_seed(42)
        second = SharedGraphClassifier("learned_bunn", architecture).eval()
        logits, diagnostics = first(self.features, self.adjacency, return_diagnostics=True)
        self.assertTrue(torch.allclose(logits, second(self.features, self.adjacency)))
        self.assertEqual(len(diagnostics), 2)
        self.assertTrue(torch.isfinite(logits).all())
        single_logits = first(self.features[:1], self.adjacency[:1])
        self.assertTrue(torch.allclose(logits[:1], single_logits, atol=1e-6, rtol=0.0))
        logits.sum().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in first.parameters()))


if __name__ == "__main__":
    unittest.main()
