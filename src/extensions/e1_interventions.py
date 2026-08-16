"""Score-blind interventions for accepted learned-BuNN checkpoints.

The functions in this module do not refit a model. They alter maps, topology,
or encoded-node order during inference while retaining every accepted weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from src.neural_models import SharedGraphClassifier
from src.neural_operators import (
    apply_heat,
    fields_from_flat,
    flat_from_fields,
    gauge_aware_metrics,
    graph_heat_kernel,
    identity_bundle_maps,
    orthogonality_error,
    to_global,
    to_local,
    validate_operator_inputs,
)


InterventionName = Literal[
    "unaltered",
    "identity_maps",
    "node_map_shuffle",
    "random_orthogonal_maps",
    "degree_preserving_topology",
    "encoded_node_permutation_equivariance",
]


@dataclass(frozen=True)
class InterventionOutput:
    logits: torch.Tensor
    diagnostics: list[dict[str, torch.Tensor]]
    adjacency: torch.Tensor
    maximum_orthogonality_error: float


def deterministic_derangement(nodes: int, seed: int) -> np.ndarray:
    """Return a reproducible permutation with no fixed node positions."""
    if nodes < 2:
        raise ValueError("A derangement requires at least two nodes")
    rng = np.random.default_rng(seed)
    base = np.arange(nodes)
    for _ in range(10_000):
        candidate = rng.permutation(nodes)
        if np.all(candidate != base):
            return candidate
    raise RuntimeError("Could not generate a derangement")


def random_o2_maps(
    batch: int,
    nodes: int,
    bundles: int,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate fixed O(2) maps with the frozen rotation/reflection split."""
    if bundles % 2:
        raise ValueError("The O(2) split requires an even bundle count")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    angles = 2.0 * torch.pi * torch.rand((nodes, bundles), generator=generator, dtype=torch.float64)
    cosine, sine = angles.cos(), angles.sin()
    maps = torch.zeros((nodes, bundles, 2, 2), dtype=torch.float64)
    maps[..., 0, 0] = cosine
    maps[..., 0, 1] = sine
    half = bundles // 2
    maps[:, :half, 1, 0] = -sine[:, :half]
    maps[:, :half, 1, 1] = cosine[:, :half]
    maps[:, half:, 1, 0] = sine[:, half:]
    maps[:, half:, 1, 1] = -cosine[:, half:]
    return maps.to(device=device, dtype=dtype).unsqueeze(0).expand(batch, -1, -1, -1, -1)


def degree_preserving_rewire(
    adjacency: torch.Tensor,
    *,
    seed: int,
    attempts_per_edge: int = 10,
) -> torch.Tensor:
    """Apply deterministic simple-graph double-edge swaps to every graph."""
    validate_operator_inputs(torch.zeros((*adjacency.shape[:2], 1), device=adjacency.device), adjacency)
    rewired = adjacency.detach().cpu().numpy().copy()
    for graph_index, graph in enumerate(rewired):
        binary = graph > 0
        edges = [tuple(edge) for edge in np.argwhere(np.triu(binary, 1))]
        edge_set = set(edges)
        rng = np.random.default_rng(seed + 1_000_003 * graph_index)
        for _ in range(max(1, attempts_per_edge * len(edges))):
            if len(edges) < 2:
                break
            first, second = rng.choice(len(edges), size=2, replace=False)
            a, b = edges[int(first)]
            c, d = edges[int(second)]
            if len({a, b, c, d}) < 4:
                continue
            if rng.integers(2):
                proposed = (tuple(sorted((a, c))), tuple(sorted((b, d))))
            else:
                proposed = (tuple(sorted((a, d))), tuple(sorted((b, c))))
            if proposed[0] == proposed[1] or proposed[0] in edge_set or proposed[1] in edge_set:
                continue
            edge_set.remove((a, b))
            edge_set.remove((c, d))
            edge_set.update(proposed)
            edges[int(first)], edges[int(second)] = proposed
        new_graph = np.zeros_like(graph)
        for u, v in edge_set:
            new_graph[u, v] = new_graph[v, u] = 1.0
        rewired[graph_index] = new_graph
    result = torch.as_tensor(rewired, device=adjacency.device, dtype=adjacency.dtype)
    if not torch.equal((result > 0).sum(-1), (adjacency > 0).sum(-1)):
        raise RuntimeError("Degree-preserving intervention changed a degree sequence")
    return result


def _bundle_layer_with_maps(
    layer, hidden: torch.Tensor, adjacency: torch.Tensor, maps: torch.Tensor,
    heat_kernel: torch.Tensor | None = None,
) -> torch.Tensor:
    fields = fields_from_flat(hidden, layer.bundles, layer.channels)
    common = to_global(fields, maps)
    updated = layer.update(flat_from_fields(common))
    kernel = heat_kernel if heat_kernel is not None else graph_heat_kernel(adjacency, layer.diffusion_time)
    diffused = apply_heat(
        kernel,
        fields_from_flat(updated, layer.bundles, layer.channels),
    )
    return flat_from_fields(to_local(diffused, maps))


@torch.no_grad()
def forward_intervened(
    model: SharedGraphClassifier,
    features: torch.Tensor,
    adjacency: torch.Tensor,
    intervention: InterventionName,
    *,
    randomization_seed: int = 0,
    adjacency_override: torch.Tensor | None = None,
    heat_kernel: torch.Tensor | None = None,
) -> InterventionOutput:
    """Run one accepted model under one pre-specified inference intervention."""
    if model.operator != "learned_bunn":
        raise ValueError("E1 interventions require a learned_bunn checkpoint")
    model.eval()
    effective_adjacency = adjacency_override if adjacency_override is not None else adjacency
    if intervention == "degree_preserving_topology":
        if adjacency_override is None:
            effective_adjacency = degree_preserving_rewire(adjacency, seed=randomization_seed)

    hidden = model.encoder(features)
    undo_permutation: torch.Tensor | None = None
    if intervention == "encoded_node_permutation_equivariance":
        permutation = torch.as_tensor(
            deterministic_derangement(hidden.shape[1], randomization_seed), device=hidden.device
        )
        undo_permutation = torch.argsort(permutation)
        hidden = hidden[:, permutation]
        effective_adjacency = effective_adjacency[:, permutation][:, :, permutation]

    diagnostics: list[dict[str, torch.Tensor]] = []
    maximum_error = 0.0
    final_maps: torch.Tensor | None = None
    for layer_index, layer in enumerate(model.propagation):
        learned_maps = layer.maps_for(hidden)
        maps = learned_maps
        if intervention == "identity_maps":
            maps = identity_bundle_maps(
                hidden.shape[0], hidden.shape[1], layer.bundles,
                device=hidden.device, dtype=hidden.dtype,
            )
        elif intervention == "node_map_shuffle":
            permutation = torch.as_tensor(
                deterministic_derangement(hidden.shape[1], randomization_seed + 10_007 * layer_index),
                device=hidden.device,
            )
            maps = learned_maps[:, permutation]
        elif intervention == "random_orthogonal_maps":
            maps = random_o2_maps(
                hidden.shape[0], hidden.shape[1], layer.bundles,
                seed=randomization_seed + 10_007 * layer_index,
                device=hidden.device, dtype=hidden.dtype,
            )
        hidden = _bundle_layer_with_maps(layer, hidden, effective_adjacency, maps, heat_kernel)
        hidden = model.dropout(model.activation(hidden))
        final_maps = maps
        maximum_error = max(maximum_error, float(orthogonality_error(maps).cpu()))

    if final_maps is None:
        raise RuntimeError("Model contains no propagation layers")
    fields = fields_from_flat(hidden, model.architecture.bundles, model.architecture.channels)
    diagnostics.append(gauge_aware_metrics(fields, final_maps, effective_adjacency))
    if undo_permutation is not None:
        hidden = hidden[:, undo_permutation]
    logits = model.classifier(hidden.mean(dim=1)).squeeze(-1)
    return InterventionOutput(logits, diagnostics, effective_adjacency, maximum_error)
