"""Synthetic known-geometry data and matched fixed-graph operator models for E2."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn

from src.neural_operators import (
    fields_from_flat,
    flat_from_fields,
    graph_heat_kernel,
    identity_bundle_maps,
    normalized_gcn_adjacency,
    orthogonality_error,
    relative_transports,
    to_global,
    to_local,
)


Family = Literal[
    "S0_no_geometry",
    "S1_recoverable_geometry",
    "S2_incorrect_topology",
    "S3_shuffled_geometry",
    "S4_transport_noise",
    "S5_unlearnable_subject_frames",
    "S6_global_feature_analogue",
]
E2Operator = Literal[
    "identity", "gcn", "trivial_bundle", "fixed_random_transport",
    "learned_local", "learned_bunn", "oracle_true_map",
]


class E2Error(ValueError):
    """Raised when an E2 generator or model contract is invalid."""


@dataclass(frozen=True)
class E2Dataset:
    features: torch.Tensor
    labels: torch.Tensor
    adjacency: torch.Tensor
    true_adjacency: torch.Tensor
    true_maps: torch.Tensor
    train_indices: torch.Tensor
    validation_indices: torch.Tensor
    test_indices: torch.Tensor
    family: str
    seed: int
    transport_noise_degrees: float


def ring_adjacency(nodes: int, degree: int = 6) -> torch.Tensor:
    if nodes < 8 or degree <= 0 or degree >= nodes or degree % 2:
        raise E2Error("Ring graph requires an even degree between 2 and nodes-1")
    adjacency = torch.zeros(nodes, nodes, dtype=torch.float32)
    for offset in range(1, degree // 2 + 1):
        index = torch.arange(nodes)
        adjacency[index, (index + offset) % nodes] = 1.0
        adjacency[index, (index - offset) % nodes] = 1.0
    return adjacency


def permuted_topology(adjacency: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(adjacency.shape[0], generator=generator)
    changed = adjacency[permutation][:, permutation]
    if torch.equal(changed, adjacency):
        permutation = permutation.roll(1)
        changed = adjacency[permutation][:, permutation]
    return changed


def direct_o2_maps(angles: torch.Tensor, bundles: int) -> torch.Tensor:
    """Match the rotation/reflection convention used by LearnedOrthogonalBundleDiffusion."""
    if bundles <= 0 or bundles % 2:
        raise E2Error("E2 requires a positive even bundle count")
    expanded = angles.unsqueeze(-1).expand(*angles.shape, bundles)
    cosine, sine = torch.cos(expanded), torch.sin(expanded)
    maps = torch.zeros((*expanded.shape, 2, 2), dtype=angles.dtype, device=angles.device)
    maps[..., 0, 0] = cosine
    maps[..., 0, 1] = sine
    half = bundles // 2
    maps[..., :half, 1, 0] = -sine[..., :half]
    maps[..., :half, 1, 1] = cosine[..., :half]
    maps[..., half:, 1, 0] = sine[..., half:]
    maps[..., half:, 1, 1] = -cosine[..., half:]
    return maps


def fixed_node_angles(nodes: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    # A fixed random ordering prevents smooth local-coordinate shortcuts.
    return (2.0 * torch.pi) * torch.rand(nodes, generator=generator) - torch.pi


def _balanced_splits(samples: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if samples % 10 or samples < 20:
        raise E2Error("samples must be a positive multiple of 10")
    half = samples // 2
    labels = torch.cat((torch.zeros(half), torch.ones(half))).float()
    generator = torch.Generator().manual_seed(seed)
    class_zero = torch.randperm(half, generator=generator)
    class_one = half + torch.randperm(half, generator=generator)
    train_per_class, validation_per_class = 3 * samples // 10, samples // 10
    train = torch.cat((class_zero[:train_per_class], class_one[:train_per_class]))
    validation = torch.cat((
        class_zero[train_per_class:train_per_class + validation_per_class],
        class_one[train_per_class:train_per_class + validation_per_class],
    ))
    test = torch.cat((
        class_zero[train_per_class + validation_per_class:],
        class_one[train_per_class + validation_per_class:],
    ))
    return labels, train, validation, test


def _smooth_signals(samples: int, nodes: int, bundles: int, labels: torch.Tensor, seed: int) -> torch.Tensor:
    """Return identical node marginals by permuting class-0 versions of smooth class-1 signals."""
    generator = torch.Generator().manual_seed(seed)
    positions = torch.arange(nodes, dtype=torch.float32) * (2.0 * torch.pi / nodes)
    signals = torch.empty(samples, nodes, bundles)
    for row in range(samples):
        phases = torch.rand(bundles, generator=generator) * (2.0 * torch.pi)
        amplitudes = 0.75 + 0.5 * torch.rand(bundles, generator=generator)
        second = 0.25 * torch.sin(2.0 * positions[:, None] + 0.7 * phases[None, :])
        smooth = amplitudes[None, :] * torch.sin(positions[:, None] + phases[None, :]) + second
        if labels[row] == 0:
            smooth = smooth[torch.randperm(nodes, generator=generator)]
        signals[row] = smooth
    return signals


def _total_variation(signal: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    differences = signal.unsqueeze(2) - signal.unsqueeze(1)
    numerator = (differences.square() * adjacency[None, :, :, None]).sum(dim=(1, 2))
    return numerator / adjacency.sum().clamp_min(1.0)


def generate_e2_dataset(
    *, family: Family, seed: int, samples: int = 600, nodes: int = 116,
    bundles: int = 8, channels: int = 2, noise_standard_deviation: float = 0.15,
    marker_strength: float = 1.0, transport_noise_degrees: float = 0.0,
) -> E2Dataset:
    if channels != 2:
        raise E2Error("The frozen E2 generator uses exactly two channels")
    labels, train, validation, test = _balanced_splits(samples, seed + 1)
    true_adjacency = ring_adjacency(nodes)
    adjacency = true_adjacency.clone()
    if family == "S2_incorrect_topology":
        adjacency = permuted_topology(true_adjacency, seed + 2)

    fixed_angles = fixed_node_angles(nodes, seed=2026081400)
    fixed_maps = direct_o2_maps(fixed_angles, bundles)
    identity_maps = identity_bundle_maps(
        1, nodes, bundles, device=torch.device("cpu"), dtype=torch.float32
    )[0].clone()
    if family == "S0_no_geometry":
        true_maps = identity_maps.unsqueeze(0).expand(samples, -1, -1, -1, -1).clone()
    elif family == "S5_unlearnable_subject_frames":
        generator = torch.Generator().manual_seed(seed + 3)
        subject_angles = (2.0 * torch.pi) * torch.rand(samples, nodes, generator=generator) - torch.pi
        true_maps = direct_o2_maps(subject_angles, bundles)
    else:
        true_maps = fixed_maps.unsqueeze(0).expand(samples, -1, -1, -1, -1).clone()

    marker_maps = true_maps.clone()
    if family == "S3_shuffled_geometry":
        permutation = torch.randperm(nodes, generator=torch.Generator().manual_seed(seed + 4))
        marker_maps = marker_maps[:, permutation]
    elif family == "S4_transport_noise":
        generator = torch.Generator().manual_seed(seed + 5)
        radians = float(transport_noise_degrees) * torch.pi / 180.0
        noise = (2.0 * torch.rand(samples, nodes, generator=generator) - 1.0) * radians
        marker_maps = direct_o2_maps(fixed_angles.unsqueeze(0) + noise, bundles)
    elif family == "S5_unlearnable_subject_frames":
        marker_maps = identity_maps.unsqueeze(0).expand(samples, -1, -1, -1, -1).clone()

    signal = _smooth_signals(samples, nodes, bundles, labels, seed + 6)
    generator = torch.Generator().manual_seed(seed + 7)
    global_marker = torch.zeros(samples, nodes, bundles, 1, 2)
    global_marker[..., 0] = marker_strength
    global_signal = torch.zeros(samples, nodes, bundles, 1, 2)
    global_signal[..., 0, 0] = signal
    global_signal[..., 0, 1] = 0.35 * signal
    isotropic_noise = noise_standard_deviation * torch.randn(
        samples, nodes, bundles, 1, 2, generator=generator
    )
    global_signal = global_signal + isotropic_noise

    local_marker = to_local(global_marker, marker_maps)
    local_signal = to_local(global_signal, true_maps)
    fields = torch.cat((local_marker, local_signal), dim=3)

    if family == "S6_global_feature_analogue":
        total_variation = _total_variation(signal, true_adjacency)
        # Replicate a whole-graph statistic into half the bundles, analogous to
        # already-global connectivity-profile node features.
        summary = (total_variation - total_variation.mean(dim=0, keepdim=True)) / total_variation.std(
            dim=0, keepdim=True
        ).clamp_min(1e-6)
        fields[:, :, bundles // 2:, 1, 0] = summary[:, None, bundles // 2:]
        fields[:, :, bundles // 2:, 1, 1] = 0.0

    features = flat_from_fields(fields).float()
    if not torch.isfinite(features).all() or orthogonality_error(true_maps) > 1e-5:
        raise E2Error("Generator produced invalid features or non-orthogonal maps")
    return E2Dataset(
        features=features,
        labels=labels,
        adjacency=adjacency,
        true_adjacency=true_adjacency,
        true_maps=true_maps.float(),
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        family=family,
        seed=seed,
        transport_noise_degrees=float(transport_noise_degrees),
    )


class SyntheticGraphClassifier(nn.Module):
    """One-layer matched operator classifier with fixed graph kernels cached once."""

    def __init__(
        self, operator: E2Operator, adjacency: torch.Tensor, *, hidden_dim: int = 32,
        bundles: int = 8, channels: int = 2, diffusion_time: float = 1.0,
        random_map_seed: int = 2026081499,
    ) -> None:
        super().__init__()
        if hidden_dim != bundles * channels * 2:
            raise E2Error("hidden_dim must equal bundles * channels * 2")
        if adjacency.shape != (adjacency.shape[0], adjacency.shape[0]):
            raise E2Error("adjacency must be square")
        self.operator = operator
        self.hidden_dim = hidden_dim
        self.bundles = bundles
        self.channels = channels
        self.update = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.activation = nn.GELU()
        self.register_buffer("gcn_kernel", normalized_gcn_adjacency(adjacency.unsqueeze(0))[0])
        self.register_buffer("heat_kernel", graph_heat_kernel(adjacency.unsqueeze(0), diffusion_time)[0])
        if operator in {"learned_local", "learned_bunn"}:
            self.map_generator = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, bundles)
            )
        else:
            self.map_generator = None
        random_angles = fixed_node_angles(adjacency.shape[0], random_map_seed)
        self.register_buffer("random_maps", direct_o2_maps(random_angles, bundles))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _learned_maps(self, features: torch.Tensor) -> torch.Tensor:
        if self.map_generator is None:
            raise E2Error("This operator has no learned map generator")
        angles = self.map_generator(features)
        cosine, sine = torch.cos(angles), torch.sin(angles)
        maps = torch.zeros((*angles.shape, 2, 2), dtype=features.dtype, device=features.device)
        maps[..., 0, 0] = cosine
        maps[..., 0, 1] = sine
        half = self.bundles // 2
        maps[..., :half, 1, 0] = -sine[..., :half]
        maps[..., :half, 1, 1] = cosine[..., :half]
        maps[..., half:, 1, 0] = sine[..., half:]
        maps[..., half:, 1, 1] = -cosine[..., half:]
        return maps

    def forward(
        self, features: torch.Tensor, *, oracle_maps: torch.Tensor | None = None,
        return_maps: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, nodes, hidden = features.shape
        if hidden != self.hidden_dim or nodes != self.gcn_kernel.shape[0]:
            raise E2Error("Feature dimensions do not match the frozen model")
        identity_maps = identity_bundle_maps(
            batch, nodes, self.bundles, device=features.device, dtype=features.dtype
        )
        maps = identity_maps
        if self.operator == "identity":
            output = self.update(features)
        elif self.operator == "gcn":
            output = torch.einsum("nm,bmh->bnh", self.gcn_kernel, self.update(features))
        else:
            if self.operator in {"learned_local", "learned_bunn"}:
                maps = self._learned_maps(features)
            elif self.operator == "fixed_random_transport":
                maps = self.random_maps.unsqueeze(0).expand(batch, -1, -1, -1, -1)
            elif self.operator == "oracle_true_map":
                if oracle_maps is None or oracle_maps.shape[:3] != (batch, nodes, self.bundles):
                    raise E2Error("oracle_true_map requires aligned per-subject true maps")
                maps = oracle_maps
            elif self.operator == "trivial_bundle":
                maps = identity_maps
            else:
                raise E2Error(f"Unknown E2 operator: {self.operator}")
            fields = fields_from_flat(features, self.bundles, self.channels)
            common = to_global(fields, maps)
            updated = fields_from_flat(self.update(flat_from_fields(common)), self.bundles, self.channels)
            if self.operator not in {"learned_local"}:
                updated = torch.einsum("nm,bmqpd->bnqpd", self.heat_kernel, updated)
            output = flat_from_fields(to_local(updated, maps))
        hidden_output = self.activation(output)
        logits = self.classifier(hidden_output.mean(dim=1)).squeeze(-1)
        return (logits, maps) if return_maps else logits


def relative_transport_error(learned_maps: torch.Tensor, true_maps: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    """Gauge-invariant mean relative-transport Frobenius error on selected directed edges."""
    learned = relative_transports(learned_maps)
    truth = relative_transports(true_maps)
    squared = (learned - truth).square().sum(dim=(-1, -2, -3))
    mask = adjacency.to(dtype=squared.dtype, device=squared.device) > 0
    return (squared * mask).sum(dim=(-1, -2)) / mask.sum().clamp_min(1)
