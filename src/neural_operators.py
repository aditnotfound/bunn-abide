"""Dense, testable propagation operators for the ABIDE-I operator audit.

BuNN follows the flat-bundle heat-diffusion construction: node fields are
mapped to a common frame, updated, diffused by a graph heat kernel, returned to
their local frames, and activated by the shared classifier.  All operators use
the same dense participant-by-participant batching convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class OperatorError(ValueError):
    """Raised when an operator receives an invalid dense graph batch."""


def validate_operator_inputs(features: torch.Tensor, adjacency: torch.Tensor) -> None:
    if features.ndim != 3 or adjacency.ndim != 3:
        raise OperatorError("Features and adjacency must both be rank-3 tensors")
    if features.shape[:2] != adjacency.shape[:2] or adjacency.shape[-2] != adjacency.shape[-1]:
        raise OperatorError("Features and adjacency batch/node dimensions must align")
    if not torch.isfinite(features).all() or not torch.isfinite(adjacency).all():
        raise OperatorError("Features and adjacency must be finite")
    if (adjacency < 0).any() or not torch.allclose(adjacency, adjacency.transpose(-1, -2), atol=1e-6, rtol=0.0):
        raise OperatorError("Adjacency must be non-negative and symmetric")
    if not torch.allclose(torch.diagonal(adjacency, dim1=-2, dim2=-1), torch.zeros_like(adjacency[..., 0]), atol=1e-7, rtol=0.0):
        raise OperatorError("Propagation adjacency must not contain self-loops")


def random_walk_laplacian(adjacency: torch.Tensor) -> torch.Tensor:
    """Return L = I - D^-1 A, with isolated nodes defined as zero-Laplacian."""
    if adjacency.ndim != 3 or adjacency.shape[-1] != adjacency.shape[-2]:
        raise OperatorError("Expected a square [batch, nodes, nodes] adjacency tensor")
    degree = adjacency.sum(dim=-1)
    transition = adjacency / degree.clamp_min(1.0).unsqueeze(-1)
    active = (degree > 0).to(adjacency.dtype)
    identity = torch.eye(adjacency.shape[-1], dtype=adjacency.dtype, device=adjacency.device)
    return active.unsqueeze(-1) * identity.unsqueeze(0) - transition


def graph_heat_kernel(adjacency: torch.Tensor, diffusion_time: float) -> torch.Tensor:
    """Dense exact heat kernel exp(-tL), appropriate for 116-node participant graphs."""
    if diffusion_time < 0:
        raise OperatorError("Diffusion time must be non-negative")
    laplacian = random_walk_laplacian(adjacency)
    return torch.linalg.matrix_exp(-float(diffusion_time) * laplacian)


def normalized_gcn_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """Kipf-Welling symmetric normalization with exactly one self-loop per node."""
    nodes = adjacency.shape[-1]
    identity = torch.eye(nodes, dtype=adjacency.dtype, device=adjacency.device).unsqueeze(0)
    augmented = adjacency + identity
    degree = augmented.sum(dim=-1).clamp_min(1.0)
    inverse_sqrt = degree.rsqrt()
    return inverse_sqrt.unsqueeze(-1) * augmented * inverse_sqrt.unsqueeze(-2)


def identity_bundle_maps(
    batch_size: int, nodes: int, bundles: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    eye = torch.eye(2, device=device, dtype=dtype)
    return eye.view(1, 1, 1, 2, 2).expand(batch_size, nodes, bundles, 2, 2)


def fields_from_flat(features: torch.Tensor, bundles: int, channels: int) -> torch.Tensor:
    batch, nodes, hidden = features.shape
    if hidden != bundles * channels * 2:
        raise OperatorError("Hidden dimension must equal bundles * channels * 2")
    return features.reshape(batch, nodes, bundles, channels, 2)


def flat_from_fields(fields: torch.Tensor) -> torch.Tensor:
    return fields.reshape(fields.shape[0], fields.shape[1], -1)


def to_global(fields: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
    """Apply O_v to each local two-dimensional field component."""
    return torch.einsum("snqij,snqpi->snqpj", maps, fields)


def to_local(fields: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
    """Apply O_v^T to return common-frame fields to node-local coordinates."""
    return torch.einsum("snqij,snqpj->snqpi", maps, fields)


def apply_heat(kernel: torch.Tensor, global_fields: torch.Tensor) -> torch.Tensor:
    return torch.einsum("snm,smqpd->snqpd", kernel, global_fields)


def orthogonality_error(maps: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(2, dtype=maps.dtype, device=maps.device)
    products = torch.matmul(maps.transpose(-1, -2), maps)
    return (products - identity).abs().amax()


def relative_transports(maps: torch.Tensor) -> torch.Tensor:
    """T[v->u] = O_u^T O_v in local coordinates, indexed [u, v]."""
    return torch.einsum("buqig,bvqjg->buvqij", maps, maps)


def invariant_edge_transport_distance(
    fields: torch.Tensor, maps: torch.Tensor, adjacency: torch.Tensor
) -> torch.Tensor:
    """Mean ||x_u - O_u^T O_v x_v|| over directed selected edges per graph."""
    transports = relative_transports(maps)
    transported = torch.einsum("buvqij,bvqpj->buvqpi", transports, fields)
    differences = fields.unsqueeze(2) - transported
    squared_norm = differences.square().sum(dim=(-1, -2, -3))
    mask = adjacency > 0
    counts = mask.sum(dim=(-1, -2))
    totals = (squared_norm * mask).sum(dim=(-1, -2))
    return torch.where(counts > 0, totals / counts.clamp_min(1), torch.zeros_like(totals))


def gauge_transform(fields: torch.Tensor, maps: torch.Tensor, gauge: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Change every local frame while preserving global fields and invariants."""
    transformed_fields = torch.einsum("snqij,snqpj->snqpi", gauge, fields)
    transformed_maps = torch.einsum("snqij,snqjg->snqig", gauge, maps)
    return transformed_fields, transformed_maps


def gauge_aware_metrics(fields: torch.Tensor, maps: torch.Tensor, adjacency: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-subject representation diagnostics in a common learned frame."""
    global_flat = flat_from_fields(to_global(fields, maps))
    centered = global_flat - global_flat.mean(dim=1, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    probabilities = singular_values / singular_values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1))
    normalized_effective_rank = effective_rank / min(centered.shape[1], centered.shape[2])
    dispersion = centered.square().mean(dim=(1, 2)) / global_flat.square().mean(dim=(1, 2)).clamp_min(1e-12)
    normalized = global_flat / global_flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cosine = torch.matmul(normalized, normalized.transpose(-1, -2))
    nodes = cosine.shape[-1]
    mean_cosine = (cosine.sum(dim=(-1, -2)) - nodes) / (nodes * (nodes - 1))
    return {
        "normalized_effective_rank": normalized_effective_rank,
        "normalized_dispersion": dispersion,
        "mean_pairwise_cosine": mean_cosine,
        "invariant_edge_transport_distance": invariant_edge_transport_distance(fields, maps, adjacency),
    }


class IdentityPropagation(nn.Module):
    """A pointwise shared update with no inter-node information exchange."""

    def __init__(self, hidden_dim: int, bundles: int, channels: int) -> None:
        super().__init__()
        self.update = nn.Linear(hidden_dim, hidden_dim)
        self.bundles = bundles
        self.channels = channels

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        validate_operator_inputs(features, adjacency)
        maps = identity_bundle_maps(
            features.shape[0], features.shape[1], self.bundles, device=features.device, dtype=features.dtype
        )
        return self.update(features), maps


class GCNPropagation(nn.Module):
    """The ordinary normalized aggregation comparator with the same pointwise width."""

    def __init__(self, hidden_dim: int, bundles: int, channels: int) -> None:
        super().__init__()
        self.update = nn.Linear(hidden_dim, hidden_dim)
        self.bundles = bundles
        self.channels = channels

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        validate_operator_inputs(features, adjacency)
        maps = identity_bundle_maps(
            features.shape[0], features.shape[1], self.bundles, device=features.device, dtype=features.dtype
        )
        return torch.bmm(normalized_gcn_adjacency(adjacency), self.update(features)), maps


class _BundleDiffusionBase(nn.Module):
    def __init__(self, hidden_dim: int, bundles: int, channels: int, diffusion_time: float) -> None:
        super().__init__()
        if hidden_dim != bundles * channels * 2 or bundles <= 0 or channels <= 0:
            raise OperatorError("Bundle hidden dimension must be bundles * channels * 2")
        if diffusion_time < 0:
            raise OperatorError("Diffusion time must be non-negative")
        self.update = nn.Linear(hidden_dim, hidden_dim)
        self.bundles = bundles
        self.channels = channels
        self.diffusion_time = float(diffusion_time)

    def maps_for(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        validate_operator_inputs(features, adjacency)
        maps = self.maps_for(features)
        if maps.shape != (*features.shape[:2], self.bundles, 2, 2):
            raise OperatorError("Bundle-map generator returned an invalid shape")
        fields = fields_from_flat(features, self.bundles, self.channels)
        common_fields = to_global(fields, maps)
        updated = self.update(flat_from_fields(common_fields))
        updated_fields = fields_from_flat(updated, self.bundles, self.channels)
        diffused = apply_heat(graph_heat_kernel(adjacency, self.diffusion_time), updated_fields)
        return flat_from_fields(to_local(diffused, maps)), maps


class TrivialBundleDiffusion(_BundleDiffusionBase):
    """Bundle heat diffusion with fixed identity transports; controls for diffusion alone."""

    def maps_for(self, features: torch.Tensor) -> torch.Tensor:
        return identity_bundle_maps(
            features.shape[0], features.shape[1], self.bundles, device=features.device, dtype=features.dtype
        )


class LearnedOrthogonalBundleDiffusion(_BundleDiffusionBase):
    """BuNN-style flat bundle diffusion with feature-conditioned O(2) maps.

    The paper's direct O(2) parameterization is used: half the bundle channels
    are rotations and half are reflections, so every map is exactly orthogonal.
    """

    def __init__(self, hidden_dim: int, bundles: int, channels: int, diffusion_time: float) -> None:
        if bundles % 2:
            raise OperatorError("Direct O(2) parameterization requires an even number of bundles")
        super().__init__(hidden_dim, bundles, channels, diffusion_time)
        self.map_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bundles),
        )

    def maps_for(self, features: torch.Tensor) -> torch.Tensor:
        angles = self.map_generator(features)
        cosine, sine = torch.cos(angles), torch.sin(angles)
        maps = torch.zeros(
            (*angles.shape, 2, 2), dtype=features.dtype, device=features.device
        )
        maps[..., 0, 0] = cosine
        maps[..., 0, 1] = sine
        half = self.bundles // 2
        maps[..., :half, 1, 0] = -sine[..., :half]
        maps[..., :half, 1, 1] = cosine[..., :half]
        maps[..., half:, 1, 0] = sine[..., half:]
        maps[..., half:, 1, 1] = -cosine[..., half:]
        return maps


@dataclass(frozen=True)
class OperatorOutput:
    features: torch.Tensor
    maps: torch.Tensor
