"""One shared graph-classification backbone for all propagation controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from src.neural_operators import (
    GCNPropagation,
    IdentityPropagation,
    LearnedLocalOrthogonalUpdate,
    LearnedOrthogonalBundleDiffusion,
    OperatorError,
    TrivialBundleDiffusion,
    fields_from_flat,
    gauge_aware_metrics,
)


OperatorName = Literal["identity", "learned_local", "gcn", "trivial_bundle", "learned_bunn"]


@dataclass(frozen=True)
class NeuralArchitecture:
    input_dim: int = 116
    hidden_dim: int = 32
    layers: int = 2
    bundles: int = 8
    channels: int = 2
    dropout: float = 0.20
    diffusion_time: float = 1.0

    def validate(self) -> None:
        if self.hidden_dim != self.bundles * self.channels * 2:
            raise OperatorError("hidden_dim must equal bundles * channels * 2")
        if self.layers <= 0 or not 0 <= self.dropout < 1:
            raise OperatorError("layers must be positive and dropout must be in [0, 1)")


class SharedGraphClassifier(nn.Module):
    """Classifies one batch of participant graphs with exactly one operator choice."""

    def __init__(self, operator: OperatorName, architecture: NeuralArchitecture = NeuralArchitecture()) -> None:
        super().__init__()
        architecture.validate()
        if operator not in {"identity", "learned_local", "gcn", "trivial_bundle", "learned_bunn"}:
            raise OperatorError(f"Unknown propagation operator: {operator!r}")
        self.operator = operator
        self.architecture = architecture
        self.encoder = nn.Linear(architecture.input_dim, architecture.hidden_dim)
        self.propagation = nn.ModuleList(
            [self._make_layer() for _ in range(architecture.layers)]
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(architecture.dropout)
        self.classifier = nn.Linear(architecture.hidden_dim, 1)

    def _make_layer(self) -> nn.Module:
        common = (self.architecture.hidden_dim, self.architecture.bundles, self.architecture.channels)
        if self.operator == "identity":
            return IdentityPropagation(*common)
        if self.operator == "gcn":
            return GCNPropagation(*common)
        if self.operator == "trivial_bundle":
            return TrivialBundleDiffusion(*common, self.architecture.diffusion_time)
        if self.operator == "learned_local":
            return LearnedLocalOrthogonalUpdate(*common, self.architecture.diffusion_time)
        return LearnedOrthogonalBundleDiffusion(*common, self.architecture.diffusion_time)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        *,
        return_diagnostics: bool = False,
        include_encoder_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        hidden = self.encoder(features)
        diagnostics: list[dict[str, torch.Tensor]] = []
        if return_diagnostics and include_encoder_diagnostics:
            encoder_fields = fields_from_flat(hidden, self.architecture.bundles, self.architecture.channels)
            encoder_maps = torch.eye(2, device=hidden.device, dtype=hidden.dtype).view(1, 1, 1, 2, 2).expand(
                hidden.shape[0], hidden.shape[1], self.architecture.bundles, 2, 2
            )
            diagnostics.append(gauge_aware_metrics(encoder_fields, encoder_maps, adjacency))
        for layer in self.propagation:
            hidden, maps = layer(hidden, adjacency)
            hidden = self.dropout(self.activation(hidden))
            if return_diagnostics:
                fields = fields_from_flat(hidden, self.architecture.bundles, self.architecture.channels)
                diagnostics.append(gauge_aware_metrics(fields, maps, adjacency))
        logits = self.classifier(hidden.mean(dim=1)).squeeze(-1)
        return (logits, diagnostics) if return_diagnostics else logits
