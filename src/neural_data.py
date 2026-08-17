"""Validated dense-graph inputs for the frozen ABIDE-I neural study.

The neural phase uses the same 754-subject connectome artifact and held-out
site assignments as the completed baseline phase.  This module intentionally
contains no split generation, model fitting, or label-dependent transform.
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


REGION_COUNT = 116
UNDIRECTED_EDGE_COUNT = REGION_COUNT * (REGION_COUNT - 1) // 2
NEURAL_DENSITIES = (0.0, 0.01, 0.05, 0.10, 0.20)


class NeuralDataError(ValueError):
    """Raised when a frozen neural input is malformed or misaligned."""


@dataclass(frozen=True)
class NeuralCohort:
    """The model-independent participant order shared by every neural run."""

    subject_ids: np.ndarray
    site_ids: np.ndarray
    labels: np.ndarray
    region_labels: np.ndarray
    connectomes: np.ndarray

    @property
    def participants(self) -> int:
        return int(self.connectomes.shape[0])


@dataclass(frozen=True)
class DenseGraphBatch:
    """A dense batch; each participant owns one graph and no edges cross subjects."""

    features: torch.Tensor
    adjacency: torch.Tensor
    density: float


def _read_baseline_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"connectome_row", "subject_id", "site_id", "label_asd"}
    if not rows or rows[0].keys() is None:
        raise NeuralDataError("Baseline table is empty")
    missing = required - set(rows[0])
    if missing:
        raise NeuralDataError(f"Baseline table lacks columns: {sorted(missing)}")
    return rows


def load_neural_cohort(npz_path: Path, baseline_table_path: Path) -> NeuralCohort:
    """Load and cross-check the frozen connectome artifact against the baseline table."""
    if not npz_path.is_file() or not baseline_table_path.is_file():
        raise NeuralDataError("Neural inputs require an NPZ connectome artifact and baseline table")
    with np.load(npz_path, allow_pickle=False) as arrays:
        required = {"subject_id", "site_id", "label_asd", "region_labels", "connectomes_fisher_z"}
        missing = required - set(arrays.files)
        if missing:
            raise NeuralDataError(f"Connectome artifact lacks arrays: {sorted(missing)}")
        subject_ids = np.asarray(arrays["subject_id"]).astype(str)
        site_ids = np.asarray(arrays["site_id"]).astype(str)
        labels = np.asarray(arrays["label_asd"], dtype=np.int64)
        region_labels = np.asarray(arrays["region_labels"]).astype(str)
        connectomes = np.asarray(arrays["connectomes_fisher_z"], dtype=np.float32)

    count = len(subject_ids)
    if (
        subject_ids.ndim != 1
        or site_ids.shape != (count,)
        or labels.shape != (count,)
        or connectomes.shape != (count, REGION_COUNT, REGION_COUNT)
        or region_labels.shape != (REGION_COUNT,)
    ):
        raise NeuralDataError("Connectome array dimensions do not match the frozen AAL-116 contract")
    if len(set(subject_ids.tolist())) != count or any(not value for value in subject_ids):
        raise NeuralDataError("Subject identifiers must be unique and non-empty")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise NeuralDataError("Neural cohort labels must contain exactly ASD=1 and control=0")
    if not np.isfinite(connectomes).all():
        raise NeuralDataError("Connectome tensor contains non-finite values")
    if not np.allclose(connectomes, np.swapaxes(connectomes, 1, 2), atol=1e-6):
        raise NeuralDataError("Connectome tensor is not symmetric")
    if not np.allclose(np.diagonal(connectomes, axis1=1, axis2=2), 0.0, atol=1e-7):
        raise NeuralDataError("Connectome diagonals must be zero")

    baseline_rows = _read_baseline_rows(baseline_table_path)
    if len(baseline_rows) != count:
        raise NeuralDataError("Baseline table row count differs from the connectome artifact")
    for expected_row, row in enumerate(baseline_rows):
        try:
            row_index = int(row["connectome_row"])
            label = int(row["label_asd"])
        except ValueError as error:
            raise NeuralDataError("Baseline table has non-integer alignment fields") from error
        if row_index != expected_row:
            raise NeuralDataError("Baseline connectome_row must be a complete zero-based sequence")
        if (
            row["subject_id"] != subject_ids[expected_row]
            or row["site_id"] != site_ids[expected_row]
            or label != int(labels[expected_row])
        ):
            raise NeuralDataError(f"Baseline/connectome alignment mismatch at row {expected_row}")
    return NeuralCohort(subject_ids, site_ids, labels, region_labels, connectomes)


def _validate_connectome_tensor(connectomes: torch.Tensor) -> None:
    if connectomes.ndim != 3 or connectomes.shape[-2:] != (REGION_COUNT, REGION_COUNT):
        raise NeuralDataError("Expected a [participants, 116, 116] connectome tensor")
    if not torch.isfinite(connectomes).all():
        raise NeuralDataError("Connectome tensor contains non-finite values")
    if not torch.allclose(connectomes, connectomes.transpose(-1, -2), atol=1e-6, rtol=0.0):
        raise NeuralDataError("Connectome tensor must be symmetric")
    if not torch.allclose(torch.diagonal(connectomes, dim1=-2, dim2=-1), torch.zeros_like(connectomes[..., 0]), atol=1e-7, rtol=0.0):
        raise NeuralDataError("Connectome tensor diagonal must be zero")


def positive_topk_adjacency(connectomes: torch.Tensor, density: float) -> torch.Tensor:
    """Return deterministic binary undirected top-positive-edge graphs.

    The strict upper-triangle order is the deterministic tie-breaker. Density is
    a fraction of all 6,670 possible undirected AAL-116 edges, not a fraction
    of only positive edges. This makes density comparable across participants.
    """
    _validate_connectome_tensor(connectomes)
    if density not in NEURAL_DENSITIES:
        raise NeuralDataError(f"Density {density!r} is outside the frozen density grid")
    participants, nodes, _ = connectomes.shape
    edge_count = math.ceil(density * UNDIRECTED_EDGE_COUNT)
    adjacency = torch.zeros(
        (participants, nodes, nodes), device=connectomes.device, dtype=connectomes.dtype
    )
    if edge_count == 0:
        return adjacency
    edge_i, edge_j = torch.triu_indices(nodes, nodes, offset=1, device=connectomes.device)
    scores = connectomes[:, edge_i, edge_j]
    positive_counts = (scores > 0).sum(dim=1)
    if torch.any(positive_counts < edge_count):
        minimum = int(positive_counts.min().item())
        raise NeuralDataError(
            f"A participant has only {minimum} positive edges, fewer than required {edge_count}"
        )
    # PyTorch's stable sort preserves the canonical triu order when scores tie.
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    selected = order[:, :edge_count]
    batch = torch.arange(participants, device=connectomes.device).unsqueeze(1)
    selected_i = edge_i[selected]
    selected_j = edge_j[selected]
    adjacency[batch, selected_i, selected_j] = 1.0
    adjacency[batch, selected_j, selected_i] = 1.0
    return adjacency


def make_dense_graph_batch(connectomes: torch.Tensor, density: float) -> DenseGraphBatch:
    """Create one dense graph per participant without mixing participants."""
    _validate_connectome_tensor(connectomes)
    features = connectomes.to(dtype=torch.float32)
    return DenseGraphBatch(features=features, adjacency=positive_topk_adjacency(features, density), density=density)


class TrainOnlyFeatureStandardizer:
    """Feature-wise standardization fitted strictly on outer/inner training graphs."""

    def __init__(self, minimum_scale: float = 1e-6) -> None:
        self.minimum_scale = float(minimum_scale)
        self.mean: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None

    def fit(self, features: torch.Tensor) -> "TrainOnlyFeatureStandardizer":
        if features.ndim != 3 or features.shape[-1] != REGION_COUNT:
            raise NeuralDataError("Feature scaler expects [participants, nodes, 116] inputs")
        self.mean = features.mean(dim=(0, 1), keepdim=True)
        scale = features.std(dim=(0, 1), keepdim=True, correction=0)
        self.scale = torch.where(scale >= self.minimum_scale, scale, torch.ones_like(scale))
        return self

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.scale is None:
            raise NeuralDataError("Feature scaler must be fitted on training data before transform")
        if features.ndim != 3 or features.shape[-1] != self.mean.shape[-1]:
            raise NeuralDataError("Feature shape differs from fitted scaler")
        return (features - self.mean.to(features)) / self.scale.to(features)

    def fit_transform(self, features: torch.Tensor) -> torch.Tensor:
        return self.fit(features).transform(features)
