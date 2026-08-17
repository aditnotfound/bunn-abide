"""Leakage-safe, checkpointable engineering training for the neural pilot.

This module has no held-out-site evaluator. It exists to make the Phase 9
pilot reproducible, measure runtime/memory, and validate recovery before a
scientific neural run is designed or launched.
"""

import csv
import hashlib
import json
import os
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from scripts.run_baselines import read_csv, validate_split_contract
from src.neural_data import NeuralCohort, TrainOnlyFeatureStandardizer, make_dense_graph_batch
from src.neural_models import NeuralArchitecture, OperatorName, SharedGraphClassifier


class PilotTrainingError(ValueError):
    """Raised when a pilot contract, split, or checkpoint is unsafe to use."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class PilotPartitions:
    outer_fold: int
    held_out_site: str
    fitting_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray


@dataclass(frozen=True)
class PilotTensors:
    fitting_features: torch.Tensor
    fitting_adjacency: torch.Tensor
    fitting_labels: torch.Tensor
    validation_features: torch.Tensor
    validation_adjacency: torch.Tensor
    validation_labels: torch.Tensor
    scaler_mean: torch.Tensor
    scaler_scale: torch.Tensor


def load_pilot_partitions(
    baseline_table_path: Path,
    outer_splits_path: Path,
    inner_splits_path: Path,
    held_out_site: str,
    inner_validation_fold: int,
) -> PilotPartitions:
    """Load exactly one outer-training/inner-validation engineering partition."""
    table = pd.read_csv(baseline_table_path, dtype={"subject_id": str, "site_id": str})
    table["label_asd"] = pd.to_numeric(table["label_asd"], errors="raise").astype(int)
    outer_rows = read_csv(outer_splits_path)
    inner_rows = read_csv(inner_splits_path)
    outer_fold, train_indices, test_indices, inner_validation = validate_split_contract(
        table, outer_rows, inner_rows, held_out_site
    )
    if inner_validation_fold not in inner_validation:
        raise PilotTrainingError(f"Unknown inner validation fold: {inner_validation_fold}")
    validation_indices = inner_validation[inner_validation_fold]
    fitting_indices = np.setdiff1d(train_indices, validation_indices)
    if (
        np.intersect1d(fitting_indices, validation_indices).size
        or np.intersect1d(fitting_indices, test_indices).size
        or np.intersect1d(validation_indices, test_indices).size
    ):
        raise PilotTrainingError("Pilot fit/validation/test partitions overlap")
    if held_out_site in set(table.loc[fitting_indices, "site_id"]) or held_out_site in set(table.loc[validation_indices, "site_id"]):
        raise PilotTrainingError("Held-out site leaked into pilot tensors")
    return PilotPartitions(
        outer_fold=outer_fold,
        held_out_site=held_out_site,
        fitting_indices=fitting_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )


def build_pilot_tensors(
    cohort: NeuralCohort,
    partitions: PilotPartitions,
    density: float,
    device: torch.device,
) -> PilotTensors:
    """Scale and graph only fitting/validation graphs; outer test is not materialized."""
    raw = torch.from_numpy(cohort.connectomes)
    fitting_raw = raw[torch.as_tensor(partitions.fitting_indices)]
    validation_raw = raw[torch.as_tensor(partitions.validation_indices)]
    scaler = TrainOnlyFeatureStandardizer().fit(fitting_raw)
    fitting_features = scaler.transform(fitting_raw)
    validation_features = scaler.transform(validation_raw)
    fitting_graph = make_dense_graph_batch(fitting_raw, density)
    validation_graph = make_dense_graph_batch(validation_raw, density)
    labels = torch.from_numpy(cohort.labels.astype(np.float32))
    return PilotTensors(
        fitting_features=fitting_features.to(device),
        fitting_adjacency=fitting_graph.adjacency.to(device),
        fitting_labels=labels[torch.as_tensor(partitions.fitting_indices)].to(device),
        validation_features=validation_features.to(device),
        validation_adjacency=validation_graph.adjacency.to(device),
        validation_labels=labels[torch.as_tensor(partitions.validation_indices)].to(device),
        scaler_mean=scaler.mean.detach().cpu(),
        scaler_scale=scaler.scale.detach().cpu(),
    )


def class_balanced_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    positives = labels.sum()
    negatives = labels.numel() - positives
    if positives <= 0 or negatives <= 0:
        raise PilotTrainingError("Fitting partition must contain both classes")
    return (negatives / positives).detach()


def _batches(count: int, batch_size: int, seed: int, device: torch.device) -> list[torch.Tensor]:
    if batch_size <= 0:
        raise PilotTrainingError("batch_size must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(count, generator=generator)
    return [permutation[start : start + batch_size].to(device) for start in range(0, count, batch_size)]


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    tensors: PilotTensors,
    batch_size: int,
    epoch_seed: int,
) -> float:
    """Train one epoch and return weighted BCE loss only, never accuracy."""
    model.train()
    pos_weight = class_balanced_pos_weight(tensors.fitting_labels)
    total_loss = 0.0
    total_rows = 0
    for indices in _batches(len(tensors.fitting_labels), batch_size, epoch_seed, tensors.fitting_labels.device):
        optimizer.zero_grad(set_to_none=True)
        logits = model(tensors.fitting_features[indices], tensors.fitting_adjacency[indices])
        loss = F.binary_cross_entropy_with_logits(logits, tensors.fitting_labels[indices], pos_weight=pos_weight)
        if not torch.isfinite(loss):
            raise PilotTrainingError("Non-finite fitting loss")
        loss.backward()
        optimizer.step()
        rows = len(indices)
        total_loss += float(loss.detach()) * rows
        total_rows += rows
    return total_loss / total_rows


@torch.no_grad()
def validation_loss(model: nn.Module, tensors: PilotTensors, batch_size: int) -> float:
    """Evaluate only BCE loss on the grouped inner validation sites."""
    model.eval()
    pos_weight = class_balanced_pos_weight(tensors.fitting_labels)
    total_loss = 0.0
    total_rows = 0
    for start in range(0, len(tensors.validation_labels), batch_size):
        indices = slice(start, start + batch_size)
        logits = model(tensors.validation_features[indices], tensors.validation_adjacency[indices])
        loss = F.binary_cross_entropy_with_logits(logits, tensors.validation_labels[indices], pos_weight=pos_weight)
        if not torch.isfinite(loss):
            raise PilotTrainingError("Non-finite validation loss")
        rows = len(tensors.validation_labels[indices])
        total_loss += float(loss) * rows
        total_rows += rows
    return total_loss / total_rows


def save_pilot_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist recovery state after an epoch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_pilot_checkpoint(path: Path, immutable_contract: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise PilotTrainingError(f"Checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("immutable_contract") != immutable_contract:
        raise PilotTrainingError("Checkpoint contract differs from the requested pilot")
    if not isinstance(payload.get("completed_epoch"), int) or payload["completed_epoch"] < 0:
        raise PilotTrainingError("Checkpoint lacks a valid completed epoch")
    return payload


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move AdamW state loaded from a CPU checkpoint to the active model device."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def pilot_contract(
    *, run_id: str, operator: OperatorName, density: float, partitions: PilotPartitions,
    architecture: NeuralArchitecture, protocol_path: Path, connectome_path: Path,
    baseline_table_path: Path, outer_splits_path: Path, inner_splits_path: Path, seed: int,
) -> dict[str, Any]:
    """Fields that must not change across a resumed pilot cell."""
    return {
        "run_id": run_id,
        "operator": operator,
        "density": density,
        "outer_fold": partitions.outer_fold,
        "held_out_site": partitions.held_out_site,
        "fitting_subject_count": len(partitions.fitting_indices),
        "validation_subject_count": len(partitions.validation_indices),
        "architecture": architecture.__dict__,
        "seed": seed,
        "sources": {
            "protocol": sha256_file(protocol_path),
            "connectomes": sha256_file(connectome_path),
            "baseline_table": sha256_file(baseline_table_path),
            "outer_splits": sha256_file(outer_splits_path),
            "inner_splits": sha256_file(inner_splits_path),
        },
    }


def run_pilot_cell(
    *,
    run_dir: Path,
    operator: OperatorName,
    density: float,
    tensors: PilotTensors,
    immutable_contract: dict[str, Any],
    architecture: NeuralArchitecture,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    seed: int,
    resume: bool,
    resume_missing_ok: bool = False,
    device: torch.device,
    epoch_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run or resume one loss-only engineering cell without outer-test tensors."""
    if max_epochs <= 0:
        raise PilotTrainingError("max_epochs must be positive")
    checkpoint = run_dir / "checkpoints" / f"{operator}_density_{density:.2f}.pt"
    seed_everything(seed)
    model = SharedGraphClassifier(operator, architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    start_epoch = 0
    history: list[dict[str, Any]] = []
    if resume and checkpoint.exists():
        payload = load_pilot_checkpoint(checkpoint, immutable_contract)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        move_optimizer_state(optimizer, device)
        start_epoch = payload["completed_epoch"] + 1
        history = payload["history"]
    elif resume and not resume_missing_ok:
        raise PilotTrainingError(f"Cannot resume missing checkpoint: {checkpoint}")

    for epoch in range(start_epoch, max_epochs):
        seed_everything(seed + epoch + 1)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_loss = train_epoch(model, optimizer, tensors, batch_size, seed + epoch + 1)
        val_loss = validation_loss(model, tensors, batch_size)
        row = {
            "operator": operator,
            "density": density,
            "epoch": epoch,
            "training_bce_loss": train_loss,
            "validation_bce_loss": val_loss,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        }
        history.append(row)
        save_pilot_checkpoint(
            checkpoint,
            {
                "immutable_contract": immutable_contract,
                "completed_epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
                "scaler_mean": tensors.scaler_mean,
                "scaler_scale": tensors.scaler_scale,
            },
        )
        if epoch_callback is not None:
            epoch_callback(row)
    return history


def write_pilot_history(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["operator", "density", "epoch", "training_bce_loss", "validation_bce_loss", "peak_gpu_memory_bytes"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
