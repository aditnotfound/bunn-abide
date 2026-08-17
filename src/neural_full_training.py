"""Leakage-safe training primitives for the full nested neural evaluation."""

import copy
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.nn import functional as F

from src.neural_data import NeuralCohort, TrainOnlyFeatureStandardizer, make_dense_graph_batch
from src.neural_models import NeuralArchitecture, OperatorName, SharedGraphClassifier
from src.neural_training import (
    PilotTrainingError,
    class_balanced_pos_weight,
    move_optimizer_state,
    save_pilot_checkpoint,
    seed_everything,
)


class FullTrainingError(ValueError):
    """Raised when a full-run fit, split, checkpoint, or metric is invalid."""


@dataclass(frozen=True)
class FitTensors:
    fitting_features: torch.Tensor
    fitting_adjacency: torch.Tensor
    fitting_labels: torch.Tensor
    evaluation_features: torch.Tensor
    evaluation_adjacency: torch.Tensor
    evaluation_labels: torch.Tensor
    evaluation_sites: np.ndarray
    evaluation_subject_ids: np.ndarray
    scaler_mean: torch.Tensor
    scaler_scale: torch.Tensor


def build_fit_tensors(
    cohort: NeuralCohort,
    fitting_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    density: float,
    device: torch.device,
) -> FitTensors:
    """Fit scaling only on fitting rows; graph topology always uses raw FC."""
    fitting_indices = np.asarray(fitting_indices, dtype=int)
    evaluation_indices = np.asarray(evaluation_indices, dtype=int)
    if not len(fitting_indices) or not len(evaluation_indices):
        raise FullTrainingError("Fitting and evaluation partitions must be nonempty")
    if np.intersect1d(fitting_indices, evaluation_indices).size:
        raise FullTrainingError("Fitting and evaluation partitions overlap")
    raw = torch.from_numpy(cohort.connectomes)
    fitting_raw = raw[torch.as_tensor(fitting_indices)]
    evaluation_raw = raw[torch.as_tensor(evaluation_indices)]
    scaler = TrainOnlyFeatureStandardizer().fit(fitting_raw)
    fitting_graph = make_dense_graph_batch(fitting_raw, density)
    evaluation_graph = make_dense_graph_batch(evaluation_raw, density)
    labels = torch.from_numpy(cohort.labels.astype(np.float32))
    return FitTensors(
        fitting_features=scaler.transform(fitting_raw).to(device),
        fitting_adjacency=fitting_graph.adjacency.to(device),
        fitting_labels=labels[torch.as_tensor(fitting_indices)].to(device),
        evaluation_features=scaler.transform(evaluation_raw).to(device),
        evaluation_adjacency=evaluation_graph.adjacency.to(device),
        evaluation_labels=labels[torch.as_tensor(evaluation_indices)].to(device),
        evaluation_sites=cohort.site_ids[evaluation_indices].astype(str),
        evaluation_subject_ids=cohort.subject_ids[evaluation_indices].astype(str),
        scaler_mean=scaler.mean.detach().cpu(),
        scaler_scale=scaler.scale.detach().cpu(),
    )


def deterministic_batches(count: int, batch_size: int, seed: int, device: torch.device) -> list[torch.Tensor]:
    if count <= 0 or batch_size <= 0:
        raise FullTrainingError("Batch count and size must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(count, generator=generator)
    return [order[start : start + batch_size].to(device) for start in range(0, count, batch_size)]


def train_epoch_clipped(
    model: SharedGraphClassifier,
    optimizer: torch.optim.Optimizer,
    tensors: FitTensors,
    batch_size: int,
    epoch_seed: int,
    gradient_clip_norm: float,
) -> tuple[float, float]:
    model.train()
    pos_weight = class_balanced_pos_weight(tensors.fitting_labels)
    total_loss = 0.0
    total_rows = 0
    maximum_gradient_norm = 0.0
    for indices in deterministic_batches(len(tensors.fitting_labels), batch_size, epoch_seed, tensors.fitting_labels.device):
        optimizer.zero_grad(set_to_none=True)
        logits = model(tensors.fitting_features[indices], tensors.fitting_adjacency[indices])
        loss = F.binary_cross_entropy_with_logits(logits, tensors.fitting_labels[indices], pos_weight=pos_weight)
        if not torch.isfinite(loss):
            raise FullTrainingError("Non-finite fitting loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        if not torch.isfinite(gradient_norm):
            raise FullTrainingError("Non-finite gradient norm")
        optimizer.step()
        rows = len(indices)
        total_loss += float(loss.detach()) * rows
        total_rows += rows
        maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm.detach()))
    return total_loss / total_rows, maximum_gradient_norm


@torch.no_grad()
def evaluation_logits(model: SharedGraphClassifier, tensors: FitTensors, batch_size: int) -> torch.Tensor:
    model.eval()
    output: list[torch.Tensor] = []
    for start in range(0, len(tensors.evaluation_labels), batch_size):
        stop = start + batch_size
        output.append(model(tensors.evaluation_features[start:stop], tensors.evaluation_adjacency[start:stop]))
    logits = torch.cat(output)
    if not torch.isfinite(logits).all():
        raise FullTrainingError("Non-finite evaluation logits")
    return logits


@torch.no_grad()
def mean_site_bce(model: SharedGraphClassifier, tensors: FitTensors, batch_size: int) -> float:
    logits = evaluation_logits(model, tensors, batch_size)
    pos_weight = class_balanced_pos_weight(tensors.fitting_labels)
    losses: list[float] = []
    for site in sorted(set(tensors.evaluation_sites.tolist())):
        mask_np = tensors.evaluation_sites == site
        mask = torch.as_tensor(mask_np, device=logits.device)
        loss = F.binary_cross_entropy_with_logits(logits[mask], tensors.evaluation_labels[mask], pos_weight=pos_weight)
        if not torch.isfinite(loss):
            raise FullTrainingError("Non-finite validation-site loss")
        losses.append(float(loss))
    return float(np.mean(losses))


def state_to_cpu(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def load_fit_checkpoint(path: Path, immutable_contract: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FullTrainingError(f"Checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("immutable_contract") != immutable_contract:
        raise FullTrainingError("Fit checkpoint immutable contract mismatch")
    if payload.get("state") not in {"running", "complete"}:
        raise FullTrainingError("Fit checkpoint has an invalid state")
    return payload


def train_early_stopped(
    *,
    checkpoint_path: Path,
    immutable_contract: dict[str, Any],
    operator: OperatorName,
    architecture: NeuralArchitecture,
    tensors: FitTensors,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    maximum_epochs: int,
    minimum_epochs: int,
    patience: int,
    minimum_delta: float,
    gradient_clip_norm: float,
    seed: int,
    resume: bool,
    device: torch.device,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    seed_everything(seed)
    model = SharedGraphClassifier(operator, architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    start_epoch = 0
    if resume and checkpoint_path.exists():
        payload = load_fit_checkpoint(checkpoint_path, immutable_contract)
        if payload["state"] == "complete":
            return payload
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        move_optimizer_state(optimizer, device)
        history = payload["history"]
        best_state = payload["best_model_state"]
        best_loss = float(payload["best_validation_site_bce"])
        best_epoch = int(payload["best_epoch"])
        epochs_without_improvement = int(payload["epochs_without_improvement"])
        start_epoch = int(payload["completed_epoch"]) + 1

    for epoch in range(start_epoch, maximum_epochs):
        seed_everything(seed + epoch + 1)
        train_loss, max_gradient = train_epoch_clipped(
            model, optimizer, tensors, batch_size, seed + epoch + 1, gradient_clip_norm
        )
        validation_loss = mean_site_bce(model, tensors, batch_size)
        improved = validation_loss < best_loss - minimum_delta
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = state_to_cpu(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append(
            {
                "epoch": epoch,
                "training_bce_loss": train_loss,
                "validation_mean_site_bce": validation_loss,
                "maximum_gradient_norm": max_gradient,
                "improved": improved,
            }
        )
        stopped = epoch + 1 >= minimum_epochs and epochs_without_improvement >= patience
        payload = {
            "state": "complete" if stopped or epoch + 1 == maximum_epochs else "running",
            "immutable_contract": immutable_contract,
            "completed_epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_site_bce": best_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "model_state": state_to_cpu(model.state_dict()),
            "best_model_state": best_state,
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "scaler_mean": tensors.scaler_mean,
            "scaler_scale": tensors.scaler_scale,
            "stop_reason": "patience" if stopped else ("maximum_epochs" if epoch + 1 == maximum_epochs else None),
        }
        save_pilot_checkpoint(checkpoint_path, payload)
        if progress is not None:
            progress({"epoch": epoch, "state": payload["state"]})
        if payload["state"] == "complete":
            if best_state is None or best_epoch < 0:
                raise FullTrainingError("Early-stopped fit completed without a best checkpoint")
            return payload
    raise FullTrainingError("Early-stopped fit left its loop without completion")


def train_fixed_epochs(
    *,
    checkpoint_path: Path,
    immutable_contract: dict[str, Any],
    operator: OperatorName,
    architecture: NeuralArchitecture,
    tensors: FitTensors,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    gradient_clip_norm: float,
    seed: int,
    resume: bool,
    device: torch.device,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    seed_everything(seed)
    model = SharedGraphClassifier(operator, architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, Any]] = []
    start_epoch = 0
    if resume and checkpoint_path.exists():
        payload = load_fit_checkpoint(checkpoint_path, immutable_contract)
        if payload["state"] == "complete":
            return payload
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        move_optimizer_state(optimizer, device)
        history = payload["history"]
        start_epoch = int(payload["completed_epoch"]) + 1
    for epoch in range(start_epoch, epochs):
        seed_everything(seed + epoch + 1)
        train_loss, max_gradient = train_epoch_clipped(
            model, optimizer, tensors, batch_size, seed + epoch + 1, gradient_clip_norm
        )
        history.append({"epoch": epoch, "training_bce_loss": train_loss, "maximum_gradient_norm": max_gradient})
        state = "complete" if epoch + 1 == epochs else "running"
        payload = {
            "state": state,
            "immutable_contract": immutable_contract,
            "completed_epoch": epoch,
            "model_state": state_to_cpu(model.state_dict()),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "scaler_mean": tensors.scaler_mean,
            "scaler_scale": tensors.scaler_scale,
        }
        save_pilot_checkpoint(checkpoint_path, payload)
        if progress is not None:
            progress({"epoch": epoch, "state": state})
    return payload


@torch.no_grad()
def site_classification_rows(
    model: SharedGraphClassifier, tensors: FitTensors, batch_size: int, threshold: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    probabilities = torch.sigmoid(evaluation_logits(model, tensors, batch_size)).cpu().numpy()
    labels = tensors.evaluation_labels.cpu().numpy().astype(int)
    rows: list[dict[str, Any]] = []
    for site in sorted(set(tensors.evaluation_sites.tolist())):
        mask = tensors.evaluation_sites == site
        site_labels = labels[mask]
        site_probabilities = probabilities[mask]
        predicted = (site_probabilities >= threshold).astype(int)
        if set(site_labels.tolist()) != {0, 1}:
            raise FullTrainingError(f"Evaluation site {site} lacks both classes")
        rows.append(
            {
                "site_id": site,
                "participants": int(mask.sum()),
                "asd": int(site_labels.sum()),
                "control": int((site_labels == 0).sum()),
                "balanced_accuracy": float(balanced_accuracy_score(site_labels, predicted)),
                "auroc": float(roc_auc_score(site_labels, site_probabilities)),
                "sensitivity": float(((predicted == 1) & (site_labels == 1)).sum() / (site_labels == 1).sum()),
                "specificity": float(((predicted == 0) & (site_labels == 0)).sum() / (site_labels == 0).sum()),
            }
        )
    return probabilities, rows


@torch.no_grad()
def diagnostic_rows(
    model: SharedGraphClassifier,
    tensors: FitTensors,
    batch_size: int,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    layer_labels = ("encoder", "layer_1", "layer_2")
    for start in range(0, len(tensors.evaluation_labels), batch_size):
        stop = start + batch_size
        _, diagnostics = model(
            tensors.evaluation_features[start:stop],
            tensors.evaluation_adjacency[start:stop],
            return_diagnostics=True,
            include_encoder_diagnostics=True,
        )
        if len(diagnostics) != len(layer_labels):
            raise FullTrainingError("Unexpected diagnostic layer count")
        for local_index, subject_id in enumerate(tensors.evaluation_subject_ids[start:stop]):
            for layer_label, layer_metrics in zip(layer_labels, diagnostics, strict=True):
                row = {"subject_id": str(subject_id), "layer": layer_label}
                for name, values in layer_metrics.items():
                    value = float(values[local_index].detach().cpu())
                    if not math.isfinite(value):
                        raise FullTrainingError("Non-finite representation diagnostic")
                    row[name] = value
                rows.append(row)
    return rows


def instantiate_best_model(
    operator: OperatorName,
    architecture: NeuralArchitecture,
    state: dict[str, torch.Tensor],
    device: torch.device,
) -> SharedGraphClassifier:
    model = SharedGraphClassifier(operator, architecture).to(device)
    model.load_state_dict(state)
    return model


def selected_candidate_index(candidate_rows: list[dict[str, Any]]) -> int:
    if not candidate_rows:
        raise FullTrainingError("No tuning candidates were scored")
    ordered = sorted(
        candidate_rows,
        key=lambda row: (
            -float(row["inner_mean_site_balanced_accuracy"]),
            -float(row["weight_decay"]),
            float(row["learning_rate"]),
            int(row["candidate_index"]),
        ),
    )
    return int(ordered[0]["candidate_index"])


def selected_final_epoch(one_based_best_epochs: list[int]) -> int:
    if not one_based_best_epochs or any(epoch <= 0 for epoch in one_based_best_epochs):
        raise FullTrainingError("Best epochs must be positive one-based values")
    return int(math.ceil(float(np.median(one_based_best_epochs))))
