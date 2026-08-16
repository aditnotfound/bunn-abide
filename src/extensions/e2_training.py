"""Checkpointable, score-blind training utilities for E2 synthetic geometry."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.nn import functional as F

from src.extensions.e2_synthetic import E2Dataset, E2Operator, SyntheticGraphClassifier, relative_transport_error


class E2TrainingError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    replace_with_retry(temporary, path)


def replace_with_retry(source: Path, destination: Path, attempts: int = 20) -> None:
    """Survive short OneDrive/antivirus locks without weakening atomic writes."""
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.05 * (attempt + 1))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batches(indices: torch.Tensor, batch_size: int, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    order = indices[torch.randperm(len(indices), generator=generator)]
    return [order[start:start + batch_size] for start in range(0, len(order), batch_size)]


def _loss_for_indices(
    model: SyntheticGraphClassifier, dataset: E2Dataset, indices: torch.Tensor,
    batch_size: int, device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            features = dataset.features[batch].to(device)
            labels = dataset.labels[batch].to(device)
            maps = dataset.true_maps[batch].to(device)
            logits = model(features, oracle_maps=maps)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            if not torch.isfinite(loss):
                raise E2TrainingError("Non-finite validation loss")
            total += float(loss) * len(batch)
            count += len(batch)
    return total / count


@dataclass(frozen=True)
class E2CellResult:
    epochs_completed: int
    best_epoch: int
    runtime_seconds: float
    prediction_path: Path | None
    manifest_path: Path


def run_e2_cell(
    *, dataset: E2Dataset, operator: E2Operator, model_seed: int, cell_dir: Path,
    protocol_sha256: str, learning_rate: float, weight_decay: float, batch_size: int,
    maximum_epochs: int, minimum_epochs: int, patience: int, gradient_clip: float,
    device: torch.device, evaluate_test: bool, resume: bool,
    epoch_callback: Callable[[dict[str, Any]], None] | None = None,
) -> E2CellResult:
    cell_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = cell_dir / "test_predictions.npz"
    manifest_path = cell_dir / "cell_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("state") != "sealed_score_blind":
            raise E2TrainingError("Existing cell manifest is not sealed")
        if evaluate_test and not prediction_path.is_file():
            raise E2TrainingError("Sealed scientific cell is missing predictions")
        return E2CellResult(
            epochs_completed=int(manifest["epochs_completed"]),
            best_epoch=int(manifest["best_epoch"]),
            runtime_seconds=float(manifest["runtime_seconds"]),
            prediction_path=prediction_path if evaluate_test else None,
            manifest_path=manifest_path,
        )

    immutable = {
        "protocol_sha256": protocol_sha256,
        "family": dataset.family,
        "data_seed": dataset.seed,
        "transport_noise_degrees": dataset.transport_noise_degrees,
        "operator": operator,
        "model_seed": model_seed,
        "samples": len(dataset.labels),
        "nodes": dataset.features.shape[1],
        "feature_dimension": dataset.features.shape[2],
        "evaluate_test": evaluate_test,
    }
    seed_everything(model_seed)
    model = SyntheticGraphClassifier(operator, dataset.adjacency).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    checkpoint_path = cell_dir / "checkpoint.pt"
    start_epoch = 0
    best_epoch = -1
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    if resume and checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("immutable") != immutable:
            raise E2TrainingError("Checkpoint immutable contract mismatch")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        start_epoch = int(saved["completed_epoch"]) + 1
        best_epoch = int(saved["best_epoch"])
        best_loss = float(saved["best_loss"])
        best_state = saved["best_state"]
        stale_epochs = int(saved["stale_epochs"])
        history = saved["history"]

    started = time.monotonic()
    epochs_completed = start_epoch
    for epoch in range(start_epoch, maximum_epochs):
        model.train()
        total_loss = 0.0
        rows = 0
        for batch in _batches(dataset.train_indices, batch_size, model_seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            features = dataset.features[batch].to(device)
            labels = dataset.labels[batch].to(device)
            maps = dataset.true_maps[batch].to(device)
            logits = model(features, oracle_maps=maps)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            if not torch.isfinite(loss):
                raise E2TrainingError("Non-finite fitting loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            if not torch.isfinite(norm):
                raise E2TrainingError("Non-finite gradient norm")
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
            rows += len(batch)
        validation_loss = _loss_for_indices(model, dataset, dataset.validation_indices, batch_size, device)
        training_loss = total_loss / rows
        history.append({"epoch": epoch + 1, "training_loss": training_loss, "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        checkpoint = {
            "immutable": immutable,
            "completed_epoch": epoch,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "best_state": best_state,
            "stale_epochs": stale_epochs,
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "history": history,
        }
        temporary = checkpoint_path.parent / f".{checkpoint_path.name}.{os.getpid()}.tmp"
        torch.save(checkpoint, temporary)
        replace_with_retry(temporary, checkpoint_path)
        epochs_completed = epoch + 1
        if epoch_callback is not None:
            epoch_callback({"epoch": epoch + 1, "maximum_epochs": maximum_epochs})
        if epoch + 1 >= minimum_epochs and stale_epochs >= patience:
            break
    if best_state is None or best_epoch < 0:
        raise E2TrainingError("Training produced no finite validation checkpoint")
    model.load_state_dict(best_state)

    prediction_hash = None
    if evaluate_test:
        model.eval()
        probabilities: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        transport_errors: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(dataset.test_indices), batch_size):
                batch = dataset.test_indices[start:start + batch_size]
                features = dataset.features[batch].to(device)
                maps = dataset.true_maps[batch].to(device)
                logits, fitted_maps = model(features, oracle_maps=maps, return_maps=True)
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
                labels.append(dataset.labels[batch].numpy())
                transport_errors.append(
                    relative_transport_error(fitted_maps, maps, dataset.adjacency.to(device)).cpu().numpy()
                )
        temporary = prediction_path.parent / f".{prediction_path.name}.{os.getpid()}.tmp.npz"
        np.savez_compressed(
            temporary,
            probability=np.concatenate(probabilities),
            label=np.concatenate(labels),
            transport_error=np.concatenate(transport_errors),
            test_index=dataset.test_indices.numpy(),
        )
        replace_with_retry(temporary, prediction_path)
        prediction_hash = sha256_file(prediction_path)

    runtime = time.monotonic() - started
    history_path = cell_dir / "training_history.json"
    write_json_atomic(history_path, {"history": history})
    manifest = {
        "state": "sealed_score_blind",
        "immutable": immutable,
        "parameter_count": model.parameter_count(),
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch + 1,
        "runtime_seconds": runtime,
        "history_sha256": sha256_file(history_path),
        "prediction_sha256": prediction_hash,
    }
    write_json_atomic(manifest_path, manifest)
    return E2CellResult(
        epochs_completed=epochs_completed,
        best_epoch=best_epoch + 1,
        runtime_seconds=runtime,
        prediction_path=prediction_path if evaluate_test else None,
        manifest_path=manifest_path,
    )


def audit_e2_cell(cell_dir: Path, *, require_predictions: bool) -> dict[str, Any]:
    manifest_path = cell_dir / "cell_manifest.json"
    if not manifest_path.is_file():
        raise E2TrainingError("Cell manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("state") != "sealed_score_blind":
        raise E2TrainingError("Cell is not sealed")
    history_path = cell_dir / "training_history.json"
    if sha256_file(history_path) != manifest.get("history_sha256"):
        raise E2TrainingError("Training-history hash mismatch")
    prediction_path = cell_dir / "test_predictions.npz"
    if require_predictions:
        if sha256_file(prediction_path) != manifest.get("prediction_sha256"):
            raise E2TrainingError("Prediction hash mismatch")
        values = np.load(prediction_path)
        required = {"probability", "label", "transport_error", "test_index"}
        if set(values.files) != required:
            raise E2TrainingError("Prediction archive fields differ from contract")
        lengths = {len(values[name]) for name in required}
        if len(lengths) != 1 or next(iter(lengths)) <= 0:
            raise E2TrainingError("Prediction arrays are empty or misaligned")
        if not np.isfinite(values["probability"]).all() or not np.isfinite(values["transport_error"]).all():
            raise E2TrainingError("Prediction archive contains non-finite values")
        if ((values["probability"] < 0) | (values["probability"] > 1)).any():
            raise E2TrainingError("Probability outside [0,1]")
        if set(np.unique(values["label"])) != {0.0, 1.0}:
            raise E2TrainingError("Test archive must contain both classes")
        if len(np.unique(values["test_index"])) != len(values["test_index"]):
            raise E2TrainingError("Duplicate test index")
    elif prediction_path.exists() or manifest.get("prediction_sha256") is not None:
        raise E2TrainingError("Score-blind smoke unexpectedly materialized test predictions")
    return {
        "state": "audit_passed_score_blind",
        "manifest_sha256": sha256_file(manifest_path),
        "prediction_sha256": manifest.get("prediction_sha256"),
    }
