"""Run the score-blind, recovery-safe neural engineering pilot on one GPU.

This is intentionally *not* the scientific neural evaluation.  It materializes
only outer-training and grouped-inner-validation tensors for a single frozen
site split, trains each shared-backbone operator across the fixed density grid,
and logs loss, runtime, memory, and checkpoint-recovery artifacts.  It never
calculates predictions, accuracy, AUROC, or any other held-out-site metric.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Direct execution (``python scripts/run_neural_pilot.py``) puts ``scripts``
# rather than the repository root on ``sys.path``.  Keep the managed launcher
# simple while making internal project imports explicit and deterministic.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.run_baselines import (
    publish_sns_notification,
    sha256_file,
    verify_frozen_hashes,
)
from src.neural_data import load_neural_cohort
from src.neural_models import NeuralArchitecture
from src.neural_training import (
    PilotTrainingError,
    build_pilot_tensors,
    load_pilot_checkpoint,
    load_pilot_partitions,
    pilot_contract,
    run_pilot_cell,
    utc_now,
    write_json_atomic,
    write_pilot_history,
)


CELL_OPERATORS = ("identity", "gcn", "trivial_bundle", "learned_bunn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-protocol", default="configs/neural_pilot_protocol.json")
    parser.add_argument("--operator-contract", default="configs/neural_operator_contract.json")
    parser.add_argument("--baseline-inputs", default="configs/baseline_inputs_and_splits.json")
    parser.add_argument("--table", default="data/processed/abide_i_baseline_table.csv")
    parser.add_argument("--connectomes", default="data/processed/abide_i_connectomes_fisher_z.npz")
    parser.add_argument("--outer-splits", default="data/processed/splits/outer_loso_assignments.csv")
    parser.add_argument("--inner-splits", default="data/processed/splits/inner_grouped_assignments.csv")
    parser.add_argument("--output-root", default="outputs/runs/neural")
    parser.add_argument("--run-id", required=True, help="Stable artifact directory name.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None, help="Engineering override recorded in metadata.")
    parser.add_argument(
        "--stop-after-checkpoints", type=int, default=None,
        help=(
            "Engineering recovery-test hook: intentionally interrupt after this many newly "
            "saved epoch checkpoints. It must be followed by the same run with --resume."
        ),
    )
    parser.add_argument("--notification-topic-arn", default=os.environ.get("BUNN_SNS_TOPIC_ARN"))
    parser.add_argument("--require-notification", action="store_true")
    parser.add_argument("--code-version", default="unknown")
    return parser.parse_args()


def cell_label(operator: str, density: float) -> str:
    return f"{operator}_density_{density:.2f}"


def immutable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "run_id", "run_kind", "pilot_protocol_sha256", "operator_contract_sha256",
        "frozen_input_hashes", "source_hashes", "held_out_site_excluded_from_tensors",
        "inner_validation_fold", "operators", "densities", "seed", "batch_size",
        "max_epochs", "optimizer", "learning_rate", "weight_decay", "architecture",
        "outer_test_evaluation", "code_version",
    )
    return {field: metadata[field] for field in fields}


def update_status(run_dir: Path, **changes: Any) -> dict[str, Any]:
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    status.update(changes)
    status["last_updated_utc"] = utc_now()
    write_json_atomic(status_path, status)
    return status


def configure_or_resume(run_dir: Path, metadata: dict[str, Any], resume: bool) -> None:
    metadata_path = run_dir / "metadata.json"
    if resume:
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Cannot resume missing pilot metadata: {metadata_path}")
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if immutable_metadata(existing) != immutable_metadata(metadata):
            raise PilotTrainingError("Cannot resume: immutable pilot metadata differ")
        existing.update({"status": "running", "resumed_utc": utc_now()})
        write_json_atomic(metadata_path, existing)
        return
    if run_dir.exists():
        raise FileExistsError(f"Pilot directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    write_json_atomic(run_dir / "metadata.json", metadata)
    write_json_atomic(
        run_dir / "status.json",
        {
            "state": "running", "run_id": metadata["run_id"], "pid": os.getpid(),
            "started_utc": metadata["started_utc"], "last_updated_utc": utc_now(),
            "current_operator": None, "current_density": None, "current_epoch": None,
            "current_stage": "initialised", "completed_cells": [], "completed_cell_count": 0,
            "total_cells": len(metadata["operators"]) * len(metadata["densities"]),
            "outer_test_evaluation": False,
        },
    )


def verify_completed_cell(checkpoint_path: Path, contract: dict[str, Any], max_epochs: int) -> list[dict[str, Any]] | None:
    """Return validated history only after a cell reaches its planned last epoch."""
    if not checkpoint_path.is_file():
        return None
    payload = load_pilot_checkpoint(checkpoint_path, contract)
    history = payload.get("history")
    if payload["completed_epoch"] != max_epochs - 1 or not isinstance(history, list) or len(history) != max_epochs:
        return None
    expected_epochs = list(range(max_epochs))
    if [row.get("epoch") for row in history] != expected_epochs:
        raise PilotTrainingError(f"Pilot checkpoint has an invalid epoch history: {checkpoint_path}")
    return history


def notify(args: argparse.Namespace, run_dir: Path, state: str, detail: str) -> dict[str, Any]:
    label = state.upper()
    return publish_sns_notification(
        run_dir,
        args.notification_topic_arn,
        f"BuNN neural pilot {label}: {args.run_id}",
        f"Neural engineering pilot {args.run_id} {detail}. Check status.json and metadata.json before interpreting anything.",
    )


def construct_metadata(
    args: argparse.Namespace, protocol: dict[str, Any], operator_contract: dict[str, Any],
    frozen_hashes: dict[str, Any], paths: dict[str, Path], max_epochs: int,
) -> dict[str, Any]:
    return {
        "run_id": args.run_id,
        "run_kind": "engineering_pilot_score_blind",
        "status": "running",
        "started_utc": utc_now(),
        "code_version": args.code_version,
        "pilot_protocol_sha256": sha256_file(paths["pilot_protocol"]),
        "operator_contract_sha256": sha256_file(paths["operator_contract"]),
        "frozen_input_hashes": frozen_hashes,
        "source_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "held_out_site_excluded_from_tensors": protocol["held_out_site_excluded_from_tensors"],
        "inner_validation_fold": int(protocol["inner_validation_fold"]),
        "operators": list(protocol["operators"]),
        "densities": list(protocol["densities"]),
        "seed": int(protocol["seed"]),
        "batch_size": int(protocol["batch_size"]),
        "max_epochs": max_epochs,
        "optimizer": protocol["optimizer"],
        "learning_rate": float(protocol["learning_rate"]),
        "weight_decay": float(protocol["weight_decay"]),
        "architecture": {
            "input_dim": int(operator_contract["cohort"]["node_feature_dimension"]),
            "hidden_dim": int(operator_contract["shared_backbone"]["pilot_hidden_dimension"]),
            "layers": int(operator_contract["shared_backbone"]["pilot_layers"]),
            "bundles": int(operator_contract["operators"]["learned_bunn"]["bundle_count"]),
            "channels": int(operator_contract["operators"]["learned_bunn"]["vector_field_channels"]),
            "dropout": float(operator_contract["shared_backbone"]["pilot_dropout"]),
            "diffusion_time": 1.0,
        },
        "outer_test_evaluation": False,
        "selection_metric": protocol["selection_metric"],
        "recovery_test": {
            "intentional_interruption_requested": args.stop_after_checkpoints is not None,
            "resumed": False,
        },
        "interpretation": (
            "Engineering-only validation of numerical stability, resource use, and recovery. "
            "No held-out-site predictive metric, prediction, or final model selection is calculated."
        ),
    }


def run(args: argparse.Namespace) -> Path:
    if args.require_notification and not args.notification_topic_arn:
        raise ValueError("--require-notification needs --notification-topic-arn or BUNN_SNS_TOPIC_ARN")
    if not torch.cuda.is_available():
        raise RuntimeError("The neural pilot requires CUDA; no CUDA device is available")
    if args.stop_after_checkpoints is not None and args.stop_after_checkpoints <= 0:
        raise ValueError("--stop-after-checkpoints must be positive")

    paths = {
        "pilot_protocol": Path(args.pilot_protocol), "operator_contract": Path(args.operator_contract),
        "baseline_inputs": Path(args.baseline_inputs), "baseline_table": Path(args.table),
        "connectomes": Path(args.connectomes), "outer_splits": Path(args.outer_splits),
        "inner_splits": Path(args.inner_splits),
    }
    protocol = json.loads(paths["pilot_protocol"].read_text(encoding="utf-8"))
    operator_contract = json.loads(paths["operator_contract"].read_text(encoding="utf-8"))
    if protocol.get("pilot_version") != 1 or operator_contract.get("contract_version") != 1:
        raise ValueError("Unsupported neural pilot or operator contract version")
    if tuple(protocol.get("operators", [])) != CELL_OPERATORS:
        raise ValueError("Pilot operator ordering does not match the frozen shared-operator contract")
    if protocol.get("outer_test_evaluation") is not False or protocol.get("automatic_result_analysis") is not False:
        raise ValueError("Pilot must explicitly disable outer-test evaluation and result analysis")
    if list(protocol["densities"]) != list(operator_contract["graph_construction"]["densities"]):
        raise ValueError("Pilot density grid differs from neural operator contract")
    max_epochs = int(args.max_epochs) if args.max_epochs is not None else int(protocol["max_epochs"])
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    frozen_hashes = verify_frozen_hashes(
        paths["baseline_inputs"], paths["baseline_table"], paths["outer_splits"], paths["inner_splits"]
    )
    metadata = construct_metadata(args, protocol, operator_contract, frozen_hashes, paths, max_epochs)
    run_dir = Path(args.output_root) / args.run_id
    configure_or_resume(run_dir, metadata, args.resume)
    if args.require_notification:
        start = notify(args, run_dir, "started", "started on the GPU")
        if start["status"] != "published":
            raise RuntimeError("Required SNS start notification could not be published")

    device = torch.device("cuda")
    cohort = load_neural_cohort(paths["connectomes"], paths["baseline_table"])
    partitions = load_pilot_partitions(
        paths["baseline_table"], paths["outer_splits"], paths["inner_splits"],
        str(protocol["held_out_site_excluded_from_tensors"]), int(protocol["inner_validation_fold"]),
    )
    architecture = NeuralArchitecture(**metadata["architecture"])
    all_rows: list[dict[str, Any]] = []
    completed_cells: list[str] = []
    cell_runtimes: dict[str, float] = {}
    newly_checkpointed_epochs = 0
    for density in protocol["densities"]:
        tensors = build_pilot_tensors(cohort, partitions, float(density), device)
        for operator in protocol["operators"]:
            label = cell_label(str(operator), float(density))
            contract = pilot_contract(
                run_id=args.run_id, operator=operator, density=float(density), partitions=partitions,
                architecture=architecture, protocol_path=paths["pilot_protocol"], connectome_path=paths["connectomes"],
                baseline_table_path=paths["baseline_table"], outer_splits_path=paths["outer_splits"],
                inner_splits_path=paths["inner_splits"], seed=int(protocol["seed"]),
            )
            checkpoint_path = run_dir / "checkpoints" / f"{label}.pt"
            existing = verify_completed_cell(checkpoint_path, contract, max_epochs) if args.resume else None
            if existing is not None:
                all_rows.extend(existing)
                completed_cells.append(label)
                update_status(
                    run_dir, state="running", current_operator=None, current_density=None, current_epoch=None,
                    current_stage="validated_completed_cell", completed_cells=completed_cells,
                    completed_cell_count=len(completed_cells), total_cells=len(protocol["operators"]) * len(protocol["densities"]),
                )
                continue
            update_status(
                run_dir, state="running", current_operator=operator, current_density=float(density),
                current_epoch=None, current_stage="training_loss_only", completed_cells=completed_cells,
                completed_cell_count=len(completed_cells), total_cells=len(protocol["operators"]) * len(protocol["densities"]),
            )
            started = time.perf_counter()

            def record_epoch(row: dict[str, Any]) -> None:
                nonlocal newly_checkpointed_epochs
                update_status(
                    run_dir, state="running", current_operator=operator, current_density=float(density),
                    current_epoch=int(row["epoch"]), current_stage="checkpointed_epoch_loss_only",
                    completed_cells=completed_cells, completed_cell_count=len(completed_cells),
                    total_cells=len(protocol["operators"]) * len(protocol["densities"]),
                )
                newly_checkpointed_epochs += 1
                if (
                    args.stop_after_checkpoints is not None
                    and newly_checkpointed_epochs >= args.stop_after_checkpoints
                ):
                    raise InterruptedError(
                        "intentional recovery test after a durable epoch checkpoint"
                    )

            rows = run_pilot_cell(
                run_dir=run_dir, operator=operator, density=float(density), tensors=tensors,
                immutable_contract=contract, architecture=architecture, learning_rate=float(protocol["learning_rate"]),
                weight_decay=float(protocol["weight_decay"]), batch_size=int(protocol["batch_size"]),
                max_epochs=max_epochs, seed=int(protocol["seed"]), resume=args.resume, device=device,
                epoch_callback=record_epoch,
            )
            cell_runtimes[label] = time.perf_counter() - started
            history = verify_completed_cell(checkpoint_path, contract, max_epochs)
            if history is None:
                raise PilotTrainingError(f"Cell did not produce a verified complete checkpoint: {label}")
            all_rows.extend(history)
            completed_cells.append(label)
            update_status(
                run_dir, state="running", current_operator=None, current_density=None, current_epoch=None,
                current_stage="cell_checkpoint_verified", completed_cells=completed_cells,
                completed_cell_count=len(completed_cells), total_cells=len(protocol["operators"]) * len(protocol["densities"]),
            )
        del tensors
        torch.cuda.empty_cache()

    expected_rows = len(protocol["operators"]) * len(protocol["densities"]) * max_epochs
    if len(all_rows) != expected_rows:
        raise PilotTrainingError(f"Pilot expected {expected_rows} loss rows but found {len(all_rows)}")
    write_pilot_history(run_dir / "pilot_loss_history.csv", all_rows)
    summary = {
        "run_id": args.run_id, "run_kind": "engineering_pilot_score_blind",
        "cells_completed": completed_cells, "cell_count": len(completed_cells),
        "loss_history_rows": len(all_rows), "cell_runtimes_seconds": cell_runtimes,
        "max_gpu_memory_bytes": max(int(row["peak_gpu_memory_bytes"]) for row in all_rows),
        "outer_test_evaluation": False,
        "notice": "Loss-only engineering artifacts. No predictive result or model-selection conclusion is present.",
    }
    write_json_atomic(run_dir / "summary.json", summary)
    metadata_path = run_dir / "metadata.json"
    final_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    prior_recovery = final_metadata.get("recovery_test", {})
    final_metadata.update({
        "status": "complete", "completed_utc": utc_now(),
        "recovery_test": {
            "intentional_interruption_requested": bool(
                prior_recovery.get("intentional_interruption_requested")
                or args.stop_after_checkpoints is not None
            ),
            "resumed": bool(args.resume),
            "new_epoch_checkpoints_this_invocation": newly_checkpointed_epochs,
        },
        "artifact_hashes": {
            "pilot_loss_history.csv": sha256_file(run_dir / "pilot_loss_history.csv"),
            "summary.json": sha256_file(run_dir / "summary.json"),
        },
    })
    write_json_atomic(metadata_path, final_metadata)
    update_status(
        run_dir, state="complete", current_operator=None, current_density=None, current_epoch=None,
        current_stage="completion_audited_loss_only", completed_cells=completed_cells,
        completed_cell_count=len(completed_cells), total_cells=len(protocol["operators"]) * len(protocol["densities"]),
    )
    notification = notify(args, run_dir, "complete", "completed all loss-only pilot cells")
    final_metadata["notification_status"] = notification["status"]
    write_json_atomic(metadata_path, final_metadata)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return run_dir


def mark_failed(args: argparse.Namespace, error: Exception) -> None:
    run_dir = Path(args.output_root) / args.run_id
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.is_file():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "status": "interrupted" if isinstance(error, InterruptedError) else "failed",
        "failed_utc": utc_now(), "failure_type": type(error).__name__, "failure_message": str(error),
    })
    write_json_atomic(metadata_path, metadata)
    update_status(run_dir, state=metadata["status"], current_stage="terminal_error", failure_type=type(error).__name__, failure_message=str(error))
    notification = notify(args, run_dir, metadata["status"], f"ended with {type(error).__name__}")
    metadata["notification_status"] = notification["status"]
    write_json_atomic(metadata_path, metadata)


def handle_sigterm(_signal: int, _frame: Any) -> None:
    raise InterruptedError("received SIGTERM")


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        run(args)
    except Exception as error:
        mark_failed(args, error)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
