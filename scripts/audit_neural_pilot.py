"""Independently audit a score-blind neural engineering-pilot artifact.

The auditor deliberately verifies only run integrity, numerical finiteness, and
the absence of held-out-site prediction artifacts.  It never calculates or
prints an accuracy, AUROC, predicted label, probability, or loss value.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch


HISTORY_FIELDS = (
    "operator", "density", "epoch", "training_bce_loss", "validation_bce_loss",
    "peak_gpu_memory_bytes",
)
PROHIBITED_ARTIFACT_TOKENS = ("prediction", "probability", "metric", "auroc", "accuracy")


class PilotAuditError(ValueError):
    """Raised when a purported engineering-only pilot is incomplete or unsafe."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PilotAuditError(f"Unreadable JSON artifact: {path}") from error


def expected_cells(metadata: dict[str, Any]) -> list[tuple[str, float]]:
    operators = metadata.get("operators")
    densities = metadata.get("densities")
    if not isinstance(operators, list) or not operators or not isinstance(densities, list) or not densities:
        raise PilotAuditError("Metadata lacks nonempty operators or density grid")
    try:
        cells = [(str(operator), float(density)) for density in densities for operator in operators]
    except (TypeError, ValueError) as error:
        raise PilotAuditError("Metadata has an invalid operator/density grid") from error
    if len(set(cells)) != len(cells):
        raise PilotAuditError("Metadata repeats a pilot cell")
    return cells


def read_history(path: Path, expected_epochs: int, cells: list[tuple[str, float]]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != HISTORY_FIELDS:
            raise PilotAuditError("Pilot loss-history schema differs from the frozen schema")
        rows = list(reader)
    if len(rows) != len(cells) * expected_epochs:
        raise PilotAuditError("Pilot loss-history row count is incomplete")
    expected_cell_set = set(cells)
    grouped: dict[tuple[str, float], list[int]] = {cell: [] for cell in cells}
    for row in rows:
        if set(row) != set(HISTORY_FIELDS):
            raise PilotAuditError("Pilot history has an unexpected field")
        try:
            cell = (row["operator"], float(row["density"]))
            epoch = int(row["epoch"])
            training_loss = float(row["training_bce_loss"])
            validation_loss = float(row["validation_bce_loss"])
            peak_memory = int(row["peak_gpu_memory_bytes"])
        except (TypeError, ValueError) as error:
            raise PilotAuditError("Pilot history has nonnumeric engineering fields") from error
        if cell not in expected_cell_set:
            raise PilotAuditError("Pilot history contains an unknown operator/density cell")
        if not math.isfinite(training_loss) or not math.isfinite(validation_loss) or peak_memory < 0:
            raise PilotAuditError("Pilot history contains an invalid numerical-stability record")
        grouped[cell].append(epoch)
    expected_epoch_sequence = list(range(expected_epochs))
    for cell, epochs in grouped.items():
        if epochs != expected_epoch_sequence:
            raise PilotAuditError(f"Pilot history has an incomplete or reordered epoch sequence for {cell}")
    return rows


def verify_checkpoints(
    run_dir: Path, cells: list[tuple[str, float]], expected_epochs: int,
) -> None:
    checkpoint_dir = run_dir / "checkpoints"
    observed = set(checkpoint_dir.glob("*.pt")) if checkpoint_dir.is_dir() else set()
    expected_paths = {
        checkpoint_dir / f"{operator}_density_{density:.2f}.pt" for operator, density in cells
    }
    if observed != expected_paths:
        raise PilotAuditError("Pilot checkpoint set differs from the frozen operator/density grid")
    for path in expected_paths:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            contract = payload["immutable_contract"]
            history = payload["history"]
            completed_epoch = int(payload["completed_epoch"])
        except (KeyError, TypeError, ValueError, OSError, RuntimeError) as error:
            raise PilotAuditError(f"Unreadable pilot checkpoint: {path}") from error
        expected_operator, density_text = path.stem.rsplit("_density_", maxsplit=1)
        if contract.get("operator") != expected_operator or float(contract.get("density")) != float(density_text):
            raise PilotAuditError(f"Checkpoint contract does not match its filename: {path.name}")
        if completed_epoch != expected_epochs - 1 or not isinstance(history, list) or len(history) != expected_epochs:
            raise PilotAuditError(f"Checkpoint is not complete for its planned epochs: {path.name}")
        if [row.get("epoch") for row in history] != list(range(expected_epochs)):
            raise PilotAuditError(f"Checkpoint epoch history is invalid: {path.name}")


def audit_pilot_run(run_dir: Path, *, require_recovery_test: bool = False) -> dict[str, Any]:
    metadata = load_json(run_dir / "metadata.json")
    status = load_json(run_dir / "status.json")
    summary = load_json(run_dir / "summary.json")
    if metadata.get("run_kind") != "engineering_pilot_score_blind":
        raise PilotAuditError("Run is not labelled as an engineering score-blind pilot")
    if metadata.get("status") != "complete" or status.get("state") != "complete":
        raise PilotAuditError("Pilot has not completed cleanly")
    if metadata.get("outer_test_evaluation") is not False or status.get("outer_test_evaluation") is not False:
        raise PilotAuditError("Pilot must explicitly disable outer-test evaluation")
    if summary.get("outer_test_evaluation") is not False:
        raise PilotAuditError("Summary does not preserve the outer-test embargo")
    expected_epoch_count = metadata.get("max_epochs")
    if not isinstance(expected_epoch_count, int) or expected_epoch_count <= 0:
        raise PilotAuditError("Metadata has an invalid planned epoch count")
    cells = expected_cells(metadata)
    expected_labels = [f"{operator}_density_{density:.2f}" for operator, density in cells]
    if summary.get("cells_completed") != expected_labels or summary.get("cell_count") != len(cells):
        raise PilotAuditError("Summary does not cover exactly the planned pilot cells")
    if summary.get("loss_history_rows") != len(cells) * expected_epoch_count:
        raise PilotAuditError("Summary loss-history count is inconsistent")
    for name in ("pilot_loss_history.csv", "summary.json"):
        expected_hash = metadata.get("artifact_hashes", {}).get(name)
        if not isinstance(expected_hash, str) or sha256_file(run_dir / name) != expected_hash:
            raise PilotAuditError(f"Artifact hash mismatch: {name}")
    top_level_names = {path.name.lower() for path in run_dir.iterdir() if path.is_file()}
    banned = sorted(
        name for name in top_level_names
        if any(token in name for token in PROHIBITED_ARTIFACT_TOKENS)
    )
    if banned:
        raise PilotAuditError(f"Pilot contains forbidden predictive artifact names: {banned}")
    history = read_history(run_dir / "pilot_loss_history.csv", expected_epoch_count, cells)
    verify_checkpoints(run_dir, cells, expected_epoch_count)
    recovery = metadata.get("recovery_test", {})
    if require_recovery_test and not (
        recovery.get("intentional_interruption_requested") is True and recovery.get("resumed") is True
    ):
        raise PilotAuditError("Required managed recovery test is not recorded as resumed")
    return {
        "state": "passed",
        "run_id": metadata.get("run_id"),
        "run_kind": metadata.get("run_kind"),
        "outer_test_evaluation": False,
        "operator_density_cells": len(cells),
        "epochs_per_cell": expected_epoch_count,
        "loss_history_rows_checked": len(history),
        "checkpoints_checked": len(cells),
        "recovery_test_required": require_recovery_test,
        "recovery_test_resumed": recovery.get("resumed") is True,
        "notice": "Integrity-only certificate; no held-out-site prediction or metric was calculated or reported.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-recovery-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_pilot_run(args.run_dir, require_recovery_test=args.require_recovery_test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
