from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from scripts.audit_neural_pilot import PilotAuditError, audit_pilot_run


FIELDS = [
    "operator", "density", "epoch", "training_bce_loss", "validation_bce_loss",
    "peak_gpu_memory_bytes",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_valid_pilot(root: Path) -> Path:
    run = root / "pilot"
    checkpoint_dir = run / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    metadata = {
        "run_id": "synthetic", "run_kind": "engineering_pilot_score_blind", "status": "complete",
        "outer_test_evaluation": False, "operators": ["identity"], "densities": [0.0],
        "max_epochs": 2,
        "recovery_test": {"intentional_interruption_requested": True, "resumed": True},
    }
    status = {"state": "complete", "outer_test_evaluation": False}
    (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run / "status.json").write_text(json.dumps(status), encoding="utf-8")
    rows = [
        {"operator": "identity", "density": 0.0, "epoch": epoch, "training_bce_loss": 1.0, "validation_bce_loss": 1.0, "peak_gpu_memory_bytes": 0}
        for epoch in range(2)
    ]
    with (run / "pilot_loss_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    torch.save(
        {"immutable_contract": {"operator": "identity", "density": 0.0}, "completed_epoch": 1, "history": rows},
        checkpoint_dir / "identity_density_0.00.pt",
    )
    summary = {"cells_completed": ["identity_density_0.00"], "cell_count": 1, "loss_history_rows": 2, "outer_test_evaluation": False}
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    metadata["artifact_hashes"] = {
        "pilot_loss_history.csv": digest(run / "pilot_loss_history.csv"),
        "summary.json": digest(run / "summary.json"),
    }
    (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return run


class NeuralPilotAuditTests(unittest.TestCase):
    def test_valid_synthetic_pilot_passes_without_predictive_values(self) -> None:
        with TemporaryDirectory() as temporary:
            report = audit_pilot_run(build_valid_pilot(Path(temporary)), require_recovery_test=True)
        self.assertEqual(report["state"], "passed")
        self.assertFalse(report["outer_test_evaluation"])

    def test_history_mutation_fails_hash_audit(self) -> None:
        with TemporaryDirectory() as temporary:
            run = build_valid_pilot(Path(temporary))
            with (run / "pilot_loss_history.csv").open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaises(PilotAuditError):
                audit_pilot_run(run, require_recovery_test=True)


if __name__ == "__main__":
    unittest.main()
