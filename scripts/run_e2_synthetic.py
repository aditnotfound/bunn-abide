"""Run a score-blind E2 timing smoke or the frozen full synthetic study."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_baselines import publish_sns_notification
from src.extensions.e2_synthetic import generate_e2_dataset
from src.extensions.e2_training import audit_e2_cell, run_e2_cell, sha256_file, write_json_atomic


OPERATORS = (
    "identity", "gcn", "trivial_bundle", "fixed_random_transport",
    "learned_local", "learned_bunn", "oracle_true_map",
)
FULL_CONDITIONS = (
    ("S0_no_geometry", 0.0),
    ("S1_recoverable_geometry", 0.0),
    ("S2_incorrect_topology", 0.0),
    ("S3_shuffled_geometry", 0.0),
    ("S4_transport_noise", 15.0),
    ("S4_transport_noise", 30.0),
    ("S4_transport_noise", 60.0),
    ("S4_transport_noise", 120.0),
    ("S5_unlearnable_subject_frames", 0.0),
    ("S6_global_feature_analogue", 0.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/extensions/e2_synthetic_geometry_v1.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/extensions")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--notification-topic-arn", default=os.environ.get("BUNN_SNS_TOPIC_ARN"))
    parser.add_argument("--require-notification", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def now() -> str:
    return datetime.now(UTC).isoformat()


def main() -> int:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_any_E2_test_prediction_or_metric":
        raise RuntimeError("E2 protocol is not frozen")
    if args.mode == "full" and not args.require_notification:
        raise RuntimeError("Full E2 requires --require-notification")
    run_dir = args.output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol_hash = sha256_file(args.protocol)
    device = resolve_device(args.device)
    metadata = {
        "run_id": args.run_id,
        "mode": args.mode,
        "started_utc": now(),
        "protocol_sha256": protocol_hash,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        prior = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("run_id", "mode", "protocol_sha256"):
            if prior.get(key) != metadata[key]:
                raise RuntimeError("Existing run metadata conflicts with requested run")
    else:
        write_json_atomic(metadata_path, metadata)
    if args.require_notification:
        alert = None if not args.notification_topic_arn else publish_sns_notification(
            run_dir, args.notification_topic_arn, f"BuNN E2 STARTED: {args.run_id}",
            "Frozen E2 synthetic-geometry run started. Test values remain embargoed until audit.",
        )
        if alert is None or alert.get("status") != "published":
            raise RuntimeError("Required E2 start notification failed")

    if args.mode == "smoke":
        conditions = (("S0_no_geometry", 0.0), ("S1_recoverable_geometry", 0.0))
        operators = ("gcn", "learned_local", "learned_bunn", "oracle_true_map")
        seeds = protocol["data"]["replicate_seeds"][:1]
        samples = 100
        maximum_epochs = 8
        minimum_epochs = 8
        patience = 8
        evaluate_test = False
    else:
        conditions = FULL_CONDITIONS
        operators = OPERATORS
        seeds = protocol["data"]["replicate_seeds"]
        samples = int(protocol["data"]["samples_per_replicate"])
        training = protocol["training"]
        maximum_epochs = int(training["maximum_epochs"])
        minimum_epochs = int(training["minimum_epochs"])
        patience = int(training["early_stopping_patience"])
        evaluate_test = True

    expected = len(conditions) * len(operators) * len(seeds)
    completed = 0
    audit_hashes: dict[str, str] = {}
    for family, noise in conditions:
        for data_seed in seeds:
            dataset = generate_e2_dataset(
                family=family, seed=int(data_seed), samples=samples,
                nodes=int(protocol["data"]["nodes"]), bundles=int(protocol["data"]["bundles"]),
                channels=int(protocol["data"]["channels"]),
                noise_standard_deviation=float(protocol["data"]["noise_standard_deviation"]),
                marker_strength=float(protocol["data"]["frame_marker_strength"]),
                transport_noise_degrees=noise,
            )
            condition_id = f"{family}__noise_{noise:g}__seed_{data_seed}"
            for operator in operators:
                cell_id = f"{condition_id}__{operator}"
                cell_dir = run_dir / "cells" / cell_id

                def heartbeat(value: dict[str, Any], *, active: str = cell_id) -> None:
                    write_json_atomic(run_dir / "status.json", {
                        "state": "running_score_blind", "updated_utc": now(),
                        "completed_cells": completed, "expected_cells": expected,
                        "active_cell": active, **value,
                    })

                result = run_e2_cell(
                    dataset=dataset, operator=operator, model_seed=int(data_seed) + 1000,
                    cell_dir=cell_dir, protocol_sha256=protocol_hash,
                    learning_rate=float(protocol["training"]["learning_rate"]),
                    weight_decay=float(protocol["training"]["weight_decay"]),
                    batch_size=int(protocol["training"]["batch_size"]),
                    maximum_epochs=maximum_epochs, minimum_epochs=minimum_epochs,
                    patience=patience, gradient_clip=float(protocol["training"]["gradient_clip_global_norm"]),
                    device=device, evaluate_test=evaluate_test, resume=args.resume,
                    epoch_callback=heartbeat,
                )
                audit = audit_e2_cell(cell_dir, require_predictions=evaluate_test)
                audit_path = cell_dir / "score_blind_audit.json"
                write_json_atomic(audit_path, audit)
                audit_hashes[cell_id] = sha256_file(audit_path)
                completed += 1
                heartbeat({"epoch": result.epochs_completed, "cell_complete": True})
    certificate = {
        "state": "complete_all_cells_score_blind_audited",
        "completed_utc": now(),
        "run_id": args.run_id,
        "mode": args.mode,
        "protocol_sha256": protocol_hash,
        "expected_cells": expected,
        "completed_cells": completed,
        "audit_hashes": audit_hashes,
    }
    write_json_atomic(run_dir / "manager_complete.json", certificate)
    write_json_atomic(run_dir / "status.json", {
        "state": certificate["state"], "updated_utc": now(),
        "completed_cells": completed, "expected_cells": expected, "active_cell": None,
    })
    if args.require_notification:
        alert = publish_sns_notification(
            run_dir, args.notification_topic_arn, f"BuNN E2 COMPLETE: {args.run_id}",
            f"All {completed} E2 cells completed and passed score-blind audit.",
        )
        if alert.get("status") != "published":
            raise RuntimeError("Required E2 completion notification failed")
    print(json.dumps({"state": certificate["state"], "cells": completed, "results_opened": False}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        try:
            failed_args = parse_args()
            failed_dir = failed_args.output_root / failed_args.run_id
            write_json_atomic(failed_dir / "status.json", {
                "state": "failed", "updated_utc": now(),
                "error_type": type(error).__name__, "error": str(error),
            })
            if failed_args.require_notification and failed_args.notification_topic_arn:
                publish_sns_notification(
                    failed_dir, failed_args.notification_topic_arn,
                    f"BuNN E2 FAILED: {failed_args.run_id}",
                    f"E2 stopped with {type(error).__name__}. Check the score-blind status and runner log before resuming.",
                )
        finally:
            raise
