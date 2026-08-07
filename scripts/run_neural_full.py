"""Run the frozen nested LOSO neural operator evaluation with sealed artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.run_baselines import (
    publish_sns_notification,
    read_csv,
    sha256_file,
    validate_split_contract,
    verify_frozen_hashes,
    write_csv,
    write_json_atomic,
)
from src.neural_data import load_neural_cohort
from src.neural_full_training import (
    FullTrainingError,
    build_fit_tensors,
    diagnostic_rows,
    instantiate_best_model,
    selected_candidate_index,
    selected_final_epoch,
    site_classification_rows,
    train_early_stopped,
    train_fixed_epochs,
)
from src.neural_models import NeuralArchitecture
from src.neural_training import utc_now


PREDICTION_FIELDS = [
    "operator", "density", "seed", "outer_fold", "held_out_site", "subject_id",
    "site_id", "label_asd", "probability_asd", "predicted_asd",
]
METRIC_FIELDS = [
    "operator", "density", "seed", "outer_fold", "held_out_site", "participants",
    "asd", "control", "balanced_accuracy", "auroc", "sensitivity", "specificity",
    "learning_rate", "weight_decay", "final_epochs", "parameter_count",
]
TUNING_FIELDS = [
    "operator", "density", "outer_fold", "held_out_site", "candidate_index",
    "learning_rate", "weight_decay", "inner_mean_site_balanced_accuracy",
    "inner_site_score_rows", "selected", "selected_final_epoch",
]
INNER_SITE_FIELDS = [
    "operator", "density", "outer_fold", "held_out_site", "candidate_index",
    "learning_rate", "weight_decay", "inner_validation_fold", "tuning_seed",
    "site_id", "participants", "asd", "control", "balanced_accuracy", "auroc",
    "sensitivity", "specificity", "best_epoch",
]
CURVE_FIELDS = [
    "fit_scope", "operator", "density", "outer_fold", "held_out_site",
    "candidate_index", "inner_validation_fold", "seed", "epoch",
    "training_bce_loss", "validation_mean_site_bce", "maximum_gradient_norm", "improved",
]
DIAGNOSTIC_FIELDS = [
    "operator", "density", "seed", "outer_fold", "held_out_site", "subject_id", "site_id",
    "layer", "normalized_effective_rank", "normalized_dispersion", "mean_pairwise_cosine",
    "invariant_edge_transport_distance",
]
RUNTIME_FIELDS = [
    "fit_scope", "operator", "density", "outer_fold", "held_out_site", "candidate_index",
    "inner_validation_fold", "seed", "epochs_completed", "runtime_seconds",
    "peak_gpu_memory_bytes", "resumed",
]
WARNING_FIELDS = [
    "operator", "density", "outer_fold", "held_out_site", "fit_scope", "failure_type", "failure_message",
]
SITE_ARTIFACT_FIELDS = {
    "predictions.csv": PREDICTION_FIELDS,
    "test_metrics.csv": METRIC_FIELDS,
    "tuning_scores.csv": TUNING_FIELDS,
    "inner_site_scores.csv": INNER_SITE_FIELDS,
    "training_curves.csv": CURVE_FIELDS,
    "diagnostics.csv": DIAGNOSTIC_FIELDS,
    "fit_runtime.csv": RUNTIME_FIELDS,
    "fit_warnings.csv": WARNING_FIELDS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/neural_full_protocol.json")
    parser.add_argument("--operator-contract", default="configs/neural_operator_contract_v2.json")
    parser.add_argument("--analysis-protocol", default="configs/neural_analysis_protocol.json")
    parser.add_argument("--baseline-inputs", default="configs/baseline_inputs_and_splits.json")
    parser.add_argument("--table", default="data/processed/abide_i_baseline_table.csv")
    parser.add_argument("--connectomes", default="data/processed/abide_i_connectomes_fisher_z.npz")
    parser.add_argument("--outer-splits", default="data/processed/splits/outer_loso_assignments.csv")
    parser.add_argument("--inner-splits", default="data/processed/splits/inner_grouped_assignments.csv")
    parser.add_argument("--output-root", default="outputs/runs/neural-full")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-kind", choices=("smoke", "full"), default="full")
    parser.add_argument("--held-out-sites", nargs="+", default=None)
    parser.add_argument("--fast-smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch-checkpoints", type=int, default=None)
    parser.add_argument("--notification-topic-arn", default=os.environ.get("BUNN_SNS_TOPIC_ARN"))
    parser.add_argument("--require-notification", action="store_true")
    parser.add_argument("--code-version", default="unknown")
    return parser.parse_args()


def configurations(contract: dict[str, Any]) -> list[tuple[str, float]]:
    output: list[tuple[str, float]] = []
    for operator, specification in contract["operators"].items():
        output.extend((operator, float(density)) for density in specification["densities"])
    if len(output) != 14 or len(set(output)) != 14:
        raise FullTrainingError("Full operator contract must define 14 unique configurations")
    return output


def label_for_site(outer_fold: int, held_out_site: str) -> str:
    return f"{outer_fold:02d}_{held_out_site}"


def label_for_configuration(operator: str, density: float) -> str:
    return f"{operator}_density_{density:.2f}"


def update_status(run_dir: Path, **changes: Any) -> None:
    path = run_dir / "status.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload.update(changes)
    payload["last_updated_utc"] = utc_now()
    write_json_atomic(path, payload)


def immutable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "run_id", "run_kind", "code_version", "source_hashes", "frozen_input_hashes",
        "configurations", "held_out_sites", "site_to_outer_fold", "protocol",
        "operator_contract", "analysis_protocol", "smoke_override",
    )
    return {field: metadata[field] for field in fields}


def verified_site(path: Path, outer_fold: int, held_out_site: str) -> bool:
    completion_path = path / "complete.json"
    if not completion_path.is_file():
        return False
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if completion.get("outer_fold") != outer_fold or completion.get("held_out_site") != held_out_site:
        return False
    hashes = completion.get("artifact_hashes", {})
    return all((path / name).is_file() and hashes.get(name) == sha256_file(path / name) for name in SITE_ARTIFACT_FIELDS)


def initialise_run(run_dir: Path, metadata: dict[str, Any], resume: bool) -> list[str]:
    metadata_path = run_dir / "metadata.json"
    if resume:
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Cannot resume without {metadata_path}")
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if immutable_metadata(existing) != immutable_metadata(metadata):
            raise FullTrainingError("Full-run immutable metadata mismatch")
        existing.update({"status": "running", "resumed_utc": utc_now()})
        for stale_field in ("failed_utc", "failure_type", "failure_message"):
            existing.pop(stale_field, None)
        write_json_atomic(metadata_path, existing)
        completed = []
        for site in metadata["held_out_sites"]:
            fold = int(next(key for key, value in metadata["site_to_outer_fold"].items() if value == site))
            if verified_site(run_dir / "folds" / label_for_site(fold, site), fold, site):
                completed.append(site)
        update_status(
            run_dir, state="running", pid=os.getpid(), resumed_utc=utc_now(),
            completed_sites=completed, completed_site_count=len(completed),
            failure_type=None, failure_message=None,
        )
        return completed
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    write_json_atomic(metadata_path, metadata)
    write_json_atomic(
        run_dir / "status.json",
        {
            "state": "running", "run_id": metadata["run_id"], "pid": os.getpid(),
            "started_utc": metadata["started_utc"], "last_updated_utc": utc_now(),
            "completed_sites": [], "completed_site_count": 0,
            "total_sites": len(metadata["held_out_sites"]), "current_site": None,
            "current_configuration": None, "current_stage": "initialised",
            "held_out_results_embargoed": True,
        },
    )
    return []


def save_site(
    run_dir: Path,
    outer_fold: int,
    held_out_site: str,
    rows: dict[str, list[dict[str, Any]]],
) -> Path:
    folds = run_dir / "folds"
    folds.mkdir(parents=True, exist_ok=True)
    label = label_for_site(outer_fold, held_out_site)
    final = folds / label
    if final.exists():
        raise FileExistsError(f"Site artifact already exists: {final}")
    temporary = folds / f".{label}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True)
    for name, fields in SITE_ARTIFACT_FIELDS.items():
        write_csv(temporary / name, rows[name], fields)
    write_json_atomic(
        temporary / "complete.json",
        {
            "state": "complete", "outer_fold": outer_fold, "held_out_site": held_out_site,
            "completed_utc": utc_now(),
            "row_counts": {name: len(rows[name]) for name in SITE_ARTIFACT_FIELDS},
            "artifact_hashes": {name: sha256_file(temporary / name) for name in SITE_ARTIFACT_FIELDS},
        },
    )
    os.replace(temporary, final)
    return final


def classification_row_prefix(operator: str, density: float, outer_fold: int, site: str) -> dict[str, Any]:
    return {"operator": operator, "density": density, "outer_fold": outer_fold, "held_out_site": site}


def fit_contract(base: dict[str, Any], **changes: Any) -> dict[str, Any]:
    payload = dict(base)
    payload.update(changes)
    return payload


def run_site(
    *,
    run_dir: Path,
    site: str,
    outer_fold: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    inner_validation: dict[int, np.ndarray],
    cohort: Any,
    table: pd.DataFrame,
    protocol: dict[str, Any],
    operator_contract: dict[str, Any],
    configuration_grid: list[tuple[str, float]],
    smoke_override: dict[str, Any] | None,
    resume: bool,
    device: torch.device,
    base_contract: dict[str, Any],
    checkpoint_counter: list[int],
    stop_after_checkpoints: int | None,
) -> dict[str, list[dict[str, Any]]]:
    rows = {name: [] for name in SITE_ARTIFACT_FIELDS}
    architecture = NeuralArchitecture(
        input_dim=operator_contract["shared_backbone"]["input_dimension"],
        hidden_dim=operator_contract["shared_backbone"]["hidden_dimension"],
        layers=operator_contract["shared_backbone"]["layers"],
        bundles=operator_contract["shared_backbone"]["bundles"],
        channels=operator_contract["shared_backbone"]["vector_field_channels"],
        dropout=operator_contract["shared_backbone"]["dropout"],
        diffusion_time=operator_contract["operators"]["learned_bunn"]["diffusion_time"],
    )
    training = protocol["training"]
    candidates = list(protocol["tuning"]["candidates"])
    tuning_seeds = list(protocol["tuning"]["seeds"])
    final_seeds = list(protocol["final_fit"]["seeds"])
    inner_folds = sorted(inner_validation)
    if smoke_override:
        candidates = candidates[: smoke_override["candidate_count"]]
        tuning_seeds = tuning_seeds[: smoke_override["tuning_seed_count"]]
        final_seeds = final_seeds[: smoke_override["final_seed_count"]]
        inner_folds = inner_folds[: smoke_override["inner_fold_count"]]
        training = {**training, **smoke_override["training"]}

    def progress(stage: str, operator: str, density: float, **extra: Any):
        def callback(epoch_state: dict[str, Any]) -> None:
            checkpoint_counter[0] += 1
            update_status(
                run_dir, state="running", current_site=site,
                current_configuration=label_for_configuration(operator, density),
                current_stage=stage, current_epoch=epoch_state["epoch"], **extra,
            )
            if stop_after_checkpoints is not None and checkpoint_counter[0] >= stop_after_checkpoints:
                raise InterruptedError("intentional full-run recovery test after durable epoch checkpoint")
        return callback

    for operator, density in configuration_grid:
        config_label = label_for_configuration(operator, density)
        update_status(run_dir, current_site=site, current_configuration=config_label, current_stage="inner_tuning", current_epoch=None)
        fold_tensors: dict[int, Any] = {}
        for fold in inner_folds:
            validation_indices = np.asarray(inner_validation[fold], dtype=int)
            fitting_indices = np.setdiff1d(train_indices, validation_indices)
            fold_tensors[fold] = build_fit_tensors(cohort, fitting_indices, validation_indices, density, device)

        candidate_rows: list[dict[str, Any]] = []
        candidate_best_epochs: dict[int, list[int]] = {}
        for candidate_index, candidate in enumerate(candidates):
            candidate_site_scores: list[float] = []
            candidate_best_epochs[candidate_index] = []
            starting_inner_rows = len(rows["inner_site_scores.csv"])
            for fold in inner_folds:
                tensors = fold_tensors[fold]
                for tuning_seed in tuning_seeds:
                    checkpoint = run_dir / "work" / label_for_site(outer_fold, site) / config_label / "tuning" / f"candidate_{candidate_index}" / f"fold_{fold}_seed_{tuning_seed}.pt"
                    contract = fit_contract(
                        base_contract, fit_scope="inner_tuning", operator=operator, density=density,
                        outer_fold=outer_fold, held_out_site=site, candidate_index=candidate_index,
                        inner_validation_fold=fold, seed=tuning_seed, candidate=candidate,
                    )
                    checkpoint_preexisting = checkpoint.exists()
                    started = time.perf_counter()
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    payload = train_early_stopped(
                        checkpoint_path=checkpoint, immutable_contract=contract, operator=operator,
                        architecture=architecture, tensors=tensors,
                        learning_rate=float(candidate["learning_rate"]), weight_decay=float(candidate["weight_decay"]),
                        batch_size=int(training["batch_size"]), maximum_epochs=int(training["maximum_epochs"]),
                        minimum_epochs=int(training["minimum_epochs"]), patience=int(training["early_stopping_patience"]),
                        minimum_delta=float(training["early_stopping_minimum_delta"]),
                        gradient_clip_norm=float(training["gradient_clip_global_norm"]), seed=int(tuning_seed),
                        resume=resume, device=device,
                        progress=progress("inner_tuning_checkpointed", operator, density, candidate_index=candidate_index, inner_validation_fold=fold, seed=tuning_seed),
                    )
                    runtime = time.perf_counter() - started
                    best_model = instantiate_best_model(operator, architecture, payload["best_model_state"], device)
                    _, site_rows = site_classification_rows(
                        best_model, tensors, int(training["batch_size"]), float(protocol["evaluation"]["decision_threshold"])
                    )
                    for metric in site_rows:
                        candidate_site_scores.append(float(metric["balanced_accuracy"]))
                        rows["inner_site_scores.csv"].append(
                            {
                                **classification_row_prefix(operator, density, outer_fold, site),
                                "candidate_index": candidate_index,
                                "learning_rate": candidate["learning_rate"], "weight_decay": candidate["weight_decay"],
                                "inner_validation_fold": fold, "tuning_seed": tuning_seed,
                                **metric, "best_epoch": int(payload["best_epoch"]) + 1,
                            }
                        )
                    candidate_best_epochs[candidate_index].append(int(payload["best_epoch"]) + 1)
                    for history in payload["history"]:
                        rows["training_curves.csv"].append(
                            {
                                **classification_row_prefix(operator, density, outer_fold, site),
                                "fit_scope": "inner_tuning", "candidate_index": candidate_index,
                                "inner_validation_fold": fold, "seed": tuning_seed,
                                "epoch": history["epoch"], "training_bce_loss": history["training_bce_loss"],
                                "validation_mean_site_bce": history["validation_mean_site_bce"],
                                "maximum_gradient_norm": history["maximum_gradient_norm"], "improved": int(history["improved"]),
                            }
                        )
                    rows["fit_runtime.csv"].append(
                        {
                            **classification_row_prefix(operator, density, outer_fold, site),
                            "fit_scope": "inner_tuning", "candidate_index": candidate_index,
                            "inner_validation_fold": fold, "seed": tuning_seed,
                            "epochs_completed": len(payload["history"]), "runtime_seconds": runtime,
                            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
                            "resumed": int(resume and checkpoint_preexisting),
                        }
                    )
            if not candidate_site_scores:
                raise FullTrainingError("Candidate produced no inner-site scores")
            candidate_rows.append(
                {
                    **classification_row_prefix(operator, density, outer_fold, site),
                    "candidate_index": candidate_index, "learning_rate": candidate["learning_rate"],
                    "weight_decay": candidate["weight_decay"],
                    "inner_mean_site_balanced_accuracy": float(np.mean(candidate_site_scores)),
                    "inner_site_score_rows": len(rows["inner_site_scores.csv"]) - starting_inner_rows,
                    "selected": 0, "selected_final_epoch": "",
                }
            )

        chosen_index = selected_candidate_index(candidate_rows)
        selected_epochs = selected_final_epoch(candidate_best_epochs[chosen_index])
        chosen = candidates[chosen_index]
        for candidate_row in candidate_rows:
            if int(candidate_row["candidate_index"]) == chosen_index:
                candidate_row["selected"] = 1
                candidate_row["selected_final_epoch"] = selected_epochs
            rows["tuning_scores.csv"].append(candidate_row)

        del fold_tensors
        if device.type == "cuda":
            torch.cuda.empty_cache()
        final_tensors = build_fit_tensors(cohort, train_indices, test_indices, density, device)
        for final_seed in final_seeds:
            checkpoint = run_dir / "work" / label_for_site(outer_fold, site) / config_label / "final" / f"seed_{final_seed}.pt"
            contract = fit_contract(
                base_contract, fit_scope="outer_final", operator=operator, density=density,
                outer_fold=outer_fold, held_out_site=site, seed=final_seed,
                candidate_index=chosen_index, candidate=chosen, final_epochs=selected_epochs,
            )
            checkpoint_preexisting = checkpoint.exists()
            started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            payload = train_fixed_epochs(
                checkpoint_path=checkpoint, immutable_contract=contract, operator=operator,
                architecture=architecture, tensors=final_tensors,
                learning_rate=float(chosen["learning_rate"]), weight_decay=float(chosen["weight_decay"]),
                batch_size=int(training["batch_size"]), epochs=selected_epochs,
                gradient_clip_norm=float(training["gradient_clip_global_norm"]), seed=int(final_seed),
                resume=resume, device=device,
                progress=progress("outer_final_checkpointed", operator, density, seed=final_seed),
            )
            runtime = time.perf_counter() - started
            final_model = instantiate_best_model(operator, architecture, payload["model_state"], device)
            probabilities, test_site_rows = site_classification_rows(
                final_model, final_tensors, int(training["batch_size"]), float(protocol["evaluation"]["decision_threshold"])
            )
            if len(test_site_rows) != 1 or test_site_rows[0]["site_id"] != site:
                raise FullTrainingError("Outer evaluation did not contain exactly the held-out site")
            metric = test_site_rows[0]
            rows["test_metrics.csv"].append(
                {
                    **classification_row_prefix(operator, density, outer_fold, site), "seed": final_seed,
                    **{key: metric[key] for key in ("participants", "asd", "control", "balanced_accuracy", "auroc", "sensitivity", "specificity")},
                    "learning_rate": chosen["learning_rate"], "weight_decay": chosen["weight_decay"],
                    "final_epochs": selected_epochs, "parameter_count": final_model.parameter_count(),
                }
            )
            labels = final_tensors.evaluation_labels.cpu().numpy().astype(int)
            for subject_id, site_id, label, probability in zip(
                final_tensors.evaluation_subject_ids, final_tensors.evaluation_sites, labels, probabilities, strict=True
            ):
                rows["predictions.csv"].append(
                    {
                        **classification_row_prefix(operator, density, outer_fold, site), "seed": final_seed,
                        "subject_id": subject_id, "site_id": site_id, "label_asd": int(label),
                        "probability_asd": float(probability),
                        "predicted_asd": int(probability >= protocol["evaluation"]["decision_threshold"]),
                    }
                )
            for diagnostic in diagnostic_rows(final_model, final_tensors, int(training["batch_size"])):
                subject_position = int(np.flatnonzero(final_tensors.evaluation_subject_ids == diagnostic["subject_id"])[0])
                rows["diagnostics.csv"].append(
                    {
                        **classification_row_prefix(operator, density, outer_fold, site), "seed": final_seed,
                        "site_id": final_tensors.evaluation_sites[subject_position], **diagnostic,
                    }
                )
            for history in payload["history"]:
                rows["training_curves.csv"].append(
                    {
                        **classification_row_prefix(operator, density, outer_fold, site),
                        "fit_scope": "outer_final", "candidate_index": chosen_index,
                        "inner_validation_fold": "", "seed": final_seed, "epoch": history["epoch"],
                        "training_bce_loss": history["training_bce_loss"], "validation_mean_site_bce": "",
                        "maximum_gradient_norm": history["maximum_gradient_norm"], "improved": "",
                    }
                )
            rows["fit_runtime.csv"].append(
                {
                    **classification_row_prefix(operator, density, outer_fold, site),
                    "fit_scope": "outer_final", "candidate_index": chosen_index,
                    "inner_validation_fold": "", "seed": final_seed,
                    "epochs_completed": len(payload["history"]), "runtime_seconds": runtime,
                    "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
                    "resumed": int(resume and checkpoint_preexisting),
                }
            )
        del final_tensors
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def notify(args: argparse.Namespace, run_dir: Path, state: str, message: str) -> dict[str, Any]:
    return publish_sns_notification(
        run_dir, args.notification_topic_arn,
        f"BuNN neural {args.run_kind} {state.upper()}: {args.run_id}",
        f"Run {args.run_id} {message}. Check integrity artifacts before reading results.",
    )


def run(args: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Full neural runner requires CUDA")
    if args.run_kind == "smoke" and not args.held_out_sites:
        raise ValueError("Smoke run requires explicit held-out sites")
    if args.fast_smoke and args.run_kind != "smoke":
        raise ValueError("--fast-smoke is allowed only for smoke runs")
    if args.run_kind == "full" and args.held_out_sites:
        raise ValueError("Full run always uses every frozen held-out site")
    if args.resume and not args.run_id:
        raise ValueError("Resume requires a run ID")
    if args.require_notification and not args.notification_topic_arn:
        raise ValueError("Required notification topic is missing")
    if args.stop_after_epoch_checkpoints is not None and args.stop_after_epoch_checkpoints <= 0:
        raise ValueError("Checkpoint interruption count must be positive")

    paths = {
        "protocol": Path(args.protocol), "operator_contract": Path(args.operator_contract),
        "analysis_protocol": Path(args.analysis_protocol), "baseline_inputs": Path(args.baseline_inputs),
        "table": Path(args.table), "connectomes": Path(args.connectomes),
        "outer_splits": Path(args.outer_splits), "inner_splits": Path(args.inner_splits),
    }
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    operator_contract = json.loads(paths["operator_contract"].read_text(encoding="utf-8"))
    analysis_protocol = json.loads(paths["analysis_protocol"].read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != 1 or operator_contract.get("contract_version") != 2 or analysis_protocol.get("protocol_version") != 1:
        raise FullTrainingError("Unsupported frozen neural protocol version")
    configuration_grid = configurations(operator_contract)
    frozen_hashes = verify_frozen_hashes(paths["baseline_inputs"], paths["table"], paths["outer_splits"], paths["inner_splits"])
    table = pd.read_csv(paths["table"], dtype={"subject_id": str, "site_id": str})
    cohort = load_neural_cohort(paths["connectomes"], paths["table"])
    outer_rows = read_csv(paths["outer_splits"])
    inner_rows = read_csv(paths["inner_splits"])
    available_sites = sorted({row["held_out_site"] for row in outer_rows})
    selected_sites = args.held_out_sites or available_sites
    if sorted(set(selected_sites)) != sorted(selected_sites):
        raise ValueError("Held-out sites are repeated")
    if set(selected_sites) - set(available_sites):
        raise ValueError("Unknown held-out site requested")
    smoke_override = None
    if args.fast_smoke:
        smoke_override = {
            "candidate_count": 1, "tuning_seed_count": 1, "final_seed_count": 1, "inner_fold_count": 1,
            "training": {"maximum_epochs": 2, "minimum_epochs": 1, "early_stopping_patience": 2},
            "notice": "Engineering-only reduced workload; never a scientific result.",
        }
    site_to_outer_fold = {
        str(int(next(row["outer_fold"] for row in outer_rows if row["held_out_site"] == site))): site
        for site in selected_sites
    }
    metadata = {
        "run_id": args.run_id, "run_kind": args.run_kind, "status": "running", "started_utc": utc_now(),
        "code_version": args.code_version,
        "source_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "frozen_input_hashes": frozen_hashes,
        "configurations": [{"operator": operator, "density": density} for operator, density in configuration_grid],
        "held_out_sites": selected_sites, "site_to_outer_fold": site_to_outer_fold,
        "protocol": protocol, "operator_contract": operator_contract, "analysis_protocol": analysis_protocol,
        "smoke_override": smoke_override,
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda},
        "results_embargoed": True,
    }
    run_dir = Path(args.output_root) / args.run_id
    completed_sites = initialise_run(run_dir, metadata, args.resume)
    if args.require_notification:
        alert = notify(args, run_dir, "started", f"started with {len(selected_sites)} held-out sites")
        if alert["status"] != "published":
            raise RuntimeError("Required STARTED notification failed")

    torch.use_deterministic_algorithms(bool(protocol["training"]["deterministic_algorithms"]))
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    base_contract = {
        "run_id": args.run_id, "code_version": args.code_version,
        "protocol_sha256": sha256_file(paths["protocol"]),
        "operator_contract_sha256": sha256_file(paths["operator_contract"]),
        "data_sha256": sha256_file(paths["connectomes"]),
        "outer_splits_sha256": sha256_file(paths["outer_splits"]),
        "inner_splits_sha256": sha256_file(paths["inner_splits"]),
        "smoke_override": smoke_override,
    }
    checkpoint_counter = [0]
    for site in selected_sites:
        outer_fold, train_indices, test_indices, inner_validation = validate_split_contract(table, outer_rows, inner_rows, site)
        if site in completed_sites:
            continue
        update_status(run_dir, current_site=site, current_outer_fold=outer_fold, current_stage="starting_site", current_configuration=None)
        site_rows = run_site(
            run_dir=run_dir, site=site, outer_fold=outer_fold,
            train_indices=train_indices, test_indices=test_indices, inner_validation=inner_validation,
            cohort=cohort, table=table, protocol=protocol, operator_contract=operator_contract,
            configuration_grid=configuration_grid, smoke_override=smoke_override, resume=args.resume,
            device=device, base_contract=base_contract, checkpoint_counter=checkpoint_counter,
            stop_after_checkpoints=args.stop_after_epoch_checkpoints,
        )
        save_site(run_dir, outer_fold, site, site_rows)
        completed_sites.append(site)
        update_status(
            run_dir, completed_sites=completed_sites, completed_site_count=len(completed_sites),
            current_site=None, current_outer_fold=None, current_configuration=None, current_stage="site_sealed",
        )

    aggregate_rows = {name: [] for name in SITE_ARTIFACT_FIELDS}
    for site in selected_sites:
        outer_fold = int(next(key for key, value in site_to_outer_fold.items() if value == site))
        path = run_dir / "folds" / label_for_site(outer_fold, site)
        if not verified_site(path, outer_fold, site):
            raise FullTrainingError(f"Completed site failed verification: {site}")
        for name, fields in SITE_ARTIFACT_FIELDS.items():
            with (path / name).open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != fields:
                    raise FullTrainingError(f"Site schema mismatch: {site}/{name}")
                aggregate_rows[name].extend(reader)
    for name, fields in SITE_ARTIFACT_FIELDS.items():
        write_csv(run_dir / name, aggregate_rows[name], fields)
    summary = {
        "run_id": args.run_id, "run_kind": args.run_kind, "status": "complete",
        "held_out_sites": selected_sites, "configuration_count": len(configuration_grid),
        "row_counts": {name: len(rows) for name, rows in aggregate_rows.items()},
        "results_embargoed": True,
        "notice": "Counts and integrity state only. Predictive values require a separate successful score-blind audit.",
    }
    write_json_atomic(run_dir / "summary.json", summary)
    metadata_path = run_dir / "metadata.json"
    final_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    final_metadata.update(
        {
            "status": "complete", "completed_utc": utc_now(),
            "artifact_hashes": {name: sha256_file(run_dir / name) for name in [*SITE_ARTIFACT_FIELDS, "summary.json"]},
            "notification_status": "pending",
        }
    )
    write_json_atomic(metadata_path, final_metadata)
    update_status(
        run_dir, state="complete", current_stage="completion_sealed_results_embargoed",
        current_site=None, current_outer_fold=None, current_configuration=None,
        current_epoch=None, candidate_index=None, inner_fold=None, seed=None,
        failure_type=None, failure_message=None, completed_sites=completed_sites,
        completed_site_count=len(completed_sites), held_out_results_embargoed=True,
    )
    alert = notify(args, run_dir, "complete", "completed and remains results-embargoed")
    final_metadata["notification_status"] = alert["status"]
    write_json_atomic(metadata_path, final_metadata)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return run_dir


def mark_failed(args: argparse.Namespace, error: Exception) -> None:
    run_dir = Path(args.output_root) / args.run_id
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.is_file():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    state = "interrupted" if isinstance(error, InterruptedError) else "failed"
    metadata.update({"status": state, "failed_utc": utc_now(), "failure_type": type(error).__name__, "failure_message": str(error)})
    write_json_atomic(metadata_path, metadata)
    update_status(run_dir, state=state, current_stage="terminal_error", failure_type=type(error).__name__, failure_message=str(error), held_out_results_embargoed=True)
    alert = notify(args, run_dir, state, f"ended with {type(error).__name__}")
    metadata["notification_status"] = alert["status"]
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
