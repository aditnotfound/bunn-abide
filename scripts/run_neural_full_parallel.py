"""Coordinate site-parallel execution of the frozen full neural evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scripts.run_baselines import (
    publish_sns_notification,
    read_csv,
    sha256_file,
    verify_frozen_hashes,
    write_csv,
    write_json_atomic,
)
from scripts.run_neural_full import (
    SITE_ARTIFACT_FIELDS,
    configurations,
    label_for_site,
    verified_site,
)
from src.neural_full_training import FullTrainingError
from src.neural_training import utc_now


class ParallelRunError(RuntimeError):
    """Raised when a worker, merge, or immutable execution contract fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/neural_full_protocol.json")
    parser.add_argument("--operator-contract", default="configs/neural_operator_contract_v2.json")
    parser.add_argument("--analysis-protocol", default="configs/neural_analysis_protocol.json")
    parser.add_argument("--parallel-contract", default="configs/neural_parallel_execution.json")
    parser.add_argument("--baseline-inputs", default="configs/baseline_inputs_and_splits.json")
    parser.add_argument("--table", default="data/processed/abide_i_baseline_table.csv")
    parser.add_argument("--connectomes", default="data/processed/abide_i_connectomes_fisher_z.npz")
    parser.add_argument("--outer-splits", default="data/processed/splits/outer_loso_assignments.csv")
    parser.add_argument("--inner-splits", default="data/processed/splits/inner_grouped_assignments.csv")
    parser.add_argument("--output-root", default="outputs/runs/neural-full-parallel")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-kind", choices=("smoke", "full"), default="full")
    parser.add_argument("--held-out-sites", nargs="+", default=None)
    parser.add_argument("--worker-count", type=int, default=3)
    parser.add_argument("--fast-smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-worker-index", type=int, default=None)
    parser.add_argument("--stop-after-epoch-checkpoints", type=int, default=None)
    parser.add_argument("--notification-topic-arn", default=os.environ.get("BUNN_SNS_TOPIC_ARN"))
    parser.add_argument("--require-notification", action="store_true")
    parser.add_argument("--code-version", default="unknown")
    return parser.parse_args()


def partition_sites(sites: list[str], worker_count: int) -> list[list[str]]:
    """Deterministically distribute ordered sites across non-empty workers."""
    if worker_count < 1:
        raise ValueError("Worker count must be positive")
    if len(sites) < worker_count:
        raise ValueError("Worker count cannot exceed the number of selected sites")
    assignments = [[] for _ in range(worker_count)]
    for index, site in enumerate(sites):
        assignments[index % worker_count].append(site)
    return assignments


def parallel_immutable(metadata: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "run_id", "run_kind", "code_version", "source_hashes", "frozen_input_hashes",
        "configurations", "held_out_sites", "site_to_outer_fold", "protocol",
        "operator_contract", "analysis_protocol", "smoke_override", "parallel_execution",
    )
    return {field: metadata[field] for field in fields}


def build_metadata(args: argparse.Namespace) -> tuple[dict[str, Any], list[list[str]]]:
    paths = {
        "protocol": Path(args.protocol), "operator_contract": Path(args.operator_contract),
        "analysis_protocol": Path(args.analysis_protocol), "parallel_contract": Path(args.parallel_contract),
        "baseline_inputs": Path(args.baseline_inputs),
        "table": Path(args.table), "connectomes": Path(args.connectomes),
        "outer_splits": Path(args.outer_splits), "inner_splits": Path(args.inner_splits),
    }
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    operator_contract = json.loads(paths["operator_contract"].read_text(encoding="utf-8"))
    analysis_protocol = json.loads(paths["analysis_protocol"].read_text(encoding="utf-8"))
    parallel_contract = json.loads(paths["parallel_contract"].read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != 1 or operator_contract.get("contract_version") != 2:
        raise FullTrainingError("Unsupported frozen neural protocol version")
    if analysis_protocol.get("protocol_version") != 1:
        raise FullTrainingError("Unsupported frozen analysis protocol version")
    if parallel_contract.get("contract_version") != 1:
        raise FullTrainingError("Unsupported parallel execution contract version")
    if args.worker_count != parallel_contract.get("worker_count"):
        raise ValueError("Worker count differs from the frozen parallel execution contract")
    configuration_grid = configurations(operator_contract)
    frozen_hashes = verify_frozen_hashes(
        paths["baseline_inputs"], paths["table"], paths["outer_splits"], paths["inner_splits"]
    )
    outer_rows = read_csv(paths["outer_splits"])
    available_sites = sorted({row["held_out_site"] for row in outer_rows})
    selected_sites = args.held_out_sites or available_sites
    if not selected_sites or len(selected_sites) != len(set(selected_sites)):
        raise ValueError("Held-out sites must be non-empty and unique")
    if set(selected_sites) - set(available_sites):
        raise ValueError("Unknown held-out site requested")
    if args.run_kind == "smoke" and not args.held_out_sites:
        raise ValueError("Parallel smoke requires explicit held-out sites")
    if args.run_kind == "full" and set(selected_sites) != set(available_sites):
        raise ValueError("Parallel full run must cover all frozen held-out sites")
    if args.fast_smoke and args.run_kind != "smoke":
        raise ValueError("Fast smoke is allowed only for smoke runs")
    assignments = partition_sites(selected_sites, args.worker_count)
    smoke_override = None
    if args.fast_smoke:
        smoke_override = {
            "candidate_count": 1, "tuning_seed_count": 1, "final_seed_count": 1,
            "inner_fold_count": 1,
            "training": {"maximum_epochs": 2, "minimum_epochs": 1, "early_stopping_patience": 2},
            "notice": "Engineering-only reduced workload; never a scientific result.",
        }
    site_to_outer_fold = {
        str(int(next(row["outer_fold"] for row in outer_rows if row["held_out_site"] == site))): site
        for site in selected_sites
    }
    metadata = {
        "run_id": args.run_id, "run_kind": args.run_kind, "status": "running",
        "started_utc": utc_now(), "code_version": args.code_version,
        "source_hashes": {
            name: sha256_file(path) for name, path in paths.items() if name != "parallel_contract"
        },
        "frozen_input_hashes": frozen_hashes,
        "configurations": [
            {"operator": operator, "density": density} for operator, density in configuration_grid
        ],
        "held_out_sites": selected_sites, "site_to_outer_fold": site_to_outer_fold,
        "protocol": protocol, "operator_contract": operator_contract,
        "analysis_protocol": analysis_protocol, "smoke_override": smoke_override,
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
        },
        "execution_mode": "site_parallel",
        "parallel_execution": {
            **parallel_contract,
            "contract_sha256": sha256_file(paths["parallel_contract"]),
            "assignments": {
                f"worker_{index:02d}": sites for index, sites in enumerate(assignments)
            },
        },
        "results_embargoed": True,
    }
    return metadata, assignments


def initialise_parent(run_dir: Path, metadata: dict[str, Any], resume: bool) -> None:
    metadata_path = run_dir / "metadata.json"
    if resume:
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Cannot resume without {metadata_path}")
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if parallel_immutable(existing) != parallel_immutable(metadata):
            raise ParallelRunError("Parallel-run immutable metadata mismatch")
        existing.update({"status": "running", "resumed_utc": utc_now()})
        for field in ("failed_utc", "failure_type", "failure_message"):
            existing.pop(field, None)
        write_json_atomic(metadata_path, existing)
    else:
        if run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        write_json_atomic(metadata_path, metadata)
    write_json_atomic(
        run_dir / "status.json",
        {
            "state": "running", "run_id": metadata["run_id"], "coordinator_pid": os.getpid(),
            "started_utc": metadata.get("started_utc"), "last_updated_utc": utc_now(),
            "completed_sites": [], "completed_site_count": 0,
            "total_sites": len(metadata["held_out_sites"]), "workers": {},
            "current_stage": "parallel_initialised", "held_out_results_embargoed": True,
            "failure_type": None, "failure_message": None,
        },
    )


def worker_command(
    args: argparse.Namespace,
    run_dir: Path,
    worker_index: int,
    sites: list[str],
) -> list[str]:
    worker_id = f"worker_{worker_index:02d}"
    worker_dir = run_dir / "workers" / worker_id
    command = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "run_neural_full.py"),
        "--protocol", args.protocol, "--operator-contract", args.operator_contract,
        "--analysis-protocol", args.analysis_protocol, "--baseline-inputs", args.baseline_inputs,
        "--table", args.table, "--connectomes", args.connectomes,
        "--outer-splits", args.outer_splits, "--inner-splits", args.inner_splits,
        "--output-root", str(run_dir / "workers"), "--run-id", worker_id,
        "--run-kind", args.run_kind, "--held-out-sites", *sites,
        "--execution-shard", "--code-version", args.code_version,
    ]
    if args.fast_smoke:
        command.append("--fast-smoke")
    if args.resume and worker_dir.exists():
        command.append("--resume")
    if (
        not args.resume
        and args.stop_worker_index == worker_index
        and args.stop_after_epoch_checkpoints is not None
    ):
        command.extend(["--stop-after-epoch-checkpoints", str(args.stop_after_epoch_checkpoints)])
    return command


def worker_snapshot(run_dir: Path, assignments: list[list[str]]) -> tuple[dict[str, Any], list[str]]:
    snapshots: dict[str, Any] = {}
    completed: set[str] = set()
    for index, sites in enumerate(assignments):
        worker_id = f"worker_{index:02d}"
        status_path = run_dir / "workers" / worker_id / "status.json"
        if status_path.is_file():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                status = {"state": "unreadable"}
        else:
            status = {"state": "starting"}
        snapshots[worker_id] = {
            "state": status.get("state"), "assigned_sites": sites,
            "current_site": status.get("current_site"),
            "current_configuration": status.get("current_configuration"),
            "current_stage": status.get("current_stage"),
            "completed_site_count": status.get("completed_site_count", 0),
            "last_updated_utc": status.get("last_updated_utc"),
        }
        completed.update(status.get("completed_sites", []))
    return snapshots, sorted(completed)


def update_parent_status(run_dir: Path, assignments: list[list[str]], stage: str) -> None:
    workers, completed = worker_snapshot(run_dir, assignments)
    write_json_atomic(
        run_dir / "status.json",
        {
            "state": "running", "run_id": run_dir.name, "coordinator_pid": os.getpid(),
            "last_updated_utc": utc_now(), "completed_sites": completed,
            "completed_site_count": len(completed), "total_sites": sum(map(len, assignments)),
            "workers": workers, "current_stage": stage,
            "held_out_results_embargoed": True, "failure_type": None, "failure_message": None,
        },
    )


def validate_worker_metadata(
    parent: dict[str, Any], worker: dict[str, Any], worker_id: str, expected_sites: list[str]
) -> None:
    if worker.get("run_id") != worker_id or worker.get("status") != "complete":
        raise ParallelRunError(f"Worker did not complete cleanly: {worker_id}")
    if worker.get("execution_shard") is not True or worker.get("held_out_sites") != expected_sites:
        raise ParallelRunError(f"Worker assignment mismatch: {worker_id}")
    shared_fields = (
        "run_kind", "code_version", "source_hashes", "frozen_input_hashes",
        "configurations", "protocol", "operator_contract", "analysis_protocol", "smoke_override",
    )
    for field in shared_fields:
        if worker.get(field) != parent.get(field):
            raise ParallelRunError(f"Worker immutable field mismatch: {worker_id}/{field}")


def merge_worker_runs(run_dir: Path, metadata: dict[str, Any], assignments: list[list[str]]) -> None:
    folds_final = run_dir / "folds"
    if folds_final.exists():
        raise FileExistsError(f"Canonical folds already exist: {folds_final}")
    folds_temporary = run_dir / f".folds.tmp-{os.getpid()}"
    if folds_temporary.exists():
        shutil.rmtree(folds_temporary)
    folds_temporary.mkdir()
    site_owner = {
        site: f"worker_{index:02d}"
        for index, sites in enumerate(assignments)
        for site in sites
    }
    manifest_sites: list[dict[str, Any]] = []
    ordered_folds = sorted(
        ((int(fold), site) for fold, site in metadata["site_to_outer_fold"].items()),
        key=lambda item: item[0],
    )
    try:
        for outer_fold, site in ordered_folds:
            worker_id = site_owner[site]
            worker_dir = run_dir / "workers" / worker_id
            worker_metadata = json.loads((worker_dir / "metadata.json").read_text(encoding="utf-8"))
            validate_worker_metadata(
                metadata, worker_metadata, worker_id,
                metadata["parallel_execution"]["assignments"][worker_id],
            )
            label = label_for_site(outer_fold, site)
            source = worker_dir / "folds" / label
            if not verified_site(source, outer_fold, site):
                raise ParallelRunError(f"Worker site failed verification: {worker_id}/{site}")
            destination = folds_temporary / label
            shutil.copytree(source, destination)
            if not verified_site(destination, outer_fold, site):
                raise ParallelRunError(f"Copied site failed verification: {site}")
            completion_path = source / "complete.json"
            manifest_sites.append(
                {
                    "outer_fold": outer_fold, "held_out_site": site, "worker_id": worker_id,
                    "source_completion_sha256": sha256_file(completion_path),
                    "artifact_hashes": json.loads(completion_path.read_text(encoding="utf-8"))["artifact_hashes"],
                }
            )
        os.replace(folds_temporary, folds_final)
    except Exception:
        if folds_temporary.exists():
            shutil.rmtree(folds_temporary)
        raise

    aggregate_rows = {name: [] for name in SITE_ARTIFACT_FIELDS}
    for outer_fold, site in ordered_folds:
        site_dir = folds_final / label_for_site(outer_fold, site)
        for name, fields in SITE_ARTIFACT_FIELDS.items():
            with (site_dir / name).open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != fields:
                    raise ParallelRunError(f"Canonical site schema mismatch: {site}/{name}")
                aggregate_rows[name].extend(reader)
    for name, fields in SITE_ARTIFACT_FIELDS.items():
        write_csv(run_dir / name, aggregate_rows[name], fields)
    manifest = {
        "contract_version": 1, "run_id": metadata["run_id"],
        "worker_count": len(assignments), "sites": manifest_sites,
        "results_embargoed": True,
    }
    write_json_atomic(run_dir / "parallel_manifest.json", manifest)
    summary = {
        "run_id": metadata["run_id"], "run_kind": metadata["run_kind"], "status": "complete",
        "held_out_sites": metadata["held_out_sites"],
        "configuration_count": len(metadata["configurations"]),
        "row_counts": {name: len(rows) for name, rows in aggregate_rows.items()},
        "results_embargoed": True,
        "notice": "Counts and integrity state only. Predictive values require a separate successful score-blind audit.",
    }
    write_json_atomic(run_dir / "summary.json", summary)
    final_metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    final_metadata.update(
        {
            "status": "complete", "completed_utc": utc_now(), "notification_status": "pending",
            "artifact_hashes": {
                name: sha256_file(run_dir / name)
                for name in [*SITE_ARTIFACT_FIELDS, "summary.json", "parallel_manifest.json"]
            },
        }
    )
    write_json_atomic(run_dir / "metadata.json", final_metadata)


def notify(args: argparse.Namespace, run_dir: Path, state: str, message: str) -> dict[str, Any]:
    return publish_sns_notification(
        run_dir, args.notification_topic_arn,
        f"BuNN neural parallel {args.run_kind} {state.upper()}: {args.run_id}",
        f"Run {args.run_id} {message}. Check integrity artifacts before reading results.",
    )


def run(args: argparse.Namespace) -> Path:
    if args.require_notification and not args.notification_topic_arn:
        raise ValueError("Required notification topic is missing")
    if args.stop_worker_index is not None:
        if args.run_kind != "smoke" or args.resume:
            raise ValueError("Failure injection is allowed only on a fresh smoke run")
        if args.stop_after_epoch_checkpoints is None or args.stop_after_epoch_checkpoints <= 0:
            raise ValueError("Failure injection requires a positive checkpoint count")
        if not 0 <= args.stop_worker_index < args.worker_count:
            raise ValueError("Failure-injection worker index is outside the worker range")
    metadata, assignments = build_metadata(args)
    run_dir = Path(args.output_root) / args.run_id
    initialise_parent(run_dir, metadata, args.resume)
    if args.require_notification:
        alert = notify(args, run_dir, "started", f"started {args.worker_count} workers")
        if alert.get("status") != "published":
            raise ParallelRunError("Required STARTED notification failed")

    worker_logs = run_dir / "worker_logs"
    worker_logs.mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        }
    )
    processes: list[subprocess.Popen[bytes]] = []
    handles: list[Any] = []
    try:
        for index, sites in enumerate(assignments):
            handle = (worker_logs / f"worker_{index:02d}.log").open("ab")
            handles.append(handle)
            process = subprocess.Popen(
                worker_command(args, run_dir, index, sites), cwd=PROJECT_ROOT,
                env=environment, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
            )
            processes.append(process)
        while any(process.poll() is None for process in processes):
            update_parent_status(run_dir, assignments, "workers_running")
            time.sleep(2)
        update_parent_status(run_dir, assignments, "workers_terminal_merge_pending")
        failures = [index for index, process in enumerate(processes) if process.returncode != 0]
        if failures:
            raise ParallelRunError(f"Parallel workers failed: {failures}")
        merge_worker_runs(run_dir, metadata, assignments)
        workers, completed = worker_snapshot(run_dir, assignments)
        write_json_atomic(
            run_dir / "status.json",
            {
                "state": "complete", "run_id": args.run_id, "coordinator_pid": os.getpid(),
                "last_updated_utc": utc_now(), "completed_sites": completed,
                "completed_site_count": len(completed), "total_sites": len(metadata["held_out_sites"]),
                "workers": workers, "current_stage": "completion_sealed_results_embargoed",
                "held_out_results_embargoed": True, "failure_type": None, "failure_message": None,
            },
        )
        alert = notify(args, run_dir, "complete", "completed and remains results-embargoed")
        final_metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        final_metadata["notification_status"] = alert.get("status")
        write_json_atomic(run_dir / "metadata.json", final_metadata)
        return run_dir
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    finally:
        for handle in handles:
            handle.close()


def mark_failed(args: argparse.Namespace, error: BaseException) -> None:
    run_dir = Path(args.output_root) / args.run_id
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.is_file():
        return
    state = "interrupted" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "status": state, "failed_utc": utc_now(),
            "failure_type": type(error).__name__, "failure_message": str(error),
        }
    )
    write_json_atomic(metadata_path, metadata)
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    status.update(
        {
            "state": state, "current_stage": "terminal_error", "last_updated_utc": utc_now(),
            "failure_type": type(error).__name__, "failure_message": str(error),
            "held_out_results_embargoed": True,
        }
    )
    write_json_atomic(status_path, status)
    alert = notify(args, run_dir, state, f"ended with {type(error).__name__}")
    metadata["notification_status"] = alert.get("status")
    write_json_atomic(metadata_path, metadata)


def handle_sigterm(_signal: int, _frame: Any) -> None:
    raise InterruptedError("received SIGTERM")


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        run(args)
    except BaseException as error:
        mark_failed(args, error)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
