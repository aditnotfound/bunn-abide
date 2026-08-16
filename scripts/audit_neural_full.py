"""Score-blind integrity audit for full or engineering-smoke neural runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from scripts.run_baselines import read_csv, validate_split_contract
from scripts.run_neural_full import (
    CURVE_FIELDS,
    DIAGNOSTIC_FIELDS,
    INNER_SITE_FIELDS,
    METRIC_FIELDS,
    PREDICTION_FIELDS,
    RUNTIME_FIELDS,
    SITE_ARTIFACT_FIELDS,
    TUNING_FIELDS,
    WARNING_FIELDS,
    configurations,
    label_for_site,
)
from src.neural_full_training import selected_candidate_index


class FullAuditError(ValueError):
    """Raised when a neural run is incomplete, inconsistent, or un-auditable."""


def validate_configuration_grid(
    metadata_configurations: list[dict[str, Any]],
    operator_contract: dict[str, Any],
    summary_count: Any,
) -> list[tuple[str, float]]:
    """Validate configuration membership without relying on JSON object-key order."""
    contract_grid = configurations(operator_contract)
    metadata_grid = [
        (str(row["operator"]), float(row["density"]))
        for row in metadata_configurations
    ]
    if (
        len(metadata_grid) != len(contract_grid)
        or len(set(metadata_grid)) != len(metadata_grid)
        or set(metadata_grid) != set(contract_grid)
        or summary_count != len(contract_grid)
    ):
        raise FullAuditError("Configuration grid mismatch")
    return metadata_grid


def validate_parallel_provenance(
    run_dir: Path,
    metadata: dict[str, Any],
    selected_sites: list[str],
) -> dict[str, Any] | None:
    """Validate worker ownership and canonical-copy provenance without reading values."""
    execution_mode = metadata.get("execution_mode")
    if execution_mode is None:
        return None
    if execution_mode != "site_parallel":
        raise FullAuditError("Unknown neural execution mode")
    contract = metadata.get("parallel_execution")
    if not isinstance(contract, dict) or contract.get("contract_version") != 1:
        raise FullAuditError("Invalid parallel execution contract")
    worker_count = contract.get("worker_count")
    assignments = contract.get("assignments")
    if not isinstance(worker_count, int) or worker_count < 2 or not isinstance(assignments, dict):
        raise FullAuditError("Invalid parallel worker declaration")
    expected_workers = [f"worker_{index:02d}" for index in range(worker_count)]
    if sorted(assignments) != expected_workers:
        raise FullAuditError("Parallel worker IDs are incomplete")
    assigned_sites = [site for worker in expected_workers for site in assignments[worker]]
    if len(assigned_sites) != len(set(assigned_sites)) or set(assigned_sites) != set(selected_sites):
        raise FullAuditError("Parallel site ownership is missing or duplicated")
    manifest_path = run_dir / "parallel_manifest.json"
    expected_hash = metadata.get("artifact_hashes", {}).get("parallel_manifest.json")
    if not isinstance(expected_hash, str) or sha256_file(manifest_path) != expected_hash:
        raise FullAuditError("Parallel manifest hash mismatch")
    manifest = load_json(manifest_path)
    if (
        manifest.get("contract_version") != 1
        or manifest.get("run_id") != metadata.get("run_id")
        or manifest.get("worker_count") != worker_count
        or manifest.get("results_embargoed") is not True
    ):
        raise FullAuditError("Parallel manifest contract mismatch")
    entries = manifest.get("sites")
    if not isinstance(entries, list) or len(entries) != len(selected_sites):
        raise FullAuditError("Parallel manifest site count mismatch")
    entry_sites = [entry.get("held_out_site") for entry in entries]
    if len(entry_sites) != len(set(entry_sites)) or set(entry_sites) != set(selected_sites):
        raise FullAuditError("Parallel manifest site coverage mismatch")
    shared_fields = (
        "run_kind", "code_version", "source_hashes", "frozen_input_hashes",
        "configurations", "protocol", "operator_contract", "analysis_protocol", "smoke_override",
    )
    site_to_fold = {site: int(fold) for fold, site in metadata["site_to_outer_fold"].items()}
    for worker_id in expected_workers:
        worker_dir = run_dir / "workers" / worker_id
        worker_metadata = load_json(worker_dir / "metadata.json")
        if (
            worker_metadata.get("run_id") != worker_id
            or worker_metadata.get("status") != "complete"
            or worker_metadata.get("execution_shard") is not True
            or worker_metadata.get("held_out_sites") != assignments[worker_id]
        ):
            raise FullAuditError(f"Parallel worker metadata mismatch: {worker_id}")
        for field in shared_fields:
            if worker_metadata.get(field) != metadata.get(field):
                raise FullAuditError(f"Parallel worker immutable mismatch: {worker_id}/{field}")
        for name in [*SITE_ARTIFACT_FIELDS, "summary.json"]:
            worker_hash = worker_metadata.get("artifact_hashes", {}).get(name)
            if not isinstance(worker_hash, str) or sha256_file(worker_dir / name) != worker_hash:
                raise FullAuditError(f"Parallel worker root hash mismatch: {worker_id}/{name}")
    for entry in entries:
        site = entry["held_out_site"]
        worker_id = entry.get("worker_id")
        if worker_id not in assignments or site not in assignments[worker_id]:
            raise FullAuditError(f"Parallel manifest ownership mismatch: {site}")
        outer_fold = site_to_fold[site]
        if entry.get("outer_fold") != outer_fold:
            raise FullAuditError(f"Parallel manifest fold mismatch: {site}")
        label = label_for_site(outer_fold, site)
        source = run_dir / "workers" / worker_id / "folds" / label
        canonical = run_dir / "folds" / label
        source_completion = source / "complete.json"
        if sha256_file(source_completion) != entry.get("source_completion_sha256"):
            raise FullAuditError(f"Parallel source completion hash mismatch: {site}")
        source_marker = load_json(source_completion)
        canonical_marker = load_json(canonical / "complete.json")
        if source_marker != canonical_marker or source_marker.get("artifact_hashes") != entry.get("artifact_hashes"):
            raise FullAuditError(f"Parallel canonical marker mismatch: {site}")
        for name in SITE_ARTIFACT_FIELDS:
            expected = entry["artifact_hashes"].get(name)
            if sha256_file(source / name) != expected or sha256_file(canonical / name) != expected:
                raise FullAuditError(f"Parallel site copy mismatch: {site}/{name}")
    return {"worker_count": worker_count, "site_count": len(entries)}


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
        raise FullAuditError(f"Unreadable JSON: {path}") from error


def csv_rows(path: Path, fields: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            raise FullAuditError(f"Schema mismatch: {path}")
        return list(reader)


def finite_float(value: str, name: str) -> float:
    try:
        output = float(value)
    except ValueError as error:
        raise FullAuditError(f"Invalid numeric field {name}") from error
    if not math.isfinite(output):
        raise FullAuditError(f"Non-finite numeric field {name}")
    return output


def exact_metric_check(predictions: list[dict[str, str]], metric: dict[str, str]) -> None:
    labels = np.asarray([int(row["label_asd"]) for row in predictions], dtype=int)
    probabilities = np.asarray([finite_float(row["probability_asd"], "probability_asd") for row in predictions])
    predicted = np.asarray([int(row["predicted_asd"]) for row in predictions], dtype=int)
    if np.any((probabilities < 0) | (probabilities > 1)) or np.any(predicted != (probabilities >= 0.5)):
        raise FullAuditError("Prediction probability/threshold contract failed")
    expected = {
        "participants": len(labels), "asd": int(labels.sum()), "control": int((labels == 0).sum()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "sensitivity": float(((predicted == 1) & (labels == 1)).sum() / (labels == 1).sum()),
        "specificity": float(((predicted == 0) & (labels == 0)).sum() / (labels == 0).sum()),
    }
    for field in ("participants", "asd", "control"):
        if int(metric[field]) != expected[field]:
            raise FullAuditError(f"Metric count mismatch: {field}")
    for field in ("balanced_accuracy", "auroc", "sensitivity", "specificity"):
        if not math.isclose(finite_float(metric[field], field), expected[field], abs_tol=1e-12, rel_tol=1e-12):
            raise FullAuditError(f"Saved metric differs from predictions: {field}")


def audit_full_run(
    run_dir: Path,
    table_path: Path,
    outer_splits_path: Path,
    inner_splits_path: Path,
    *,
    expected_run_id: str,
    allow_smoke: bool,
) -> dict[str, Any]:
    metadata = load_json(run_dir / "metadata.json")
    status = load_json(run_dir / "status.json")
    summary = load_json(run_dir / "summary.json")
    if metadata.get("run_id") != expected_run_id or summary.get("run_id") != expected_run_id:
        raise FullAuditError("Run-ID mismatch")
    if metadata.get("status") != "complete" or status.get("state") != "complete":
        raise FullAuditError("Run is not complete")
    run_kind = metadata.get("run_kind")
    if run_kind not in {"full", "smoke"} or (run_kind == "smoke" and not allow_smoke):
        raise FullAuditError("Unexpected run kind")
    if metadata.get("results_embargoed") is not True or status.get("held_out_results_embargoed") is not True:
        raise FullAuditError("Results embargo is not active")
    protocol = metadata["protocol"]
    operator_contract = metadata["operator_contract"]
    config_grid = validate_configuration_grid(
        metadata["configurations"], operator_contract, summary.get("configuration_count")
    )
    selected_sites = list(metadata["held_out_sites"])
    if not selected_sites or len(selected_sites) != len(set(selected_sites)):
        raise FullAuditError("Held-out site list is empty or repeated")
    if run_kind == "full" and len(selected_sites) != 18:
        raise FullAuditError("Full run does not cover all 18 sites")
    parallel_provenance = validate_parallel_provenance(run_dir, metadata, selected_sites)
    table = pd.read_csv(table_path, dtype={"subject_id": str, "site_id": str})
    outer_rows = read_csv(outer_splits_path)
    inner_rows = read_csv(inner_splits_path)

    for name in [*SITE_ARTIFACT_FIELDS, "summary.json"]:
        expected_hash = metadata.get("artifact_hashes", {}).get(name)
        if not isinstance(expected_hash, str) or sha256_file(run_dir / name) != expected_hash:
            raise FullAuditError(f"Root artifact hash mismatch: {name}")

    root_rows = {name: csv_rows(run_dir / name, fields) for name, fields in SITE_ARTIFACT_FIELDS.items()}
    merged_from_sites = {name: [] for name in SITE_ARTIFACT_FIELDS}
    expected_test_subjects: dict[str, set[str]] = {}
    inner_validation_site_counts: dict[str, int] = {}
    for site in selected_sites:
        outer_fold, _train_indices, test_indices, inner_validation = validate_split_contract(table, outer_rows, inner_rows, site)
        expected_test_subjects[site] = set(table.iloc[test_indices]["subject_id"].astype(str))
        override = metadata.get("smoke_override")
        folds = sorted(inner_validation)
        if override:
            folds = folds[: int(override["inner_fold_count"])]
        inner_validation_site_counts[site] = sum(len(set(table.iloc[inner_validation[fold]]["site_id"])) for fold in folds)
        site_dir = run_dir / "folds" / label_for_site(outer_fold, site)
        completion = load_json(site_dir / "complete.json")
        if completion.get("state") != "complete" or completion.get("outer_fold") != outer_fold or completion.get("held_out_site") != site:
            raise FullAuditError(f"Invalid site completion marker: {site}")
        for name, fields in SITE_ARTIFACT_FIELDS.items():
            path = site_dir / name
            if completion.get("artifact_hashes", {}).get(name) != sha256_file(path):
                raise FullAuditError(f"Site artifact hash mismatch: {site}/{name}")
            site_rows = csv_rows(path, fields)
            if completion.get("row_counts", {}).get(name) != len(site_rows):
                raise FullAuditError(f"Site row-count mismatch: {site}/{name}")
            merged_from_sites[name].extend(site_rows)
    for name in SITE_ARTIFACT_FIELDS:
        if root_rows[name] != merged_from_sites[name]:
            raise FullAuditError(f"Root/site aggregation mismatch: {name}")

    override = metadata.get("smoke_override")
    candidates = list(protocol["tuning"]["candidates"])
    tuning_seeds = list(protocol["tuning"]["seeds"])
    final_seeds = list(protocol["final_fit"]["seeds"])
    inner_fold_count = 4
    if override:
        candidates = candidates[: int(override["candidate_count"])]
        tuning_seeds = tuning_seeds[: int(override["tuning_seed_count"])]
        final_seeds = final_seeds[: int(override["final_seed_count"])]
        inner_fold_count = int(override["inner_fold_count"])

    expected_prediction_keys: dict[tuple[str, str, float, int], set[str]] = {}
    for site in selected_sites:
        for operator, density in config_grid:
            for seed in final_seeds:
                expected_prediction_keys[(site, operator, density, int(seed))] = expected_test_subjects[site]
    predictions_by_key: dict[tuple[str, str, float, int], list[dict[str, str]]] = defaultdict(list)
    seen_prediction_subject_keys: set[tuple[str, str, float, int, str]] = set()
    for row in root_rows["predictions.csv"]:
        key = (row["held_out_site"], row["operator"], float(row["density"]), int(row["seed"]))
        subject_key = (*key, row["subject_id"])
        if key not in expected_prediction_keys or subject_key in seen_prediction_subject_keys:
            raise FullAuditError("Unexpected or duplicate held-out prediction")
        if row["site_id"] != row["held_out_site"] or int(row["label_asd"]) not in {0, 1}:
            raise FullAuditError("Prediction site/label contract failed")
        seen_prediction_subject_keys.add(subject_key)
        predictions_by_key[key].append(row)
    if set(predictions_by_key) != set(expected_prediction_keys):
        raise FullAuditError("Prediction configuration coverage mismatch")
    for key, expected_subjects in expected_prediction_keys.items():
        if {row["subject_id"] for row in predictions_by_key[key]} != expected_subjects:
            raise FullAuditError("Prediction participant coverage mismatch")

    metrics_by_key: dict[tuple[str, str, float, int], dict[str, str]] = {}
    for row in root_rows["test_metrics.csv"]:
        key = (row["held_out_site"], row["operator"], float(row["density"]), int(row["seed"]))
        if key in metrics_by_key or key not in expected_prediction_keys:
            raise FullAuditError("Unexpected or duplicate metric row")
        finite_float(row["learning_rate"], "learning_rate")
        finite_float(row["weight_decay"], "weight_decay")
        if int(row["final_epochs"]) <= 0 or int(row["parameter_count"]) <= 0:
            raise FullAuditError("Invalid final-fit metadata")
        metrics_by_key[key] = row
    if set(metrics_by_key) != set(expected_prediction_keys):
        raise FullAuditError("Metric coverage mismatch")
    for key in metrics_by_key:
        exact_metric_check(predictions_by_key[key], metrics_by_key[key])

    tuning_by_config: dict[tuple[str, str, float], list[dict[str, str]]] = defaultdict(list)
    for row in root_rows["tuning_scores.csv"]:
        key = (row["held_out_site"], row["operator"], float(row["density"]))
        tuning_by_config[key].append(row)
        finite_float(row["inner_mean_site_balanced_accuracy"], "inner_mean_site_balanced_accuracy")
    expected_config_keys = {(site, operator, density) for site in selected_sites for operator, density in config_grid}
    if set(tuning_by_config) != expected_config_keys:
        raise FullAuditError("Tuning configuration coverage mismatch")
    selected_rows: dict[tuple[str, str, float], dict[str, str]] = {}
    for key, rows in tuning_by_config.items():
        if len(rows) != len(candidates) or Counter(row["selected"] for row in rows) != Counter({"0": len(candidates) - 1, "1": 1}):
            raise FullAuditError("Candidate count/selection mismatch")
        candidate_rows = [
            {
                "candidate_index": int(row["candidate_index"]),
                "inner_mean_site_balanced_accuracy": finite_float(row["inner_mean_site_balanced_accuracy"], "candidate_score"),
                "weight_decay": finite_float(row["weight_decay"], "weight_decay"),
                "learning_rate": finite_float(row["learning_rate"], "learning_rate"),
            }
            for row in rows
        ]
        expected_index = selected_candidate_index(candidate_rows)
        selected = next(row for row in rows if row["selected"] == "1")
        if int(selected["candidate_index"]) != expected_index or int(selected["selected_final_epoch"]) <= 0:
            raise FullAuditError("Selected candidate or final epoch violates the frozen rule")
        selected_rows[key] = selected

    expected_inner_rows = sum(
        len(config_grid) * len(candidates) * len(tuning_seeds) * inner_validation_site_counts[site]
        for site in selected_sites
    )
    if len(root_rows["inner_site_scores.csv"]) != expected_inner_rows:
        raise FullAuditError("Inner-site score row count mismatch")
    for row in root_rows["inner_site_scores.csv"]:
        if int(row["best_epoch"]) <= 0 or int(row["participants"]) <= 0:
            raise FullAuditError("Invalid inner-site fit row")
        for field in ("balanced_accuracy", "auroc", "sensitivity", "specificity"):
            value = finite_float(row[field], field)
            if not 0 <= value <= 1:
                raise FullAuditError("Inner-site metric outside [0,1]")

    expected_diagnostic_keys = {
        (row["held_out_site"], row["operator"], float(row["density"]), int(row["seed"]), row["subject_id"], layer)
        for row in root_rows["predictions.csv"] for layer in ("encoder", "layer_1", "layer_2")
    }
    observed_diagnostic_keys = set()
    for row in root_rows["diagnostics.csv"]:
        key = (row["held_out_site"], row["operator"], float(row["density"]), int(row["seed"]), row["subject_id"], row["layer"])
        if key in observed_diagnostic_keys:
            raise FullAuditError("Duplicate diagnostic row")
        observed_diagnostic_keys.add(key)
        for field in DIAGNOSTIC_FIELDS[-4:]:
            finite_float(row[field], field)
    if observed_diagnostic_keys != expected_diagnostic_keys:
        raise FullAuditError("Diagnostic coverage mismatch")

    expected_runtime_rows = len(selected_sites) * len(config_grid) * (
        len(candidates) * inner_fold_count * len(tuning_seeds) + len(final_seeds)
    )
    if len(root_rows["fit_runtime.csv"]) != expected_runtime_rows:
        raise FullAuditError("Fit-runtime coverage mismatch")
    for row in root_rows["fit_runtime.csv"]:
        if int(row["epochs_completed"]) <= 0 or finite_float(row["runtime_seconds"], "runtime_seconds") < 0:
            raise FullAuditError("Invalid fit runtime row")
        if int(row["peak_gpu_memory_bytes"]) < 0 or row["resumed"] not in {"0", "1"}:
            raise FullAuditError("Invalid GPU/recovery runtime metadata")

    if not root_rows["training_curves.csv"]:
        raise FullAuditError("Training-curve artifact is empty")
    for row in root_rows["training_curves.csv"]:
        if int(row["epoch"]) < 0:
            raise FullAuditError("Negative training epoch")
        finite_float(row["training_bce_loss"], "training_bce_loss")
        finite_float(row["maximum_gradient_norm"], "maximum_gradient_norm")
        if row["fit_scope"] == "inner_tuning":
            finite_float(row["validation_mean_site_bce"], "validation_mean_site_bce")

    warning_count = len(root_rows["fit_warnings.csv"])
    report = {
        "state": "passed", "run_id": expected_run_id, "run_kind": run_kind,
        "sites_checked": len(selected_sites), "configurations_per_site": len(config_grid),
        "prediction_rows_checked": len(root_rows["predictions.csv"]),
        "metric_rows_recomputed_without_reporting_values": len(root_rows["test_metrics.csv"]),
        "tuning_rows_checked": len(root_rows["tuning_scores.csv"]),
        "inner_site_rows_checked": len(root_rows["inner_site_scores.csv"]),
        "diagnostic_rows_checked": len(root_rows["diagnostics.csv"]),
        "runtime_rows_checked": len(root_rows["fit_runtime.csv"]),
        "warning_rows": warning_count, "results_remain_embargoed": True,
        "notice": "Score-blind integrity certificate; no predictive or representation value is reported.",
    }
    if parallel_provenance is not None:
        report["parallel_workers_checked"] = parallel_provenance["worker_count"]
        report["parallel_site_copies_checked"] = parallel_provenance["site_count"]
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--table", default="data/processed/abide_i_baseline_table.csv", type=Path)
    parser.add_argument("--outer-splits", default="data/processed/splits/outer_loso_assignments.csv", type=Path)
    parser.add_argument("--inner-splits", default="data/processed/splits/inner_grouped_assignments.csv", type=Path)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_full_run(
        args.run_dir, args.table, args.outer_splits, args.inner_splits,
        expected_run_id=args.expected_run_id, allow_smoke=args.allow_smoke,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
