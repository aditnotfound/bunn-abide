"""Frozen E2 aggregation, inference, and evidence classification."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


class E2AnalysisError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= 0.5).astype(int)
    positive = labels == 1
    negative = labels == 0
    sensitivity = float((prediction[positive] == 1).mean())
    specificity = float((prediction[negative] == 0).mean())
    order = np.argsort(probability, kind="mergesort")
    sorted_probability = probability[order]
    ranks = np.empty(len(probability), dtype=float)
    start = 0
    while start < len(probability):
        end = start + 1
        while end < len(probability) and sorted_probability[end] == sorted_probability[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positives = int(positive.sum())
    negatives = int(negative.sum())
    auroc = (ranks[positive].sum() - positives * (positives + 1) / 2.0) / (positives * negatives)
    return {
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "auroc": float(auroc),
        "brier_score": float(np.mean((probability - labels) ** 2)),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def paired_bootstrap(values: np.ndarray, *, draws: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(draws, len(values)))
    samples = values[indices].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    extreme = 0
    total = 2 ** len(values)
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(float(np.mean(values * np.asarray(signs))))
        extreme += statistic >= observed - 1e-15
    return extreme / total


def load_replicate_metrics(
    run_dir: Path, protocol: dict[str, Any], protocol_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manager_path = run_dir / "manager_complete.json"
    manager = json.loads(manager_path.read_text(encoding="utf-8"))
    if manager.get("state") != "complete_all_cells_score_blind_audited":
        raise E2AnalysisError("E2 manager is not complete and score-blind audited")
    if manager.get("protocol_sha256") != protocol_hash or manager.get("mode") != "full":
        raise E2AnalysisError("E2 manager protocol or run mode mismatch")
    expected = 10 * 7 * len(protocol["data"]["replicate_seeds"])
    if manager.get("expected_cells") != expected or manager.get("completed_cells") != expected:
        raise E2AnalysisError("E2 manager cell count differs from the frozen matrix")
    full_audit_path = run_dir / "score_blind_full_audit.json"
    if not full_audit_path.is_file():
        raise E2AnalysisError("Independent E2 full-run audit is missing")
    full_audit = json.loads(full_audit_path.read_text(encoding="utf-8"))
    if (
        full_audit.get("state") != "complete_full_run_reaudit_passed_score_blind"
        or full_audit.get("protocol_sha256") != protocol_hash
        or full_audit.get("manager_completion_sha256") != sha256_file(manager_path)
        or full_audit.get("audited_cells") != expected
        or full_audit.get("sealed_replicates") != 10 * len(protocol["data"]["replicate_seeds"])
        or full_audit.get("scientific_values_opened") is not False
    ):
        raise E2AnalysisError("Independent E2 full-run audit does not match the frozen source")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, float, int, str]] = set()
    for cell_id, expected_audit_hash in sorted(manager["audit_hashes"].items()):
        cell_dir = run_dir / "cells" / cell_id
        audit_path = cell_dir / "score_blind_audit.json"
        if sha256_file(audit_path) != expected_audit_hash:
            raise E2AnalysisError(f"Audit hash mismatch: {cell_id}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("state") != "audit_passed_score_blind":
            raise E2AnalysisError(f"Score-blind audit did not pass: {cell_id}")
        manifest_path = cell_dir / "cell_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sha256_file(manifest_path) != audit.get("manifest_sha256"):
            raise E2AnalysisError(f"Manifest hash mismatch: {cell_id}")
        immutable = manifest["immutable"]
        if immutable.get("protocol_sha256") != protocol_hash or not immutable.get("evaluate_test"):
            raise E2AnalysisError(f"Invalid scientific cell contract: {cell_id}")
        prediction_path = cell_dir / "test_predictions.npz"
        if sha256_file(prediction_path) != audit.get("prediction_sha256"):
            raise E2AnalysisError(f"Prediction hash mismatch: {cell_id}")
        values = np.load(prediction_path)
        metrics = binary_metrics(values["label"], values["probability"])
        key = (
            str(immutable["family"]), float(immutable["transport_noise_degrees"]),
            int(immutable["data_seed"]), str(immutable["operator"]),
        )
        if key in seen:
            raise E2AnalysisError(f"Duplicate E2 cell: {key}")
        seen.add(key)
        rows.append({
            "family": key[0], "transport_noise_degrees": key[1], "replicate_seed": key[2],
            "operator": key[3], **metrics,
            "mean_transport_error": float(np.mean(values["transport_error"])),
            "parameter_count": int(manifest["parameter_count"]),
            "runtime_seconds": float(manifest["runtime_seconds"]),
            "best_epoch": int(manifest["best_epoch"]),
        })
    if len(rows) != expected:
        raise E2AnalysisError("E2 row count differs from frozen expectation")
    return rows, manager


def _value_map(rows: list[dict[str, Any]], endpoint: str) -> dict[tuple[str, float, int, str], float]:
    return {
        (row["family"], row["transport_noise_degrees"], row["replicate_seed"], row["operator"]): float(row[endpoint])
        for row in rows
    }


def frozen_contrasts(rows: list[dict[str, Any]], analysis: dict[str, Any]) -> list[dict[str, Any]]:
    values = _value_map(rows, "balanced_accuracy")
    seeds = sorted({row["replicate_seed"] for row in rows})

    def v(seed: int, family: str, operator: str, noise: float = 0.0) -> float:
        try:
            return values[(family, noise, seed, operator)]
        except KeyError as error:
            raise E2AnalysisError(f"Missing frozen contrast cell: {error}") from error

    definitions = [
        ("primary_conditional_advantage", lambda s: (v(s, "S1_recoverable_geometry", "learned_bunn") - v(s, "S1_recoverable_geometry", "gcn")) - (v(s, "S0_no_geometry", "learned_bunn") - v(s, "S0_no_geometry", "gcn"))),
        ("oracle_positive_control", lambda s: v(s, "S1_recoverable_geometry", "oracle_true_map") - v(s, "S1_recoverable_geometry", "gcn")),
        ("matched_capacity_anchor", lambda s: v(s, "S1_recoverable_geometry", "learned_bunn") - v(s, "S1_recoverable_geometry", "learned_local")),
        ("S0_capacity_warning", lambda s: v(s, "S0_no_geometry", "learned_bunn") - v(s, "S0_no_geometry", "gcn")),
        ("S1_minus_S2_learned_bunn", lambda s: v(s, "S1_recoverable_geometry", "learned_bunn") - v(s, "S2_incorrect_topology", "learned_bunn")),
        ("S1_minus_S3_learned_bunn", lambda s: v(s, "S1_recoverable_geometry", "learned_bunn") - v(s, "S3_shuffled_geometry", "learned_bunn")),
        ("S1_minus_S5_learned_bunn", lambda s: v(s, "S1_recoverable_geometry", "learned_bunn") - v(s, "S5_unlearnable_subject_frames", "learned_bunn")),
        ("S6_learned_bunn_minus_gcn", lambda s: v(s, "S6_global_feature_analogue", "learned_bunn") - v(s, "S6_global_feature_analogue", "gcn")),
    ]
    output: list[dict[str, Any]] = []
    for index, (name, function) in enumerate(definitions):
        paired = np.asarray([function(seed) for seed in seeds], dtype=float) * 100.0
        lower, upper = paired_bootstrap(
            paired, draws=int(analysis["inference"]["paired_percentile_bootstrap_draws"]),
            seed=int(analysis["inference"]["bootstrap_seed"]) + index,
        )
        output.append({
            "contrast": name, "replicates": len(paired), "mean_difference_pp": float(paired.mean()),
            "ci95_lower_pp": lower, "ci95_upper_pp": upper,
            "exact_sign_flip_p": exact_sign_flip_p(paired),
            "replicate_values_pp": paired.tolist(),
        })
    return output


def classify_evidence(contrasts: list[dict[str, Any]], analysis: dict[str, Any]) -> dict[str, Any]:
    by_name = {row["contrast"]: row for row in contrasts}
    oracle = by_name["oracle_positive_control"]["ci95_lower_pp"] > 0
    primary_row = by_name["primary_conditional_advantage"]
    primary = (
        primary_row["mean_difference_pp"] >= analysis["inference"]["minimum_practical_difference_percentage_points"]
        and primary_row["ci95_lower_pp"] > 0
        and primary_row["exact_sign_flip_p"] < analysis["inference"]["alpha"]
    )
    anchor = by_name["matched_capacity_anchor"]["ci95_lower_pp"] > 0
    capacity_warning = by_name["S0_capacity_warning"]["mean_difference_pp"] > analysis["inference"]["minimum_practical_difference_percentage_points"]
    if not oracle:
        classification = "invalid_positive_control"
    elif primary and anchor and not capacity_warning:
        classification = "supported"
    elif primary and (capacity_warning or not anchor):
        classification = "capacity_or_optimization_confound"
    elif oracle:
        classification = "oracle_only"
    else:
        classification = "no_mechanistic_support_detected"
    return {
        "classification": classification,
        "oracle_positive_control_passed": oracle,
        "primary_conditional_advantage_passed": primary,
        "matched_capacity_anchor_passed": anchor,
        "capacity_warning_triggered": capacity_warning,
    }
