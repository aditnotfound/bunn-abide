"""Run the frozen Step 11 confirmatory analysis for the audited neural study.

The command is deliberately guarded by an exact run-ID acknowledgement.  All
analysis choices are loaded from a separately versioned execution contract
that was frozen after the score-blind integrity audit and before unsealing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_baselines import SITE_METRIC_FIELDS as BASELINE_SITE_METRIC_FIELDS
from scripts.run_neural_full import (
    DIAGNOSTIC_FIELDS,
    METRIC_FIELDS,
    PREDICTION_FIELDS,
    RUNTIME_FIELDS,
    TUNING_FIELDS,
    WARNING_FIELDS,
)


SITE_SEED_FIELDS = [
    "operator", "density", "seed", "held_out_site", "participants", "asd", "control",
    "balanced_accuracy", "auroc", "sensitivity", "specificity", "parameter_count",
]
SITE_CONFIGURATION_FIELDS = [
    "operator", "density", "held_out_site", "seed_count", "participants",
    "mean_balanced_accuracy", "sd_balanced_accuracy", "mean_auroc", "mean_sensitivity",
    "mean_specificity", "parameter_count",
]
SEED_SUMMARY_FIELDS = [
    "operator", "density", "seed", "held_out_sites", "unweighted_mean_site_balanced_accuracy",
]
CURVE_FIELDS = [
    "curve_operator", "anchor_operator", "held_out_site", "normalized_auc_balanced_accuracy",
    "anchor_balanced_accuracy", "mean_nonzero_density_balanced_accuracy", "change_from_anchor",
]
CONTRAST_FIELDS = [
    "contrast", "estimand", "primary", "held_out_sites", "observed_mean_difference",
    "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_resamples", "bootstrap_seed",
    "exact_sign_flip_p", "practical_margin", "upper_excludes_practical_margin",
]
DENSITY_CONTRAST_FIELDS = [
    "contrast_family", "density", "held_out_sites", "observed_mean_difference",
    "bootstrap_ci_low", "bootstrap_ci_high", "raw_exact_sign_flip_p", "holm_adjusted_p",
    "bootstrap_resamples", "bootstrap_seed",
]
REPRESENTATION_FIELDS = [
    "operator", "density", "held_out_site", "layer", "seed_count", "participants_per_seed",
    "normalized_effective_rank", "normalized_dispersion", "mean_pairwise_cosine",
    "invariant_edge_transport_distance",
]
REPRESENTATION_CONTRAST_FIELDS = [
    "endpoint", "layer", "estimand", "primary", "held_out_sites", "observed_mean_difference",
    "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_resamples", "bootstrap_seed",
]
RUNTIME_SUMMARY_FIELDS = [
    "fit_scope", "operator", "density", "fits", "total_runtime_seconds", "mean_runtime_seconds",
    "median_runtime_seconds", "maximum_peak_gpu_memory_bytes", "resumed_fits",
]
SELECTED_TUNING_FIELDS = [
    "operator", "density", "held_out_site", "candidate_index", "learning_rate", "weight_decay",
    "inner_mean_site_balanced_accuracy", "selected_final_epoch",
]
WARNING_SUMMARY_FIELDS = ["operator", "density", "fit_scope", "failure_type", "warning_rows"]
ENDPOINTS = [
    "normalized_effective_rank", "normalized_dispersion", "mean_pairwise_cosine",
    "invariant_edge_transport_distance",
]


class NeuralAnalysisError(ValueError):
    """Raised when an artifact cannot support the frozen Step 11 analysis."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--integrity-certificate", required=True, type=Path)
    parser.add_argument(
        "--analysis-contract", type=Path,
        default=Path("configs/neural_confirmatory_analysis_v1.json"),
    )
    parser.add_argument("--baseline-per-site", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--confirm-unblind-run-id", default=None)
    return parser.parse_args()


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
        raise NeuralAnalysisError(f"Unreadable JSON: {path}") from error


def read_csv(path: Path, expected_fields: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise NeuralAnalysisError(f"Missing required artifact: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_fields:
            raise NeuralAnalysisError(f"Schema mismatch for {path.name}: {reader.fieldnames!r}")
        return list(reader)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise NeuralAnalysisError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(parsed):
        raise NeuralAnalysisError(f"{label} is not finite")
    return parsed


def finite_int(value: Any, label: str) -> int:
    parsed = finite_float(value, label)
    if not parsed.is_integer():
        raise NeuralAnalysisError(f"{label} is not an integer")
    return int(parsed)


def normalized_auc(densities: list[float], values: list[float]) -> float:
    if len(densities) != len(values) or len(densities) < 2:
        raise NeuralAnalysisError("A density curve requires aligned values and at least two points")
    if any(right <= left for left, right in zip(densities, densities[1:])):
        raise NeuralAnalysisError("Density points must be strictly increasing")
    area = sum(
        (right_x - left_x) * (left_y + right_y) / 2.0
        for left_x, right_x, left_y, right_y in zip(
            densities, densities[1:], values, values[1:]
        )
    )
    return float(area / (densities[-1] - densities[0]))


def bootstrap_interval(
    differences: np.ndarray, indices: np.ndarray
) -> tuple[float, float, float]:
    if differences.ndim != 1 or differences.size != indices.shape[1]:
        raise NeuralAnalysisError("Bootstrap differences are not site-aligned")
    means = differences[indices].mean(axis=1)
    return (
        float(differences.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def sign_matrix(site_count: int) -> np.ndarray:
    if site_count <= 0 or site_count > 20:
        raise NeuralAnalysisError("Exact sign-flip implementation supports 1-20 paired sites")
    masks = np.arange(1 << site_count, dtype=np.uint32)[:, None]
    bits = (masks >> np.arange(site_count, dtype=np.uint32)) & 1
    return np.where(bits == 1, 1.0, -1.0).astype(np.float32)


def exact_sign_flip_p(differences: np.ndarray, signs: np.ndarray) -> float:
    observed = abs(float(differences.mean()))
    null_statistics = np.abs((signs @ differences.astype(float)) / differences.size)
    return float(np.mean(null_statistics >= observed - 1e-15))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def validate_gate(
    run_dir: Path, certificate_path: Path, contract: dict[str, Any]
) -> dict[str, Any]:
    certificate = load_json(certificate_path)
    metadata = load_json(run_dir / "metadata.json")
    run_id = contract.get("input_run_id")
    if certificate.get("state") != "passed" or certificate.get("run_id") != run_id:
        raise NeuralAnalysisError("The required score-blind integrity certificate did not pass")
    if certificate.get("results_remain_embargoed") is not True:
        raise NeuralAnalysisError("Integrity certificate does not preserve the pre-analysis embargo")
    if metadata.get("run_id") != run_id or metadata.get("status") != "complete":
        raise NeuralAnalysisError("Run metadata does not identify the frozen completed run")
    if metadata.get("run_kind") != "full" or metadata.get("results_embargoed") is not True:
        raise NeuralAnalysisError("Refusing a non-full or previously unguarded run")
    expected_sites = list(metadata.get("held_out_sites", []))
    if len(expected_sites) != 18 or len(set(expected_sites)) != 18:
        raise NeuralAnalysisError("The confirmatory run must contain exactly 18 unique sites")
    for name in ["predictions.csv", "test_metrics.csv", "diagnostics.csv", "fit_runtime.csv", "tuning_scores.csv", "fit_warnings.csv"]:
        expected = metadata.get("artifact_hashes", {}).get(name)
        if not isinstance(expected, str) or sha256_file(run_dir / name) != expected:
            raise NeuralAnalysisError(f"Root artifact hash changed after audit: {name}")
    return metadata


def compute_site_seed_metrics(
    prediction_rows: list[dict[str, str]], runner_metric_rows: list[dict[str, str]],
    threshold: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, int, str], list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[str, float, int, str, str]] = set()
    for row in prediction_rows:
        key = (
            row["operator"], finite_float(row["density"], "density"),
            finite_int(row["seed"], "seed"), row["held_out_site"],
        )
        subject_key = (*key, row["subject_id"])
        if subject_key in seen:
            raise NeuralAnalysisError("Duplicate participant prediction")
        seen.add(subject_key)
        probability = finite_float(row["probability_asd"], "probability_asd")
        predicted = finite_int(row["predicted_asd"], "predicted_asd")
        label = finite_int(row["label_asd"], "label_asd")
        if not 0.0 <= probability <= 1.0 or predicted != int(probability >= threshold) or label not in {0, 1}:
            raise NeuralAnalysisError("Prediction threshold/range contract failed")
        grouped[key].append(row)

    runner = {}
    for row in runner_metric_rows:
        key = (
            row["operator"], finite_float(row["density"], "metric density"),
            finite_int(row["seed"], "metric seed"), row["held_out_site"],
        )
        if key in runner:
            raise NeuralAnalysisError("Duplicate runner metric")
        runner[key] = row
    if set(grouped) != set(runner):
        raise NeuralAnalysisError("Prediction and runner-metric coverage differ")

    output: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda value: (value[3], value[0], value[1], value[2])):
        operator, density, seed, site = key
        rows = grouped[key]
        labels = np.asarray([finite_int(row["label_asd"], "label") for row in rows], dtype=int)
        probabilities = np.asarray([finite_float(row["probability_asd"], "probability") for row in rows])
        predicted = probabilities >= threshold
        if set(labels.tolist()) != {0, 1}:
            raise NeuralAnalysisError(f"Held-out site lacks one class: {site}")
        positives = labels == 1
        negatives = ~positives
        calculated = {
            "participants": len(rows), "asd": int(positives.sum()), "control": int(negatives.sum()),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "auroc": float(roc_auc_score(labels, probabilities)),
            "sensitivity": float(predicted[positives].mean()),
            "specificity": float((~predicted[negatives]).mean()),
        }
        source = runner[key]
        for field in ("participants", "asd", "control"):
            if finite_int(source[field], field) != calculated[field]:
                raise NeuralAnalysisError(f"Runner {field} mismatch for {key}")
        for field in ("balanced_accuracy", "auroc", "sensitivity", "specificity"):
            if not math.isclose(finite_float(source[field], field), calculated[field], abs_tol=1e-12, rel_tol=0.0):
                raise NeuralAnalysisError(f"Runner {field} mismatch for {key}")
        output.append({
            "operator": operator, "density": density, "seed": seed, "held_out_site": site,
            **calculated, "parameter_count": finite_int(source["parameter_count"], "parameter_count"),
        })
    return output


def aggregate_configurations(site_seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in site_seed_rows:
        grouped[(row["operator"], float(row["density"]), row["held_out_site"])].append(row)
    output = []
    for (operator, density, site), rows in sorted(grouped.items(), key=lambda item: (item[0][2], item[0][0], item[0][1])):
        seeds = {int(row["seed"]) for row in rows}
        if len(rows) != 5 or len(seeds) != 5:
            raise NeuralAnalysisError(f"Expected five final seeds for {site}/{operator}/{density}")
        parameter_counts = {int(row["parameter_count"]) for row in rows}
        participants = {int(row["participants"]) for row in rows}
        if len(parameter_counts) != 1 or len(participants) != 1:
            raise NeuralAnalysisError("Seed-level metadata changed within a configuration")
        ba = np.asarray([float(row["balanced_accuracy"]) for row in rows])
        output.append({
            "operator": operator, "density": density, "held_out_site": site,
            "seed_count": len(seeds), "participants": participants.pop(),
            "mean_balanced_accuracy": float(ba.mean()), "sd_balanced_accuracy": float(ba.std(ddof=1)),
            "mean_auroc": float(np.mean([row["auroc"] for row in rows])),
            "mean_sensitivity": float(np.mean([row["sensitivity"] for row in rows])),
            "mean_specificity": float(np.mean([row["specificity"] for row in rows])),
            "parameter_count": parameter_counts.pop(),
        })
    return output


def seed_summaries(site_seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, int], list[float]] = defaultdict(list)
    for row in site_seed_rows:
        grouped[(row["operator"], float(row["density"]), int(row["seed"]))].append(float(row["balanced_accuracy"]))
    return [
        {"operator": key[0], "density": key[1], "seed": key[2], "held_out_sites": len(values),
         "unweighted_mean_site_balanced_accuracy": float(np.mean(values))}
        for key, values in sorted(grouped.items())
    ]


def configuration_map(rows: list[dict[str, Any]], value_field: str) -> dict[tuple[str, str, float], float]:
    result = {}
    for row in rows:
        key = (row["held_out_site"], row["operator"], float(row["density"]))
        if key in result:
            raise NeuralAnalysisError(f"Duplicate site/configuration aggregate: {key}")
        result[key] = float(row[value_field])
    return result


def build_predictive_curves(
    config_rows: list[dict[str, Any]], sites: list[str], contract: dict[str, Any]
) -> list[dict[str, Any]]:
    values = configuration_map(config_rows, "mean_balanced_accuracy")
    densities = [float(value) for value in contract["densities"]]
    nonzero = densities[1:]
    output = []
    for site in sorted(sites):
        for operator, anchor in contract["curve_anchors"].items():
            anchor_value = values[(site, anchor, 0.0)]
            curve_values = [anchor_value] + [values[(site, operator, density)] for density in nonzero]
            area = normalized_auc(densities, curve_values)
            output.append({
                "curve_operator": operator, "anchor_operator": anchor, "held_out_site": site,
                "normalized_auc_balanced_accuracy": area, "anchor_balanced_accuracy": anchor_value,
                "mean_nonzero_density_balanced_accuracy": float(np.mean(curve_values[1:])),
                "change_from_anchor": area - anchor_value,
            })
    return output


def baseline_map(path: Path, sites: list[str], model: str) -> dict[str, float]:
    rows = read_csv(path, BASELINE_SITE_METRIC_FIELDS)
    selected = [row for row in rows if row["model"] == model]
    result = {row["held_out_site"]: finite_float(row["balanced_accuracy"], "baseline BA") for row in selected}
    if set(result) != set(sites) or len(selected) != len(sites):
        raise NeuralAnalysisError("Baseline reference does not cover the same 18 sites exactly once")
    return result


def paired_row(
    name: str, estimand: str, differences: np.ndarray, indices: np.ndarray,
    signs: np.ndarray, contract: dict[str, Any], *, primary: bool,
) -> dict[str, Any]:
    observed, low, high = bootstrap_interval(differences, indices)
    margin = float(contract["practical_margin_balanced_accuracy"])
    return {
        "contrast": name, "estimand": estimand, "primary": int(primary),
        "held_out_sites": differences.size, "observed_mean_difference": observed,
        "bootstrap_ci_low": low, "bootstrap_ci_high": high,
        "bootstrap_resamples": indices.shape[0], "bootstrap_seed": contract["bootstrap"]["seed"],
        "exact_sign_flip_p": exact_sign_flip_p(differences, signs), "practical_margin": margin,
        "upper_excludes_practical_margin": int(high < margin),
    }


def predictive_contrasts(
    curves: list[dict[str, Any]], config_rows: list[dict[str, Any]], baseline: dict[str, float],
    sites: list[str], indices: np.ndarray, signs: np.ndarray, contract: dict[str, Any],
) -> list[dict[str, Any]]:
    curve = {(row["held_out_site"], row["curve_operator"]): row for row in curves}
    config = configuration_map(config_rows, "mean_balanced_accuracy")
    density_values = [float(value) for value in contract["densities"]][1:]
    arrays = {
        "learned_bunn_curve_minus_gcn_curve": np.asarray([
            curve[(site, "learned_bunn")]["normalized_auc_balanced_accuracy"]
            - curve[(site, "gcn")]["normalized_auc_balanced_accuracy"] for site in sites
        ]),
        "learned_bunn_curve_minus_trivial_bundle_curve": np.asarray([
            curve[(site, "learned_bunn")]["normalized_auc_balanced_accuracy"]
            - curve[(site, "trivial_bundle")]["normalized_auc_balanced_accuracy"] for site in sites
        ]),
        "learned_bunn_nonzero_minus_learned_local": np.asarray([
            np.mean([config[(site, "learned_bunn", density)] for density in density_values])
            - config[(site, "learned_local", 0.0)] for site in sites
        ]),
        "learned_bunn_nonzero_minus_identity": np.asarray([
            np.mean([config[(site, "learned_bunn", density)] for density in density_values])
            - config[(site, "identity", 0.0)] for site in sites
        ]),
        "learned_bunn_curve_minus_connectome_elastic_net": np.asarray([
            curve[(site, "learned_bunn")]["normalized_auc_balanced_accuracy"] - baseline[site]
            for site in sites
        ]),
    }
    descriptions = {
        "learned_bunn_curve_minus_gcn_curve": "normalized balanced-accuracy density-curve area; BuNN minus GCN",
        "learned_bunn_curve_minus_trivial_bundle_curve": "normalized balanced-accuracy density-curve area; BuNN minus trivial bundle",
        "learned_bunn_nonzero_minus_learned_local": "mean nonzero-density balanced accuracy; BuNN minus learned-local",
        "learned_bunn_nonzero_minus_identity": "mean nonzero-density balanced accuracy; BuNN minus identity",
        "learned_bunn_curve_minus_connectome_elastic_net": "BuNN normalized curve area minus site-matched connectome elastic net",
    }
    return [
        paired_row(name, descriptions[name], values, indices, signs, contract,
                   primary=name == "learned_bunn_curve_minus_gcn_curve")
        for name, values in arrays.items()
    ]


def density_contrasts(
    config_rows: list[dict[str, Any]], baseline: dict[str, float], sites: list[str],
    indices: np.ndarray, signs: np.ndarray, contract: dict[str, Any],
) -> list[dict[str, Any]]:
    values = configuration_map(config_rows, "mean_balanced_accuracy")
    families = {
        "learned_bunn_minus_gcn": lambda site, density: values[(site, "learned_bunn", density)] - values[(site, "gcn", density)],
        "learned_bunn_minus_trivial_bundle": lambda site, density: values[(site, "learned_bunn", density)] - values[(site, "trivial_bundle", density)],
        "learned_bunn_minus_learned_local": lambda site, density: values[(site, "learned_bunn", density)] - values[(site, "learned_local", 0.0)],
        "learned_bunn_minus_identity": lambda site, density: values[(site, "learned_bunn", density)] - values[(site, "identity", 0.0)],
        "learned_bunn_minus_connectome_elastic_net": lambda site, density: values[(site, "learned_bunn", density)] - baseline[site],
    }
    output = []
    for family, function in families.items():
        family_rows = []
        raw_p = []
        for density in [float(value) for value in contract["densities"]][1:]:
            differences = np.asarray([function(site, density) for site in sites])
            observed, low, high = bootstrap_interval(differences, indices)
            p_value = exact_sign_flip_p(differences, signs)
            raw_p.append(p_value)
            family_rows.append({
                "contrast_family": family, "density": density, "held_out_sites": len(sites),
                "observed_mean_difference": observed, "bootstrap_ci_low": low,
                "bootstrap_ci_high": high, "raw_exact_sign_flip_p": p_value,
                "bootstrap_resamples": indices.shape[0], "bootstrap_seed": contract["bootstrap"]["seed"],
            })
        adjusted = holm_adjust(raw_p)
        for row, value in zip(family_rows, adjusted, strict=True):
            row["holm_adjusted_p"] = value
            output.append(row)
    return output


def aggregate_representation(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    per_seed: dict[tuple[str, float, str, str, int], dict[str, list[float]]] = defaultdict(
        lambda: {endpoint: [] for endpoint in ENDPOINTS}
    )
    seen = set()
    for row in rows:
        key = (
            row["operator"], finite_float(row["density"], "diagnostic density"),
            row["held_out_site"], row["layer"], finite_int(row["seed"], "diagnostic seed"),
        )
        subject_key = (*key, row["subject_id"])
        if subject_key in seen:
            raise NeuralAnalysisError("Duplicate participant diagnostic")
        seen.add(subject_key)
        for endpoint in ENDPOINTS:
            per_seed[key][endpoint].append(finite_float(row[endpoint], endpoint))
    grouped: dict[tuple[str, float, str, str], list[dict[str, float]]] = defaultdict(list)
    counts: dict[tuple[str, float, str, str], set[int]] = defaultdict(set)
    for key, endpoint_values in per_seed.items():
        operator, density, site, layer, _seed = key
        grouped[(operator, density, site, layer)].append(
            {endpoint: float(np.mean(values)) for endpoint, values in endpoint_values.items()}
        )
        counts[(operator, density, site, layer)].add(len(next(iter(endpoint_values.values()))))
    output = []
    for key, seed_rows in sorted(grouped.items(), key=lambda item: (item[0][2], item[0][0], item[0][1], item[0][3])):
        if len(seed_rows) != 5 or len(counts[key]) != 1:
            raise NeuralAnalysisError(f"Diagnostic seed/participant coverage mismatch: {key}")
        operator, density, site, layer = key
        output.append({
            "operator": operator, "density": density, "held_out_site": site, "layer": layer,
            "seed_count": 5, "participants_per_seed": next(iter(counts[key])),
            **{endpoint: float(np.mean([row[endpoint] for row in seed_rows])) for endpoint in ENDPOINTS},
        })
    return output


def representation_contrasts(
    rows: list[dict[str, Any]], sites: list[str], indices: np.ndarray,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    values = {
        (row["held_out_site"], row["operator"], float(row["density"]), row["layer"]): row
        for row in rows
    }
    densities = [float(value) for value in contract["densities"]]

    def curve_change(site: str, operator: str, anchor: str, endpoint: str, layer: str) -> float:
        anchor_value = float(values[(site, anchor, 0.0, layer)][endpoint])
        curve_values = [anchor_value] + [
            float(values[(site, operator, density, layer)][endpoint]) for density in densities[1:]
        ]
        return normalized_auc(densities, curve_values) - anchor_value

    specifications = [
        ("normalized_effective_rank", "layer_2", True),
        ("normalized_dispersion", "layer_2", False),
        ("mean_pairwise_cosine", "layer_2", False),
        ("invariant_edge_transport_distance", "layer_2", False),
    ]
    output = []
    for endpoint, layer, primary in specifications:
        differences = np.asarray([
            curve_change(site, "learned_bunn", "learned_local", endpoint, layer)
            - curve_change(site, "gcn", "identity", endpoint, layer)
            for site in sites
        ])
        observed, low, high = bootstrap_interval(differences, indices)
        output.append({
            "endpoint": endpoint, "layer": layer,
            "estimand": "BuNN minus GCN difference in normalized density-curve change from matched zero-density anchor",
            "primary": int(primary), "held_out_sites": len(sites),
            "observed_mean_difference": observed, "bootstrap_ci_low": low, "bootstrap_ci_high": high,
            "bootstrap_resamples": indices.shape[0], "bootstrap_seed": contract["bootstrap"]["seed"],
        })

    differences = []
    for site in sites:
        def encoder_layer_change(operator: str, anchor: str) -> float:
            anchor_value = (
                float(values[(site, anchor, 0.0, "layer_2")]["normalized_effective_rank"])
                - float(values[(site, anchor, 0.0, "encoder")]["normalized_effective_rank"])
            )
            curve_values = [anchor_value]
            for density in densities[1:]:
                curve_values.append(
                    float(values[(site, operator, density, "layer_2")]["normalized_effective_rank"])
                    - float(values[(site, operator, density, "encoder")]["normalized_effective_rank"])
                )
            return normalized_auc(densities, curve_values) - anchor_value
        differences.append(
            encoder_layer_change("learned_bunn", "learned_local")
            - encoder_layer_change("gcn", "identity")
        )
    observed, low, high = bootstrap_interval(np.asarray(differences), indices)
    output.append({
        "endpoint": "encoder_to_layer_2_normalized_effective_rank_change", "layer": "encoder_to_layer_2",
        "estimand": "BuNN minus GCN difference in normalized density-curve change from matched zero-density anchor",
        "primary": 0, "held_out_sites": len(sites), "observed_mean_difference": observed,
        "bootstrap_ci_low": low, "bootstrap_ci_high": high,
        "bootstrap_resamples": indices.shape[0], "bootstrap_seed": contract["bootstrap"]["seed"],
    })
    return output


def runtime_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["fit_scope"], row["operator"], finite_float(row["density"], "runtime density"))].append(row)
    output = []
    for (scope, operator, density), values in sorted(grouped.items()):
        runtimes = np.asarray([finite_float(row["runtime_seconds"], "runtime") for row in values])
        peaks = [finite_int(row["peak_gpu_memory_bytes"], "peak GPU memory") for row in values]
        output.append({
            "fit_scope": scope, "operator": operator, "density": density, "fits": len(values),
            "total_runtime_seconds": float(runtimes.sum()), "mean_runtime_seconds": float(runtimes.mean()),
            "median_runtime_seconds": float(np.median(runtimes)), "maximum_peak_gpu_memory_bytes": max(peaks),
            "resumed_fits": sum(finite_int(row["resumed"], "resumed") for row in values),
        })
    return output


def selected_tuning(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["selected"] == "1"]
    return [{field: row[field] for field in SELECTED_TUNING_FIELDS} for row in selected]


def warning_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, str, str], int] = defaultdict(int)
    for row in rows:
        grouped[(row["operator"], finite_float(row["density"], "warning density"), row["fit_scope"], row["failure_type"])] += 1
    return [
        {"operator": key[0], "density": key[1], "fit_scope": key[2], "failure_type": key[3], "warning_rows": count}
        for key, count in sorted(grouped.items())
    ]


def plot_predictive_curves(
    config_rows: list[dict[str, Any]], baseline: dict[str, float], sites: list[str],
    contract: dict[str, Any], output: Path,
) -> None:
    values = configuration_map(config_rows, "mean_balanced_accuracy")
    densities = [float(value) for value in contract["densities"]]
    colors = {"gcn": "#4C78A8", "trivial_bundle": "#F58518", "learned_bunn": "#54A24B"}
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    for operator, anchor in contract["curve_anchors"].items():
        matrix = np.asarray([
            [values[(site, anchor, 0.0)]] + [values[(site, operator, density)] for density in densities[1:]]
            for site in sites
        ])
        means = matrix.mean(axis=0)
        standard_errors = matrix.std(axis=0, ddof=1) / math.sqrt(len(sites))
        axis.errorbar(densities, means, yerr=1.96 * standard_errors, marker="o", capsize=3,
                      label=operator, color=colors[operator])
    baseline_values = np.asarray([baseline[site] for site in sites])
    axis.axhline(baseline_values.mean(), color="#B279A2", linestyle="--", label="connectome elastic net")
    axis.axhline(0.5, color="#777777", linestyle=":", label="chance BA")
    axis.set(xlabel="Retained positive-edge density", ylabel="Equal-site mean balanced accuracy",
             xticks=densities)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_contrasts(rows: list[dict[str, Any]], output: Path) -> None:
    labels = [row["contrast"].replace("_", " ") for row in rows]
    means = np.asarray([row["observed_mean_difference"] for row in rows], dtype=float)
    lows = np.asarray([row["bootstrap_ci_low"] for row in rows], dtype=float)
    highs = np.asarray([row["bootstrap_ci_high"] for row in rows], dtype=float)
    positions = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(9, 5.2))
    axis.errorbar(means, positions, xerr=np.vstack((means - lows, highs - means)), fmt="o", capsize=4)
    axis.axvline(0.0, color="#777777", linestyle="--")
    axis.set(yticks=positions, yticklabels=labels, xlabel="Paired equal-site balanced-accuracy difference")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_representation(rows: list[dict[str, Any]], sites: list[str], contract: dict[str, Any], output: Path) -> None:
    values = {(row["held_out_site"], row["operator"], float(row["density"]), row["layer"]): row for row in rows}
    densities = [float(value) for value in contract["densities"]]
    colors = {"gcn": "#4C78A8", "trivial_bundle": "#F58518", "learned_bunn": "#54A24B"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, endpoint in zip(axes.flat, ENDPOINTS, strict=True):
        for operator, anchor in contract["curve_anchors"].items():
            matrix = np.asarray([
                [values[(site, anchor, 0.0, "layer_2")][endpoint]]
                + [values[(site, operator, density, "layer_2")][endpoint] for density in densities[1:]]
                for site in sites
            ], dtype=float)
            axis.plot(densities, matrix.mean(axis=0), marker="o", label=operator, color=colors[operator])
        axis.set(title=endpoint.replace("_", " "), xlabel="Density", xticks=densities)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Common-frame/gauge-aware layer-2 representation diagnostics")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def decision_summary(predictive: list[dict[str, Any]], representation: list[dict[str, Any]]) -> dict[str, Any]:
    pred = {row["contrast"]: row for row in predictive}
    rep_primary = next(row for row in representation if int(row["primary"]) == 1)
    representation_condition = float(rep_primary["bootstrap_ci_low"]) > 0.0
    predictive_condition = all(
        float(pred[name]["bootstrap_ci_low"]) > 0.0
        for name in (
            "learned_bunn_curve_minus_gcn_curve",
            "learned_bunn_nonzero_minus_learned_local",
            "learned_bunn_nonzero_minus_identity",
        )
    )
    baseline_condition = float(
        pred["learned_bunn_curve_minus_connectome_elastic_net"]["bootstrap_ci_low"]
    ) > 0.0
    if representation_condition and predictive_condition and baseline_condition:
        category = "positive_architectural_result"
    elif representation_condition and not (predictive_condition and baseline_condition):
        category = "representation_change_without_complete_predictive_advantage"
    else:
        category = "no_complete_transfer_of_proposed_anti_collapse_advantage_detected"
    return {
        "decision_rule_version": "step11_neural_confirmatory_analysis_v1",
        "representation_condition": representation_condition,
        "predictive_condition": predictive_condition,
        "baseline_condition": baseline_condition,
        "all_three_conditions": representation_condition and predictive_condition and baseline_condition,
        "result_category": category,
        "causal_language_allowed": False,
        "biological_geometry_claim_allowed": False,
    }


def analyze_run(
    run_dir: Path, certificate_path: Path, contract_path: Path,
    baseline_path: Path, output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise NeuralAnalysisError(f"Refusing to overwrite analysis directory: {output_dir}")
    contract = load_json(contract_path)
    metadata = validate_gate(run_dir, certificate_path, contract)
    sites = sorted(metadata["held_out_sites"])
    resamples = int(contract["bootstrap"]["resamples"])
    seed = int(contract["bootstrap"]["seed"])
    if resamples != 10000:
        raise NeuralAnalysisError("Confirmatory analysis requires exactly 10,000 bootstrap resamples")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(sites), size=(resamples, len(sites)))
    signs = sign_matrix(len(sites))

    site_seed = compute_site_seed_metrics(
        read_csv(run_dir / "predictions.csv", PREDICTION_FIELDS),
        read_csv(run_dir / "test_metrics.csv", METRIC_FIELDS),
        float(contract["decision_threshold"]),
    )
    configurations = aggregate_configurations(site_seed)
    if {row["held_out_site"] for row in configurations} != set(sites):
        raise NeuralAnalysisError("Neural configuration aggregates do not cover the frozen sites")
    baseline = baseline_map(baseline_path, sites, contract["baseline_reference"])
    curves = build_predictive_curves(configurations, sites, contract)
    primary_contrasts = predictive_contrasts(curves, configurations, baseline, sites, indices, signs, contract)
    secondary_density = density_contrasts(configurations, baseline, sites, indices, signs, contract)
    representation = aggregate_representation(read_csv(run_dir / "diagnostics.csv", DIAGNOSTIC_FIELDS))
    representation_tests = representation_contrasts(representation, sites, indices, contract)
    runtimes = runtime_summary(read_csv(run_dir / "fit_runtime.csv", RUNTIME_FIELDS))
    tuning = selected_tuning(read_csv(run_dir / "tuning_scores.csv", TUNING_FIELDS))
    warnings = warning_summary(read_csv(run_dir / "fit_warnings.csv", WARNING_FIELDS))

    output_dir.mkdir(parents=True)
    write_csv(output_dir / "site_seed_metrics.csv", site_seed, SITE_SEED_FIELDS)
    write_csv(output_dir / "site_configuration_metrics.csv", configurations, SITE_CONFIGURATION_FIELDS)
    write_csv(output_dir / "seed_stability.csv", seed_summaries(site_seed), SEED_SUMMARY_FIELDS)
    write_csv(output_dir / "site_predictive_curves.csv", curves, CURVE_FIELDS)
    write_csv(output_dir / "confirmatory_predictive_contrasts.csv", primary_contrasts, CONTRAST_FIELDS)
    write_csv(output_dir / "density_specific_contrasts.csv", secondary_density, DENSITY_CONTRAST_FIELDS)
    write_csv(output_dir / "site_representation_metrics.csv", representation, REPRESENTATION_FIELDS)
    write_csv(output_dir / "representation_contrasts.csv", representation_tests, REPRESENTATION_CONTRAST_FIELDS)
    write_csv(output_dir / "runtime_summary.csv", runtimes, RUNTIME_SUMMARY_FIELDS)
    write_csv(output_dir / "selected_hyperparameters.csv", tuning, SELECTED_TUNING_FIELDS)
    write_csv(output_dir / "fit_warning_summary.csv", warnings, WARNING_SUMMARY_FIELDS)
    plot_predictive_curves(configurations, baseline, sites, contract, output_dir / "predictive_density_curves.png")
    plot_contrasts(primary_contrasts, output_dir / "confirmatory_predictive_contrasts.png")
    plot_representation(representation, sites, contract, output_dir / "representation_density_curves.png")
    (output_dir / "decision_summary.json").write_text(
        json.dumps(decision_summary(primary_contrasts, representation_tests), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    source_paths = [
        run_dir / "metadata.json", run_dir / "predictions.csv", run_dir / "test_metrics.csv",
        run_dir / "diagnostics.csv", run_dir / "fit_runtime.csv", run_dir / "tuning_scores.csv",
        run_dir / "fit_warnings.csv", certificate_path, contract_path, baseline_path,
    ]
    generated = sorted(path for path in output_dir.iterdir() if path.name != "analysis_manifest.json")
    manifest = {
        "analysis_version": contract["analysis_version"], "run_id": metadata["run_id"],
        "unsealed_once_under_explicit_acknowledgement": True,
        "source_sha256": {str(path): sha256_file(path) for path in source_paths},
        "generated_sha256": {path.name: sha256_file(path) for path in generated},
        "bootstrap": contract["bootstrap"], "site_weighting": contract["site_weighting"],
        "seed_handling": contract["seed_handling"], "generated_files": [path.name for path in generated],
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"run_id": metadata["run_id"], "output_dir": str(output_dir),
            "generated_files": sorted(path.name for path in output_dir.iterdir())}


def main() -> None:
    args = parse_args()
    contract = load_json(args.analysis_contract)
    expected = contract.get("input_run_id")
    if args.confirm_unblind_run_id != expected:
        raise SystemExit(
            "Refusing to read neural results without explicit acknowledgement: "
            "pass --confirm-unblind-run-id with the exact frozen run ID."
        )
    result = analyze_run(
        args.run_dir, args.integrity_certificate, args.analysis_contract,
        args.baseline_per_site, args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
