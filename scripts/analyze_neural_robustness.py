"""Run the frozen Step 12A influence and robustness analysis.

This module consumes only the immutable Step 11 tables. It cannot alter the
confirmatory result, select a favorable excluded site or seed, or trigger new
model training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

BASELINE_SITE_METRIC_FIELDS = [
    "model", "held_out_site", "participants", "asd", "control",
    "balanced_accuracy", "auroc", "sensitivity", "specificity",
]
CURVE_FIELDS = [
    "curve_operator", "anchor_operator", "held_out_site", "normalized_auc_balanced_accuracy",
    "anchor_balanced_accuracy", "mean_nonzero_density_balanced_accuracy", "change_from_anchor",
]
SITE_SEED_FIELDS = [
    "operator", "density", "seed", "held_out_site", "participants", "asd", "control",
    "balanced_accuracy", "auroc", "sensitivity", "specificity", "parameter_count",
]
SITE_CONFIGURATION_FIELDS = [
    "operator", "density", "held_out_site", "seed_count", "participants",
    "mean_balanced_accuracy", "sd_balanced_accuracy", "mean_auroc", "mean_sensitivity",
    "mean_specificity", "parameter_count",
]
REPRESENTATION_FIELDS = [
    "operator", "density", "held_out_site", "layer", "seed_count", "participants_per_seed",
    "normalized_effective_rank", "normalized_dispersion", "mean_pairwise_cosine",
    "invariant_edge_transport_distance",
]
RUNTIME_SUMMARY_FIELDS = [
    "fit_scope", "operator", "density", "fits", "total_runtime_seconds", "mean_runtime_seconds",
    "median_runtime_seconds", "maximum_peak_gpu_memory_bytes", "resumed_fits",
]
LOO_FIELDS = [
    "contrast", "excluded_site", "retained_sites", "full_estimate",
    "leave_one_out_estimate", "change_from_full", "bootstrap_ci_low",
    "bootstrap_ci_high", "exact_sign_flip_p", "bootstrap_resamples", "bootstrap_seed",
]
SEED_CURVE_FIELDS = [
    "seed", "contrast", "held_out_sites", "observed_mean_difference",
    "bootstrap_ci_low", "bootstrap_ci_high", "exact_sign_flip_p",
    "positive_site_differences", "negative_site_differences",
    "bootstrap_resamples", "bootstrap_seed",
]
SEED_RANK_FIELDS = [
    "seed", "rank", "operator", "density", "held_out_sites",
    "unweighted_mean_site_balanced_accuracy",
]
ALTERNATIVE_FIELDS = [
    "contrast", "summary", "held_out_sites", "estimate", "confirmatory_primary",
]
EXHAUSTIVE_FIELDS = [
    "contrast", "held_out_sites", "observed_mean_difference", "bootstrap_ci_low",
    "bootstrap_ci_high", "exact_sign_flip_p", "bootstrap_resamples", "bootstrap_seed",
]
EFFICIENCY_FIELDS = [
    "operator", "configuration_count", "parameter_count", "fits", "total_runtime_hours",
    "mean_runtime_seconds", "maximum_peak_gpu_memory_gib",
    "unweighted_mean_site_curve_balanced_accuracy", "curve_minus_elastic_net",
]


class NeuralAnalysisError(ValueError):
    """Raised when Step 11 artifacts cannot support the frozen robustness analysis."""


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
        for left_x, right_x, left_y, right_y in zip(densities, densities[1:], values, values[1:])
    )
    return float(area / (densities[-1] - densities[0]))


def bootstrap_interval(differences: np.ndarray, indices: np.ndarray) -> tuple[float, float, float]:
    if differences.ndim != 1 or differences.size != indices.shape[1]:
        raise NeuralAnalysisError("Bootstrap differences are not site-aligned")
    means = differences[indices].mean(axis=1)
    return float(differences.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step11-dir", required=True, type=Path)
    parser.add_argument("--baseline-per-site", required=True, type=Path)
    parser.add_argument(
        "--contract", type=Path, default=Path("configs/neural_robustness_analysis_v1.json")
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--confirm-step11-run-id", default=None)
    return parser.parse_args()


def validate_inputs(
    step11_dir: Path, baseline_path: Path, contract: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    manifest_path = step11_dir / "analysis_manifest.json"
    if sha256_file(manifest_path) != contract["input_step11_manifest_sha256"]:
        raise NeuralAnalysisError("Step 11 manifest hash differs from the frozen robustness input")
    if sha256_file(baseline_path) != contract["input_baseline_per_site_sha256"]:
        raise NeuralAnalysisError("Baseline table hash differs from the frozen robustness input")
    manifest = load_json(manifest_path)
    if manifest.get("run_id") != contract["input_run_id"]:
        raise NeuralAnalysisError("Step 11 run ID differs from the robustness contract")
    required = [
        "site_predictive_curves.csv", "site_seed_metrics.csv",
        "site_configuration_metrics.csv", "site_representation_metrics.csv",
        "runtime_summary.csv", "decision_summary.json",
    ]
    generated_hashes = manifest.get("generated_sha256", {})
    for name in required:
        expected = generated_hashes.get(name)
        if not isinstance(expected, str) or sha256_file(step11_dir / name) != expected:
            raise NeuralAnalysisError(f"Step 11 generated artifact hash mismatch: {name}")
    decision = load_json(step11_dir / "decision_summary.json")
    if decision.get("all_three_conditions") is not False:
        raise NeuralAnalysisError("Robustness contract expects the archived negative/null Step 11 decision")
    curve_rows = read_csv(step11_dir / "site_predictive_curves.csv", CURVE_FIELDS)
    sites = sorted({row["held_out_site"] for row in curve_rows})
    if len(sites) != 18:
        raise NeuralAnalysisError("Robustness analysis requires all 18 held-out sites")
    return manifest, sites


def baseline_values(path: Path, sites: list[str]) -> tuple[dict[str, float], dict[str, int]]:
    rows = read_csv(path, BASELINE_SITE_METRIC_FIELDS)
    selected = [row for row in rows if row["model"] == "connectome_elastic_net_logistic"]
    values = {row["held_out_site"]: finite_float(row["balanced_accuracy"], "baseline BA") for row in selected}
    counts = {row["held_out_site"]: finite_int(row["participants"], "baseline participants") for row in selected}
    if set(values) != set(sites) or len(selected) != len(sites):
        raise NeuralAnalysisError("Connectome baseline does not cover the same sites exactly once")
    return values, counts


def curve_maps(rows: list[dict[str, str]]) -> tuple[dict[tuple[str, str], float], dict[str, float], dict[str, float]]:
    curves: dict[tuple[str, str], float] = {}
    identity: dict[str, float] = {}
    learned_local: dict[str, float] = {}
    for row in rows:
        site = row["held_out_site"]
        operator = row["curve_operator"]
        key = (site, operator)
        if key in curves:
            raise NeuralAnalysisError(f"Duplicate Step 11 curve: {key}")
        curves[key] = finite_float(row["normalized_auc_balanced_accuracy"], "curve area")
        anchor = row["anchor_operator"]
        anchor_value = finite_float(row["anchor_balanced_accuracy"], "anchor BA")
        if anchor == "identity":
            if site in identity and not math.isclose(identity[site], anchor_value, abs_tol=1e-12):
                raise NeuralAnalysisError("Identity anchor differs between curve rows")
            identity[site] = anchor_value
        elif anchor == "learned_local":
            learned_local[site] = anchor_value
    sites = {site for site, _operator in curves}
    expected = {(site, operator) for site in sites for operator in ("gcn", "trivial_bundle", "learned_bunn")}
    if set(curves) != expected or set(identity) != sites or set(learned_local) != sites:
        raise NeuralAnalysisError("Step 11 curve/anchor coverage is incomplete")
    return curves, identity, learned_local


def primary_representation_site_differences(
    rows: list[dict[str, str]], sites: list[str], contract: dict[str, Any]
) -> dict[str, float]:
    values = {
        (row["held_out_site"], row["operator"], finite_float(row["density"], "density"), row["layer"]):
        finite_float(row["normalized_effective_rank"], "normalized effective rank")
        for row in rows
    }
    densities = [float(value) for value in contract["densities"]]

    def change(site: str, operator: str, anchor: str) -> float:
        anchor_value = values[(site, anchor, 0.0, "layer_2")]
        curve = [anchor_value] + [values[(site, operator, density, "layer_2")] for density in densities[1:]]
        return normalized_auc(densities, curve) - anchor_value

    return {
        site: change(site, "learned_bunn", "learned_local") - change(site, "gcn", "identity")
        for site in sites
    }


def core_site_differences(
    curve_rows: list[dict[str, str]], representation_rows: list[dict[str, str]],
    baseline: dict[str, float], sites: list[str], contract: dict[str, Any],
) -> dict[str, dict[str, float]]:
    curves, _identity, _learned_local = curve_maps(curve_rows)
    representation = primary_representation_site_differences(representation_rows, sites, contract)
    return {
        "learned_bunn_curve_minus_gcn_curve": {
            site: curves[(site, "learned_bunn")] - curves[(site, "gcn")] for site in sites
        },
        "learned_bunn_curve_minus_connectome_elastic_net": {
            site: curves[(site, "learned_bunn")] - baseline[site] for site in sites
        },
        "learned_bunn_minus_gcn_matched_anchor_effective_rank_change": representation,
    }


def leave_one_site_out(
    differences: dict[str, dict[str, float]], sites: list[str], contract: dict[str, Any]
) -> list[dict[str, Any]]:
    resamples = int(contract["bootstrap"]["resamples"])
    seed = int(contract["bootstrap"]["seed"])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(sites) - 1, size=(resamples, len(sites) - 1))
    signs = sign_matrix(len(sites) - 1)
    output = []
    for contrast, by_site in differences.items():
        full = float(np.mean([by_site[site] for site in sites]))
        for excluded in sites:
            retained = [site for site in sites if site != excluded]
            values = np.asarray([by_site[site] for site in retained], dtype=float)
            observed, low, high = bootstrap_interval(values, indices)
            output.append({
                "contrast": contrast, "excluded_site": excluded, "retained_sites": len(retained),
                "full_estimate": full, "leave_one_out_estimate": observed,
                "change_from_full": observed - full, "bootstrap_ci_low": low,
                "bootstrap_ci_high": high, "exact_sign_flip_p": exact_sign_flip_p(values, signs),
                "bootstrap_resamples": resamples, "bootstrap_seed": seed,
            })
    return output


def seed_specific_curves(
    rows: list[dict[str, str]], baseline: dict[str, float], sites: list[str], contract: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    values: dict[tuple[str, str, float, int], float] = {}
    for row in rows:
        key = (
            row["held_out_site"], row["operator"], finite_float(row["density"], "seed density"),
            finite_int(row["seed"], "seed"),
        )
        if key in values:
            raise NeuralAnalysisError(f"Duplicate site/configuration/seed metric: {key}")
        values[key] = finite_float(row["balanced_accuracy"], "seed BA")
    seeds = sorted({key[3] for key in values})
    if len(seeds) != 5:
        raise NeuralAnalysisError("Expected exactly five final seeds")
    densities = [float(value) for value in contract["densities"]]
    resamples = int(contract["bootstrap"]["resamples"])
    seed_value = int(contract["bootstrap"]["seed"])
    rng = np.random.default_rng(seed_value)
    indices = rng.integers(0, len(sites), size=(resamples, len(sites)))
    signs = sign_matrix(len(sites))
    curve_output = []
    ranking_output = []
    for seed in seeds:
        site_curves: dict[tuple[str, str], float] = {}
        for site in sites:
            for operator, anchor in contract["curve_anchors"].items():
                curve = [values[(site, anchor, 0.0, seed)]] + [
                    values[(site, operator, density, seed)] for density in densities[1:]
                ]
                site_curves[(site, operator)] = normalized_auc(densities, curve)
        contrasts = {
            "learned_bunn_curve_minus_gcn_curve": np.asarray([
                site_curves[(site, "learned_bunn")] - site_curves[(site, "gcn")] for site in sites
            ]),
            "learned_bunn_curve_minus_connectome_elastic_net": np.asarray([
                site_curves[(site, "learned_bunn")] - baseline[site] for site in sites
            ]),
        }
        for contrast, differences in contrasts.items():
            observed, low, high = bootstrap_interval(differences, indices)
            curve_output.append({
                "seed": seed, "contrast": contrast, "held_out_sites": len(sites),
                "observed_mean_difference": observed, "bootstrap_ci_low": low,
                "bootstrap_ci_high": high, "exact_sign_flip_p": exact_sign_flip_p(differences, signs),
                "positive_site_differences": int((differences > 0).sum()),
                "negative_site_differences": int((differences < 0).sum()),
                "bootstrap_resamples": resamples, "bootstrap_seed": seed_value,
            })
        configuration_means = []
        configurations = sorted({(key[1], key[2]) for key in values if key[3] == seed})
        for operator, density in configurations:
            configuration_means.append((
                float(np.mean([values[(site, operator, density, seed)] for site in sites])),
                operator, density,
            ))
        for rank, (mean_value, operator, density) in enumerate(
            sorted(configuration_means, key=lambda item: (-item[0], item[1], item[2])), start=1
        ):
            ranking_output.append({
                "seed": seed, "rank": rank, "operator": operator, "density": density,
                "held_out_sites": len(sites), "unweighted_mean_site_balanced_accuracy": mean_value,
            })
    return curve_output, ranking_output


def alternative_summaries(
    differences: dict[str, dict[str, float]], counts: dict[str, int], sites: list[str]
) -> list[dict[str, Any]]:
    weights = np.asarray([counts[site] for site in sites], dtype=float)
    output = []
    for contrast, by_site in differences.items():
        values = np.asarray([by_site[site] for site in sites], dtype=float)
        summaries = {
            "equal_site_mean": float(values.mean()),
            "participant_weighted_mean": float(np.average(values, weights=weights)),
            "median_site_difference": float(np.median(values)),
        }
        for name, estimate in summaries.items():
            output.append({
                "contrast": contrast, "summary": name, "held_out_sites": len(sites),
                "estimate": estimate, "confirmatory_primary": int(name == "equal_site_mean"),
            })
    return output


def exhaustive_curve_contrasts(
    curve_rows: list[dict[str, str]], baseline: dict[str, float], sites: list[str],
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    curves, identity, learned_local = curve_maps(curve_rows)
    arrays = {
        "gcn_curve_minus_identity": np.asarray([curves[(site, "gcn")] - identity[site] for site in sites]),
        "trivial_bundle_curve_minus_identity": np.asarray([curves[(site, "trivial_bundle")] - identity[site] for site in sites]),
        "learned_bunn_curve_minus_learned_local": np.asarray([curves[(site, "learned_bunn")] - learned_local[site] for site in sites]),
        "learned_bunn_curve_minus_identity": np.asarray([curves[(site, "learned_bunn")] - identity[site] for site in sites]),
        "learned_bunn_curve_minus_gcn_curve": np.asarray([curves[(site, "learned_bunn")] - curves[(site, "gcn")] for site in sites]),
        "learned_bunn_curve_minus_trivial_bundle_curve": np.asarray([curves[(site, "learned_bunn")] - curves[(site, "trivial_bundle")] for site in sites]),
        "gcn_curve_minus_connectome_elastic_net": np.asarray([curves[(site, "gcn")] - baseline[site] for site in sites]),
        "trivial_bundle_curve_minus_connectome_elastic_net": np.asarray([curves[(site, "trivial_bundle")] - baseline[site] for site in sites]),
        "learned_bunn_curve_minus_connectome_elastic_net": np.asarray([curves[(site, "learned_bunn")] - baseline[site] for site in sites]),
    }
    if list(arrays) != contract["exhaustive_curve_contrasts"]:
        raise NeuralAnalysisError("Implemented exhaustive contrast order differs from the analysis config")
    resamples = int(contract["bootstrap"]["resamples"])
    seed = int(contract["bootstrap"]["seed"])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(sites), size=(resamples, len(sites)))
    signs = sign_matrix(len(sites))
    output = []
    for contrast, differences in arrays.items():
        observed, low, high = bootstrap_interval(differences, indices)
        output.append({
            "contrast": contrast, "held_out_sites": len(sites),
            "observed_mean_difference": observed, "bootstrap_ci_low": low,
            "bootstrap_ci_high": high, "exact_sign_flip_p": exact_sign_flip_p(differences, signs),
            "bootstrap_resamples": resamples, "bootstrap_seed": seed,
        })
    return output


def efficiency_summary(
    runtime_rows: list[dict[str, str]], configuration_rows: list[dict[str, str]],
    curve_rows: list[dict[str, str]], baseline: dict[str, float], sites: list[str],
) -> list[dict[str, Any]]:
    runtime: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in runtime_rows:
        runtime[row["operator"]].append(row)
    parameters: dict[str, set[int]] = defaultdict(set)
    configurations: dict[str, set[float]] = defaultdict(set)
    for row in configuration_rows:
        operator = row["operator"]
        parameters[operator].add(finite_int(row["parameter_count"], "parameter count"))
        configurations[operator].add(finite_float(row["density"], "configuration density"))
    curves, _identity, _learned_local = curve_maps(curve_rows)
    baseline_mean = float(np.mean([baseline[site] for site in sites]))
    output = []
    for operator in ("gcn", "trivial_bundle", "learned_bunn"):
        rows = runtime[operator]
        fits = sum(finite_int(row["fits"], "fits") for row in rows)
        total_seconds = sum(finite_float(row["total_runtime_seconds"], "total runtime") for row in rows)
        weighted_mean_runtime = total_seconds / fits
        parameter_values = parameters[operator]
        if len(parameter_values) != 1:
            raise NeuralAnalysisError(f"Parameter count varies within operator {operator}")
        curve_mean = float(np.mean([curves[(site, operator)] for site in sites]))
        output.append({
            "operator": operator, "configuration_count": len(configurations[operator]),
            "parameter_count": next(iter(parameter_values)), "fits": fits,
            "total_runtime_hours": total_seconds / 3600.0,
            "mean_runtime_seconds": weighted_mean_runtime,
            "maximum_peak_gpu_memory_gib": max(
                finite_float(row["maximum_peak_gpu_memory_bytes"], "peak GPU") for row in rows
            ) / (1024.0 ** 3),
            "unweighted_mean_site_curve_balanced_accuracy": curve_mean,
            "curve_minus_elastic_net": curve_mean - baseline_mean,
        })
    return output


def robustness_decision(
    loo_rows: list[dict[str, Any]], seed_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    primary = "learned_bunn_curve_minus_gcn_curve"
    loo = [row for row in loo_rows if row["contrast"] == primary]
    seeds = [row for row in seed_rows if row["contrast"] == primary]
    loo_estimates = [float(row["leave_one_out_estimate"]) for row in loo]
    seed_estimates = [float(row["observed_mean_difference"]) for row in seeds]
    site_sensitive = min(loo_estimates) < 0.0 < max(loo_estimates)
    seed_sensitive = min(seed_estimates) < 0.0 < max(seed_estimates)
    if site_sensitive and seed_sensitive:
        category = "mixed_site_and_seed_sensitive_null"
    elif site_sensitive:
        category = "site_sensitive_null"
    elif seed_sensitive:
        category = "seed_sensitive_null"
    else:
        category = "directionally_robust_negative_null"
    return {
        "analysis_version": "step12_neural_robustness_analysis_v1",
        "confirmatory_step11_result_unchanged": True,
        "positive_override_allowed": False,
        "site_sensitive": site_sensitive,
        "seed_sensitive": seed_sensitive,
        "robustness_category": category,
        "leave_one_site_estimates_positive": sum(value > 0 for value in loo_estimates),
        "leave_one_site_intervals_excluding_zero_positive": sum(float(row["bootstrap_ci_low"]) > 0 for row in loo),
        "seed_estimates_positive": sum(value > 0 for value in seed_estimates),
        "seed_intervals_excluding_zero_positive": sum(float(row["bootstrap_ci_low"]) > 0 for row in seeds),
    }


def plot_site_influence(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["contrast"] == "learned_bunn_curve_minus_gcn_curve"]
    selected.sort(key=lambda row: float(row["leave_one_out_estimate"]))
    means = np.asarray([float(row["leave_one_out_estimate"]) for row in selected])
    lows = np.asarray([float(row["bootstrap_ci_low"]) for row in selected])
    highs = np.asarray([float(row["bootstrap_ci_high"]) for row in selected])
    positions = np.arange(len(selected))
    fig, axis = plt.subplots(figsize=(9, 7))
    axis.errorbar(means, positions, xerr=np.vstack((means - lows, highs - means)), fmt="o", capsize=3)
    axis.axvline(0.0, color="#777777", linestyle="--", label="zero difference")
    axis.axvline(float(selected[0]["full_estimate"]), color="#E45756", linestyle=":", label="full 18-site estimate")
    axis.set(yticks=positions, yticklabels=[f"exclude {row['excluded_site']}" for row in selected],
             xlabel="BuNN minus GCN normalized curve-area difference")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_seed_stability(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(9, 4.8))
    colors = {
        "learned_bunn_curve_minus_gcn_curve": "#4C78A8",
        "learned_bunn_curve_minus_connectome_elastic_net": "#E45756",
    }
    offsets = {"learned_bunn_curve_minus_gcn_curve": -0.08,
               "learned_bunn_curve_minus_connectome_elastic_net": 0.08}
    seeds = sorted({int(row["seed"]) for row in rows})
    for contrast in colors:
        selected = sorted([row for row in rows if row["contrast"] == contrast], key=lambda row: int(row["seed"]))
        means = np.asarray([float(row["observed_mean_difference"]) for row in selected])
        lows = np.asarray([float(row["bootstrap_ci_low"]) for row in selected])
        highs = np.asarray([float(row["bootstrap_ci_high"]) for row in selected])
        positions = np.arange(len(seeds)) + offsets[contrast]
        axis.errorbar(positions, means, yerr=np.vstack((means - lows, highs - means)), fmt="o", capsize=3,
                      color=colors[contrast], label=contrast.replace("_", " "))
    axis.axhline(0.0, color="#777777", linestyle="--")
    axis.set(xticks=np.arange(len(seeds)), xticklabels=[str(seed) for seed in seeds],
             xlabel="Final training seed", ylabel="Paired equal-site curve difference")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def analyze_robustness(
    step11_dir: Path, baseline_path: Path, contract_path: Path, output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists():
        raise NeuralAnalysisError(f"Refusing to overwrite robustness directory: {output_dir}")
    contract = load_json(contract_path)
    manifest, sites = validate_inputs(step11_dir, baseline_path, contract)
    baseline, counts = baseline_values(baseline_path, sites)
    curve_rows = read_csv(step11_dir / "site_predictive_curves.csv", CURVE_FIELDS)
    seed_rows = read_csv(step11_dir / "site_seed_metrics.csv", SITE_SEED_FIELDS)
    configuration_rows = read_csv(step11_dir / "site_configuration_metrics.csv", SITE_CONFIGURATION_FIELDS)
    representation_rows = read_csv(step11_dir / "site_representation_metrics.csv", REPRESENTATION_FIELDS)
    runtime_rows = read_csv(step11_dir / "runtime_summary.csv", RUNTIME_SUMMARY_FIELDS)

    differences = core_site_differences(curve_rows, representation_rows, baseline, sites, contract)
    loo = leave_one_site_out(differences, sites, contract)
    seed_curves, seed_rankings = seed_specific_curves(seed_rows, baseline, sites, contract)
    alternatives = alternative_summaries(differences, counts, sites)
    exhaustive = exhaustive_curve_contrasts(curve_rows, baseline, sites, contract)
    efficiency = efficiency_summary(runtime_rows, configuration_rows, curve_rows, baseline, sites)
    decision = robustness_decision(loo, seed_curves)

    output_dir.mkdir(parents=True)
    write_csv(output_dir / "leave_one_site_out.csv", loo, LOO_FIELDS)
    write_csv(output_dir / "seed_specific_curves.csv", seed_curves, SEED_CURVE_FIELDS)
    write_csv(output_dir / "seed_configuration_rankings.csv", seed_rankings, SEED_RANK_FIELDS)
    write_csv(output_dir / "alternative_summaries.csv", alternatives, ALTERNATIVE_FIELDS)
    write_csv(output_dir / "exhaustive_curve_contrasts.csv", exhaustive, EXHAUSTIVE_FIELDS)
    write_csv(output_dir / "operator_efficiency.csv", efficiency, EFFICIENCY_FIELDS)
    plot_site_influence(loo, output_dir / "site_influence.png")
    plot_seed_stability(seed_curves, output_dir / "seed_stability.png")
    (output_dir / "robustness_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    generated = sorted(path for path in output_dir.iterdir() if path.name != "analysis_manifest.json")
    output_manifest = {
        "analysis_version": contract["analysis_version"], "run_id": manifest["run_id"],
        "confirmatory_step11_result_immutable": True,
        "source_sha256": {
            str(step11_dir / "analysis_manifest.json"): sha256_file(step11_dir / "analysis_manifest.json"),
            str(baseline_path): sha256_file(baseline_path), str(contract_path): sha256_file(contract_path),
        },
        "generated_sha256": {path.name: sha256_file(path) for path in generated},
        "generated_files": [path.name for path in generated],
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"run_id": manifest["run_id"], "output_dir": str(output_dir),
            "generated_files": sorted(path.name for path in output_dir.iterdir())}


def main() -> None:
    args = parse_args()
    contract = load_json(args.contract)
    if args.confirm_step11_run_id != contract.get("input_run_id"):
        raise SystemExit(
            "Refusing robustness execution without the exact archived Step 11 run ID."
        )
    print(json.dumps(analyze_robustness(
        args.step11_dir, args.baseline_per_site, args.contract, args.output_dir
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
