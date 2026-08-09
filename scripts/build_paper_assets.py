"""Build deterministic Step 13 paper assets from frozen audited analyses.

The builder performs presentation and traceability work only. It validates all
accepted input digests before writing, does not fit models, and does not create
new hypothesis tests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class PaperAssetError(ValueError):
    """Raised when the frozen paper evidence cannot be validated."""


BASELINE_FIELDS = [
    "model", "held_out_sites", "participants",
    "unweighted_mean_site_balanced_accuracy", "pooled_balanced_accuracy",
    "pooled_auroc", "pooled_sensitivity", "pooled_specificity",
]
PREDICTIVE_FIELDS = [
    "contrast", "estimand", "primary", "held_out_sites",
    "observed_mean_difference", "bootstrap_ci_low", "bootstrap_ci_high",
    "bootstrap_resamples", "bootstrap_seed", "exact_sign_flip_p",
    "practical_margin", "upper_excludes_practical_margin",
]
REPRESENTATION_FIELDS = [
    "endpoint", "layer", "estimand", "primary", "held_out_sites",
    "observed_mean_difference", "bootstrap_ci_low", "bootstrap_ci_high",
    "bootstrap_resamples", "bootstrap_seed",
]
EFFICIENCY_FIELDS = [
    "operator", "configuration_count", "parameter_count", "fits",
    "total_runtime_hours", "mean_runtime_seconds", "maximum_peak_gpu_memory_gib",
    "unweighted_mean_site_curve_balanced_accuracy", "curve_minus_elastic_net",
]
ALTERNATIVE_FIELDS = [
    "contrast", "summary", "held_out_sites", "estimate", "confirmatory_primary",
]
LOO_FIELDS = [
    "contrast", "excluded_site", "retained_sites", "full_estimate",
    "leave_one_out_estimate", "change_from_full", "bootstrap_ci_low",
    "bootstrap_ci_high", "exact_sign_flip_p", "bootstrap_resamples", "bootstrap_seed",
]
SEED_FIELDS = [
    "seed", "contrast", "held_out_sites", "observed_mean_difference",
    "bootstrap_ci_low", "bootstrap_ci_high", "exact_sign_flip_p",
    "positive_site_differences", "negative_site_differences",
    "bootstrap_resamples", "bootstrap_seed",
]
BASELINE_SITE_FIELDS = [
    "model", "held_out_site", "participants", "asd", "control",
    "balanced_accuracy", "auroc", "sensitivity", "specificity",
]
SITE_CONFIGURATION_FIELDS = [
    "operator", "density", "held_out_site", "seed_count", "participants",
    "mean_balanced_accuracy", "sd_balanced_accuracy", "mean_auroc",
    "mean_sensitivity", "mean_specificity", "parameter_count",
]
SITE_REPRESENTATION_FIELDS = [
    "operator", "density", "held_out_site", "layer", "seed_count",
    "participants_per_seed", "normalized_effective_rank", "normalized_dispersion",
    "mean_pairwise_cosine", "invariant_edge_transport_distance",
]

MODEL_LABELS = {
    "covariates_l2_logistic": "Covariates-only logistic regression",
    "connectome_elastic_net_logistic": "Connectome elastic net",
    "combined_elastic_net_logistic": "Connectome + covariates elastic net",
}
OPERATOR_LABELS = {
    "gcn": "GCN",
    "trivial_bundle": "Trivial-bundle diffusion",
    "learned_bunn": "Learned BuNN",
}
COLORS = {"gcn": "#35689B", "trivial_bundle": "#E17C05", "learned_bunn": "#3B8F3B"}
MARKERS = {"gcn": "o", "trivial_bundle": "s", "learned_bunn": "^"}
LINESTYLES = {"gcn": "-", "trivial_bundle": "--", "learned_bunn": "-."}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PaperAssetError(f"Unreadable JSON: {path}") from error
    if not isinstance(payload, dict):
        raise PaperAssetError(f"Expected a JSON object: {path}")
    return payload


def read_csv(path: Path, expected_fields: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise PaperAssetError(f"Missing required CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_fields:
            raise PaperAssetError(
                f"Schema mismatch for {path.name}: {reader.fieldnames!r}"
            )
        rows = list(reader)
    if not rows:
        raise PaperAssetError(f"Empty required CSV: {path}")
    return rows


def finite_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise PaperAssetError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(parsed):
        raise PaperAssetError(f"{label} is not finite")
    return parsed


def finite_int(value: Any, label: str) -> int:
    parsed = finite_float(value, label)
    if not parsed.is_integer():
        raise PaperAssetError(f"{label} is not an integer: {value!r}")
    return int(parsed)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, payload: dict[str, Any]) -> None:
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write_bytes(path, content)


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def validate_frozen_inputs(repo_root: Path, contract: dict[str, Any]) -> list[dict[str, Any]]:
    frozen = contract.get("frozen_inputs")
    if not isinstance(frozen, dict) or not frozen:
        raise PaperAssetError("Contract has no frozen_inputs mapping")
    records: list[dict[str, Any]] = []
    for relative, expected in sorted(frozen.items()):
        path = repo_root / relative
        if not path.is_file():
            raise PaperAssetError(f"Missing frozen input: {relative}")
        observed = sha256_file(path)
        if observed != expected:
            raise PaperAssetError(
                f"Frozen input digest mismatch for {relative}: {observed} != {expected}"
            )
        records.append({
            "path": relative.replace("\\", "/"),
            "sha256": observed,
            "bytes": path.stat().st_size,
        })
    return records


def validate_analysis_manifest(directory: Path) -> None:
    manifest = load_json(directory / "analysis_manifest.json")
    generated = manifest.get("generated_sha256")
    if generated is None:
        return
    if not isinstance(generated, dict):
        raise PaperAssetError(f"Invalid generated_sha256 in {directory}")
    for name, expected in sorted(generated.items()):
        path = directory / name
        if not path.is_file() or sha256_file(path) != expected:
            raise PaperAssetError(f"Analysis manifest validation failed: {path}")


def keyed(rows: list[dict[str, str]], field: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row[field]
        if key in result:
            raise PaperAssetError(f"Duplicate {field}: {key}")
        result[key] = row
    return result


def numeric_predictive(row: dict[str, str]) -> dict[str, Any]:
    return {
        "estimand": row["estimand"],
        "primary": bool(finite_int(row["primary"], "primary")),
        "held_out_sites": finite_int(row["held_out_sites"], "held_out_sites"),
        "estimate": finite_float(row["observed_mean_difference"], "estimate"),
        "bootstrap_ci_95": [
            finite_float(row["bootstrap_ci_low"], "bootstrap_ci_low"),
            finite_float(row["bootstrap_ci_high"], "bootstrap_ci_high"),
        ],
        "exact_sign_flip_p": finite_float(row["exact_sign_flip_p"], "exact_sign_flip_p"),
        "practical_margin": finite_float(row["practical_margin"], "practical_margin"),
        "upper_excludes_practical_margin": bool(
            finite_int(row["upper_excludes_practical_margin"], "upper_excludes_practical_margin")
        ),
    }


def build_result_snapshot(repo_root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    baseline_dir = repo_root / "outputs/analysis/step7_6_full_baselines_v2"
    step11_dir = repo_root / "outputs/analysis/step11_neural_full_parallel_v1"
    step12_dir = repo_root / "outputs/analysis/step12_neural_robustness_v1"

    cohort = load_json(repo_root / "configs/abide_i_analysis_manifest.json")
    baseline_rows = read_csv(baseline_dir / "model_summary.csv", BASELINE_FIELDS)
    predictive_rows = read_csv(
        step11_dir / "confirmatory_predictive_contrasts.csv", PREDICTIVE_FIELDS
    )
    representation_rows = read_csv(
        step11_dir / "representation_contrasts.csv", REPRESENTATION_FIELDS
    )
    efficiency_rows = read_csv(step12_dir / "operator_efficiency.csv", EFFICIENCY_FIELDS)
    alternative_rows = read_csv(step12_dir / "alternative_summaries.csv", ALTERNATIVE_FIELDS)
    loo_rows = read_csv(step12_dir / "leave_one_site_out.csv", LOO_FIELDS)
    seed_rows = read_csv(step12_dir / "seed_specific_curves.csv", SEED_FIELDS)
    step11_decision = load_json(step11_dir / "decision_summary.json")
    step12_decision = load_json(step12_dir / "robustness_decision.json")

    baselines: dict[str, Any] = {}
    for row in baseline_rows:
        model = row["model"]
        if model not in MODEL_LABELS:
            raise PaperAssetError(f"Unexpected baseline model: {model}")
        baselines[model] = {
            "label": MODEL_LABELS[model],
            "held_out_sites": finite_int(row["held_out_sites"], "held_out_sites"),
            "participants": finite_int(row["participants"], "participants"),
            "equal_site_balanced_accuracy": finite_float(
                row["unweighted_mean_site_balanced_accuracy"], "equal_site_balanced_accuracy"
            ),
            "pooled_balanced_accuracy": finite_float(
                row["pooled_balanced_accuracy"], "pooled_balanced_accuracy"
            ),
            "pooled_auroc": finite_float(row["pooled_auroc"], "pooled_auroc"),
        }

    predictive = {
        name: numeric_predictive(row)
        for name, row in keyed(predictive_rows, "contrast").items()
    }
    required_predictive = {
        "learned_bunn_curve_minus_gcn_curve",
        "learned_bunn_curve_minus_trivial_bundle_curve",
        "learned_bunn_nonzero_minus_learned_local",
        "learned_bunn_nonzero_minus_identity",
        "learned_bunn_curve_minus_connectome_elastic_net",
    }
    if set(predictive) != required_predictive:
        raise PaperAssetError("Unexpected confirmatory predictive contrast set")

    primary_representation = [row for row in representation_rows if row["primary"] == "1"]
    if len(primary_representation) != 1:
        raise PaperAssetError("Expected exactly one primary representation contrast")
    representation_row = primary_representation[0]
    representation = {
        "endpoint": representation_row["endpoint"],
        "layer": representation_row["layer"],
        "estimand": representation_row["estimand"],
        "held_out_sites": finite_int(representation_row["held_out_sites"], "held_out_sites"),
        "estimate": finite_float(representation_row["observed_mean_difference"], "estimate"),
        "bootstrap_ci_95": [
            finite_float(representation_row["bootstrap_ci_low"], "bootstrap_ci_low"),
            finite_float(representation_row["bootstrap_ci_high"], "bootstrap_ci_high"),
        ],
    }

    efficiency: dict[str, Any] = {}
    for row in efficiency_rows:
        operator = row["operator"]
        if operator not in OPERATOR_LABELS:
            raise PaperAssetError(f"Unexpected efficiency operator: {operator}")
        efficiency[operator] = {
            "label": OPERATOR_LABELS[operator],
            "parameter_count": finite_int(row["parameter_count"], "parameter_count"),
            "fits": finite_int(row["fits"], "fits"),
            "total_runtime_hours": finite_float(row["total_runtime_hours"], "total_runtime_hours"),
            "mean_runtime_seconds": finite_float(row["mean_runtime_seconds"], "mean_runtime_seconds"),
            "maximum_peak_gpu_memory_gib": finite_float(
                row["maximum_peak_gpu_memory_gib"], "maximum_peak_gpu_memory_gib"
            ),
            "equal_site_curve_balanced_accuracy": finite_float(
                row["unweighted_mean_site_curve_balanced_accuracy"],
                "equal_site_curve_balanced_accuracy",
            ),
        }

    target = "learned_bunn_curve_minus_gcn_curve"
    loo_target = [row for row in loo_rows if row["contrast"] == target]
    seed_target = [row for row in seed_rows if row["contrast"] == target]
    if len(loo_target) != 18 or len(seed_target) != 5:
        raise PaperAssetError("Unexpected robustness row count for the primary contrast")

    alternative = [
        {
            "contrast": row["contrast"],
            "summary": row["summary"],
            "estimate": finite_float(row["estimate"], "estimate"),
            "confirmatory_primary": bool(
                finite_int(row["confirmatory_primary"], "confirmatory_primary")
            ),
        }
        for row in alternative_rows
    ]

    snapshot = {
        "schema_version": "step13_result_snapshot_v1",
        "contract_version": contract["contract_version"],
        "evidence_commit": contract["evidence_commit"],
        "run_id": contract["run_id"],
        "cohort": {
            "parent_rows": finite_int(cohort["parent_rows"], "parent_rows"),
            "technical_exclusions": finite_int(
                cohort["technical_exclusions"], "technical_exclusions"
            ),
            "participants": finite_int(cohort["retained_rows"], "retained_rows"),
            "held_out_sites": finite_int(cohort["site_count"], "site_count"),
            "asd": finite_int(cohort["class_counts"]["asd"], "asd"),
            "control": finite_int(cohort["class_counts"]["control"], "control"),
            "roi_count": finite_int(cohort["connectome_artifact"]["array_shape"][1], "roi_count"),
            "edge_features": finite_int(
                cohort["connectome_artifact"]["edge_feature_shape"][1], "edge_features"
            ),
        },
        "classical_baselines": baselines,
        "confirmatory_predictive_contrasts": predictive,
        "primary_representation_contrast": representation,
        "step11_decision": step11_decision,
        "robustness": {
            "decision": step12_decision,
            "primary_leave_one_site_out": {
                "rows": len(loo_target),
                "positive_estimates": sum(
                    finite_float(row["leave_one_out_estimate"], "leave_one_out_estimate") > 0
                    for row in loo_target
                ),
                "positive_intervals_excluding_zero": sum(
                    finite_float(row["bootstrap_ci_low"], "bootstrap_ci_low") > 0
                    for row in loo_target
                ),
            },
            "primary_seed_specific": {
                "rows": len(seed_target),
                "positive_estimates": sum(
                    finite_float(row["observed_mean_difference"], "observed_mean_difference") > 0
                    for row in seed_target
                ),
                "positive_intervals_excluding_zero": sum(
                    finite_float(row["bootstrap_ci_low"], "bootstrap_ci_low") > 0
                    for row in seed_target
                ),
                "negative_intervals_excluding_zero": sum(
                    finite_float(row["bootstrap_ci_high"], "bootstrap_ci_high") < 0
                    for row in seed_target
                ),
            },
            "alternative_summaries": alternative,
        },
        "operator_efficiency": efficiency,
        "claim_boundaries": {
            "biological_geometry_claim_allowed": False,
            "causal_language_allowed": False,
            "general_bunn_inferiority_claim_allowed": False,
            "clinical_diagnostic_claim_allowed": False,
            "conditional_abide_pipeline_language_required": True,
        },
    }

    if snapshot["cohort"]["participants"] != 754 or snapshot["cohort"]["held_out_sites"] != 18:
        raise PaperAssetError("Cohort does not match the accepted 754-participant, 18-site study")
    if step11_decision.get("all_three_conditions") is not False:
        raise PaperAssetError("Step 11 decision no longer matches the accepted null result")
    if step12_decision.get("positive_override_allowed") is not False:
        raise PaperAssetError("Step 12 unexpectedly permits a positive override")
    return snapshot


def build_tables(repo_root: Path, table_dir: Path) -> list[Path]:
    baseline_dir = repo_root / "outputs/analysis/step7_6_full_baselines_v2"
    step11_dir = repo_root / "outputs/analysis/step11_neural_full_parallel_v1"
    step12_dir = repo_root / "outputs/analysis/step12_neural_robustness_v1"
    generated: list[Path] = []

    baseline_rows = read_csv(baseline_dir / "model_summary.csv", BASELINE_FIELDS)
    baseline_fields = [
        "model", "label", "held_out_sites", "participants",
        "equal_site_balanced_accuracy", "pooled_balanced_accuracy", "pooled_auroc",
    ]
    baseline_table = [{
        "model": row["model"],
        "label": MODEL_LABELS[row["model"]],
        "held_out_sites": row["held_out_sites"],
        "participants": row["participants"],
        "equal_site_balanced_accuracy": row["unweighted_mean_site_balanced_accuracy"],
        "pooled_balanced_accuracy": row["pooled_balanced_accuracy"],
        "pooled_auroc": row["pooled_auroc"],
    } for row in baseline_rows]
    path = table_dir / "baseline_models.csv"
    write_csv(path, baseline_fields, baseline_table)
    generated.append(path)

    predictive_rows = read_csv(
        step11_dir / "confirmatory_predictive_contrasts.csv", PREDICTIVE_FIELDS
    )
    predictive_fields = [
        "contrast", "estimand", "primary", "held_out_sites", "estimate",
        "ci_95_low", "ci_95_high", "exact_sign_flip_p", "practical_margin",
    ]
    path = table_dir / "confirmatory_predictive_contrasts.csv"
    write_csv(path, predictive_fields, [{
        "contrast": row["contrast"],
        "estimand": row["estimand"],
        "primary": row["primary"],
        "held_out_sites": row["held_out_sites"],
        "estimate": row["observed_mean_difference"],
        "ci_95_low": row["bootstrap_ci_low"],
        "ci_95_high": row["bootstrap_ci_high"],
        "exact_sign_flip_p": row["exact_sign_flip_p"],
        "practical_margin": row["practical_margin"],
    } for row in predictive_rows])
    generated.append(path)

    representation_rows = read_csv(
        step11_dir / "representation_contrasts.csv", REPRESENTATION_FIELDS
    )
    representation_fields = [
        "endpoint", "layer", "estimand", "primary", "held_out_sites",
        "estimate", "ci_95_low", "ci_95_high",
    ]
    path = table_dir / "representation_contrasts.csv"
    write_csv(path, representation_fields, [{
        "endpoint": row["endpoint"],
        "layer": row["layer"],
        "estimand": row["estimand"],
        "primary": row["primary"],
        "held_out_sites": row["held_out_sites"],
        "estimate": row["observed_mean_difference"],
        "ci_95_low": row["bootstrap_ci_low"],
        "ci_95_high": row["bootstrap_ci_high"],
    } for row in representation_rows])
    generated.append(path)

    efficiency_rows = read_csv(step12_dir / "operator_efficiency.csv", EFFICIENCY_FIELDS)
    efficiency_fields = [
        "operator", "label", "parameter_count", "fits", "total_runtime_hours",
        "mean_runtime_seconds", "maximum_peak_gpu_memory_gib",
        "equal_site_curve_balanced_accuracy", "curve_minus_elastic_net",
    ]
    path = table_dir / "operator_efficiency.csv"
    write_csv(path, efficiency_fields, [{
        "operator": row["operator"],
        "label": OPERATOR_LABELS[row["operator"]],
        "parameter_count": row["parameter_count"],
        "fits": row["fits"],
        "total_runtime_hours": row["total_runtime_hours"],
        "mean_runtime_seconds": row["mean_runtime_seconds"],
        "maximum_peak_gpu_memory_gib": row["maximum_peak_gpu_memory_gib"],
        "equal_site_curve_balanced_accuracy": row[
            "unweighted_mean_site_curve_balanced_accuracy"
        ],
        "curve_minus_elastic_net": row["curve_minus_elastic_net"],
    } for row in efficiency_rows])
    generated.append(path)

    alternative_rows = read_csv(step12_dir / "alternative_summaries.csv", ALTERNATIVE_FIELDS)
    path = table_dir / "robustness_summaries.csv"
    write_csv(path, ALTERNATIVE_FIELDS, alternative_rows)
    generated.append(path)
    return generated


def _plot_context() -> dict[str, Any]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    }


def _save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "bunn-abide Step 13 deterministic paper-assets builder"},
    )
    plt.close(fig)


def plot_predictive_density_curves(repo_root: Path, output: Path) -> None:
    step11 = repo_root / "outputs/analysis/step11_neural_full_parallel_v1"
    baseline_dir = repo_root / "outputs/analysis/step7_6_full_baselines_v2"
    contract = load_json(repo_root / "configs/neural_confirmatory_analysis_v1.json")
    rows = read_csv(step11 / "site_configuration_metrics.csv", SITE_CONFIGURATION_FIELDS)
    baseline_rows = read_csv(baseline_dir / "per_site_metrics.csv", BASELINE_SITE_FIELDS)
    densities = [finite_float(value, "density") for value in contract["densities"]]
    density_percent = np.asarray(densities) * 100.0
    sites = sorted({row["held_out_site"] for row in rows})
    if len(sites) != 18:
        raise PaperAssetError("Predictive figure requires 18 held-out sites")
    values = {
        (row["held_out_site"], row["operator"], finite_float(row["density"], "density")):
        finite_float(row["mean_balanced_accuracy"], "mean_balanced_accuracy")
        for row in rows
    }
    baseline = {
        row["held_out_site"]: finite_float(row["balanced_accuracy"], "balanced_accuracy")
        for row in baseline_rows
        if row["model"] == "connectome_elastic_net_logistic"
    }
    if set(baseline) != set(sites):
        raise PaperAssetError("Baseline and neural sites are not aligned")

    with plt.rc_context(_plot_context()):
        fig, axis = plt.subplots(figsize=(7.3, 4.45))
        for operator, anchor in contract["curve_anchors"].items():
            matrix = np.asarray([
                [values[(site, anchor, 0.0)]]
                + [values[(site, operator, density)] for density in densities[1:]]
                for site in sites
            ])
            mean = matrix.mean(axis=0)
            error = 1.96 * matrix.std(axis=0, ddof=1) / math.sqrt(len(sites))
            axis.errorbar(
                density_percent,
                mean,
                yerr=error,
                color=COLORS[operator],
                marker=MARKERS[operator],
                linestyle=LINESTYLES[operator],
                linewidth=1.7,
                markersize=5,
                capsize=2.5,
                label=OPERATOR_LABELS[operator],
            )
        axis.axhline(
            np.mean(list(baseline.values())), color="#8E5C85", linestyle=(0, (5, 3)),
            linewidth=1.5, label="Connectome elastic net",
        )
        axis.axhline(0.5, color="#666666", linestyle=":", linewidth=1.2, label="Chance")
        axis.set_xlabel("Retained positive-edge density (%)")
        axis.set_ylabel("Equal-site mean balanced accuracy")
        axis.set_xticks(density_percent, ["0", "1", "5", "10", "20"])
        axis.set_ylim(0.49, 0.67)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        axis.legend(loc="upper right", frameon=True, ncols=1)
        axis.text(
            0.0, -0.19, "Error bars: 1.96 × SE across 18 held-out sites.",
            transform=axis.transAxes, fontsize=7.5, color="#444444",
        )
        fig.subplots_adjust(bottom=0.20)
        _save_figure(fig, output)


def plot_representation_density_curves(repo_root: Path, output: Path) -> None:
    step11 = repo_root / "outputs/analysis/step11_neural_full_parallel_v1"
    contract = load_json(repo_root / "configs/neural_confirmatory_analysis_v1.json")
    rows = read_csv(step11 / "site_representation_metrics.csv", SITE_REPRESENTATION_FIELDS)
    densities = [finite_float(value, "density") for value in contract["densities"]]
    density_percent = np.asarray(densities) * 100.0
    sites = sorted({row["held_out_site"] for row in rows})
    values = {
        (row["held_out_site"], row["operator"], finite_float(row["density"], "density"), row["layer"]): row
        for row in rows
    }
    endpoints = [
        ("normalized_effective_rank", "Normalized effective rank"),
        ("normalized_dispersion", "Normalized dispersion"),
        ("mean_pairwise_cosine", "Mean pairwise cosine similarity"),
        ("invariant_edge_transport_distance", "Invariant edge-transport distance"),
    ]
    with plt.rc_context(_plot_context()):
        fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.15), sharex=True)
        for axis, (endpoint, title) in zip(axes.flat, endpoints, strict=True):
            for operator, anchor in contract["curve_anchors"].items():
                matrix = np.asarray([
                    [finite_float(values[(site, anchor, 0.0, "layer_2")][endpoint], endpoint)]
                    + [
                        finite_float(values[(site, operator, density, "layer_2")][endpoint], endpoint)
                        for density in densities[1:]
                    ]
                    for site in sites
                ])
                axis.plot(
                    density_percent,
                    matrix.mean(axis=0),
                    color=COLORS[operator],
                    marker=MARKERS[operator],
                    linestyle=LINESTYLES[operator],
                    linewidth=1.6,
                    markersize=4.5,
                    label=OPERATOR_LABELS[operator],
                )
            axis.set_title(title)
            axis.set_xticks(density_percent, ["0", "1", "5", "10", "20"])
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        for axis in axes[1, :]:
            axis.set_xlabel("Retained density (%)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncols=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
        fig.suptitle("Common-frame layer-2 representation diagnostics", y=1.055, fontsize=11)
        fig.text(
            0.5, -0.015,
            "Curves are equal-site means; zero-density points use the matched identity or learned-local anchor.",
            ha="center", fontsize=7.5, color="#444444",
        )
        fig.tight_layout()
        _save_figure(fig, output)


def plot_site_influence(repo_root: Path, output: Path) -> None:
    rows = read_csv(
        repo_root / "outputs/analysis/step12_neural_robustness_v1/leave_one_site_out.csv",
        LOO_FIELDS,
    )
    rows = [row for row in rows if row["contrast"] == "learned_bunn_curve_minus_gcn_curve"]
    rows.sort(key=lambda row: finite_float(row["leave_one_out_estimate"], "estimate"), reverse=True)
    if len(rows) != 18:
        raise PaperAssetError("Site-influence figure requires 18 exclusions")
    estimates = np.asarray([finite_float(row["leave_one_out_estimate"], "estimate") for row in rows])
    lows = np.asarray([finite_float(row["bootstrap_ci_low"], "ci low") for row in rows])
    highs = np.asarray([finite_float(row["bootstrap_ci_high"], "ci high") for row in rows])
    full = finite_float(rows[0]["full_estimate"], "full estimate")
    positions = np.arange(len(rows))
    labels = [f"Exclude {row['excluded_site']}" for row in rows]
    with plt.rc_context(_plot_context()):
        fig, axis = plt.subplots(figsize=(7.4, 6.2))
        axis.errorbar(
            estimates, positions, xerr=np.vstack((estimates - lows, highs - estimates)),
            fmt="o", color="#35689B", ecolor="#35689B", capsize=2.5, markersize=4.5,
        )
        axis.axvline(0.0, color="#666666", linestyle="--", linewidth=1.2, label="Zero difference")
        axis.axvline(full, color="#D64B4B", linestyle=":", linewidth=1.5, label="Full 18-site estimate")
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xlabel("BuNN − GCN normalized performance-curve difference")
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        axis.legend(loc="lower right", frameon=True)
        fig.tight_layout()
        _save_figure(fig, output)


def plot_seed_stability(repo_root: Path, output: Path) -> None:
    rows = read_csv(
        repo_root / "outputs/analysis/step12_neural_robustness_v1/seed_specific_curves.csv",
        SEED_FIELDS,
    )
    contrasts = [
        ("learned_bunn_curve_minus_gcn_curve", "BuNN − GCN", "#35689B", "o"),
        ("learned_bunn_curve_minus_connectome_elastic_net", "BuNN − elastic net", "#D64B4B", "s"),
    ]
    seeds = sorted({finite_int(row["seed"], "seed") for row in rows})
    positions = np.arange(len(seeds), dtype=float)
    with plt.rc_context(_plot_context()):
        fig, axis = plt.subplots(figsize=(7.3, 4.2))
        for offset, (contrast, label, color, marker) in zip((-0.10, 0.10), contrasts, strict=True):
            selected = {finite_int(row["seed"], "seed"): row for row in rows if row["contrast"] == contrast}
            if set(selected) != set(seeds):
                raise PaperAssetError(f"Incomplete seed rows for {contrast}")
            estimates = np.asarray([
                finite_float(selected[seed]["observed_mean_difference"], "estimate") for seed in seeds
            ])
            lows = np.asarray([
                finite_float(selected[seed]["bootstrap_ci_low"], "ci low") for seed in seeds
            ])
            highs = np.asarray([
                finite_float(selected[seed]["bootstrap_ci_high"], "ci high") for seed in seeds
            ])
            axis.errorbar(
                positions + offset,
                estimates,
                yerr=np.vstack((estimates - lows, highs - estimates)),
                fmt=marker,
                color=color,
                ecolor=color,
                capsize=2.5,
                markersize=4.5,
                label=label,
            )
        axis.axhline(0.0, color="#666666", linestyle="--", linewidth=1.2)
        axis.set_xticks(positions, [str(seed) for seed in seeds])
        axis.set_xlabel("Final training seed")
        axis.set_ylabel("Equal-site performance-curve difference")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        axis.legend(loc="lower left", frameon=True)
        fig.tight_layout()
        _save_figure(fig, output)


def build_paper_assets(
    repo_root: Path,
    contract_path: Path,
    paper_generated_dir: Path,
    reproducibility_dir: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    contract = load_json(contract_path)
    source_records = validate_frozen_inputs(repo_root, contract)
    validate_analysis_manifest(repo_root / "outputs/analysis/step11_neural_full_parallel_v1")
    validate_analysis_manifest(repo_root / "outputs/analysis/step12_neural_robustness_v1")

    figure_dir = paper_generated_dir / "figures"
    table_dir = paper_generated_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    reproducibility_dir.mkdir(parents=True, exist_ok=True)

    def logical_path(path: Path) -> str:
        if path.is_relative_to(paper_generated_dir):
            return f"paper/generated/{path.relative_to(paper_generated_dir).as_posix()}"
        if path.is_relative_to(reproducibility_dir):
            return f"reproducibility/{path.relative_to(reproducibility_dir).as_posix()}"
        raise PaperAssetError(f"Generated path is outside the declared output roots: {path}")

    generated: list[Path] = []
    figure_names = contract.get("generated_figures")
    expected_figure_names = [
        "predictive_density_curves.png",
        "representation_density_curves.png",
        "site_influence.png",
        "seed_stability.png",
    ]
    if figure_names != expected_figure_names:
        raise PaperAssetError("Contract has an unexpected generated_figures list")
    figure_builders = {
        "predictive_density_curves.png": plot_predictive_density_curves,
        "representation_density_curves.png": plot_representation_density_curves,
        "site_influence.png": plot_site_influence,
        "seed_stability.png": plot_seed_stability,
    }
    for name in figure_names:
        destination = figure_dir / name
        figure_builders[name](repo_root, destination)
        generated.append(destination)

    generated.extend(build_tables(repo_root, table_dir))
    snapshot = build_result_snapshot(repo_root, contract)
    snapshot_path = reproducibility_dir / "result_snapshot.json"
    write_json(snapshot_path, snapshot)
    generated.append(snapshot_path)

    generated_records = [{
        "path": logical_path(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    } for path in sorted(generated)]

    artifact_manifest = {
        "schema_version": "step13_artifact_manifest_v1",
        "contract_version": contract["contract_version"],
        "evidence_commit": contract["evidence_commit"],
        "run_id": contract["run_id"],
        "frozen_inputs": source_records,
        "generated_paper_assets": generated_records,
        "privacy_tier": "local audit package; public-release filtering occurs in Step 13.4",
    }
    artifact_manifest_path = reproducibility_dir / "artifact_manifest.json"
    write_json(artifact_manifest_path, artifact_manifest)

    asset_records = generated_records + [{
        "path": logical_path(artifact_manifest_path),
        "sha256": sha256_file(artifact_manifest_path),
        "bytes": artifact_manifest_path.stat().st_size,
    }]
    asset_manifest = {
        "schema_version": "step13_paper_asset_manifest_v1",
        "contract_version": contract["contract_version"],
        "assets": sorted(asset_records, key=lambda row: row["path"]),
    }
    asset_manifest_path = paper_generated_dir / "paper_asset_manifest.json"
    write_json(asset_manifest_path, asset_manifest)

    output_paths = [logical_path(path) for path in generated]
    output_paths.extend([
        logical_path(artifact_manifest_path),
        logical_path(asset_manifest_path),
    ])
    expected = set(contract.get("required_outputs", []))
    if set(output_paths) != expected:
        missing = sorted(expected - set(output_paths))
        unexpected = sorted(set(output_paths) - expected)
        raise PaperAssetError(
            f"Generated output contract mismatch; missing={missing}, unexpected={unexpected}"
        )
    return {
        "contract_version": contract["contract_version"],
        "validated_inputs": len(source_records),
        "generated_outputs": sorted(output_paths),
        "result_category": snapshot["step11_decision"]["result_category"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--contract", type=Path, default=Path("configs/paper_assets_v1.json")
    )
    parser.add_argument(
        "--paper-generated-dir", type=Path, default=Path("paper/generated")
    )
    parser.add_argument(
        "--reproducibility-dir", type=Path, default=Path("reproducibility")
    )
    arguments = parser.parse_args()
    repo_root = arguments.repo_root.resolve()
    contract = arguments.contract
    if not contract.is_absolute():
        contract = repo_root / contract
    paper_dir = arguments.paper_generated_dir
    if not paper_dir.is_absolute():
        paper_dir = repo_root / paper_dir
    reproducibility_dir = arguments.reproducibility_dir
    if not reproducibility_dir.is_absolute():
        reproducibility_dir = repo_root / reproducibility_dir
    result = build_paper_assets(repo_root, contract, paper_dir, reproducibility_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
