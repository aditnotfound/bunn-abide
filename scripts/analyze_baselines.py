"""Produce the pre-specified Step 7.6 baseline analysis from sealed outputs.

The module recomputes every predictive summary from participant-level held-out
predictions, verifies it against the runner's saved per-site metrics, and uses
paired site-level bootstrapping for every model contrast.  It is intentionally
separate from the baseline runner so its code and synthetic-data tests can be
reviewed before a completed research run is unblinded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_baselines import METRIC_FIELDS, MODEL_NAMES, PREDICTION_FIELDS, TUNING_FIELDS, WARNING_FIELDS


ANALYSIS_VERSION = "step7_6_baseline_analysis_v1"
SUMMARY_FIELDS = [
    "model",
    "held_out_sites",
    "participants",
    "unweighted_mean_site_balanced_accuracy",
    "pooled_balanced_accuracy",
    "pooled_auroc",
    "pooled_sensitivity",
    "pooled_specificity",
]
SITE_METRIC_FIELDS = [
    "model",
    "held_out_site",
    "participants",
    "asd",
    "control",
    "balanced_accuracy",
    "auroc",
    "sensitivity",
    "specificity",
]
PAIRWISE_FIELDS = [
    "left_model",
    "right_model",
    "estimand",
    "held_out_sites",
    "observed_mean_difference",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "bootstrap_resamples",
    "bootstrap_seed",
]
SELECTED_TUNING_FIELDS = [
    "model",
    "held_out_site",
    "C",
    "l1_ratio",
    "inner_mean_site_balanced_accuracy",
    "inner_sites_scored",
]
WARNING_SUMMARY_FIELDS = ["model", "fit_phase", "warning_category", "warning_rows"]


class AnalysisError(ValueError):
    """Raised when a completed baseline artifact cannot support the analysis."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/baseline_protocol.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--confirm-unblind-run-id",
        default=None,
        help="Required explicit acknowledgement. It must exactly equal metadata.json run_id.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path, expected_fields: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise AnalysisError(f"Missing required artifact: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_fields:
            raise AnalysisError(
                f"Unexpected schema in {path.name}: {reader.fieldnames!r} != {expected_fields!r}"
            )
        return list(reader)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise AnalysisError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(parsed):
        raise AnalysisError(f"{label} is not finite")
    return parsed


def finite_int(value: str, label: str) -> int:
    parsed = finite_float(value, label)
    if not parsed.is_integer():
        raise AnalysisError(f"{label} is not an integer: {value!r}")
    return int(parsed)


def validate_predictions(
    rows: list[dict[str, str]], threshold: float
) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    participant_model_keys: set[tuple[str, str]] = set()
    for row in rows:
        model = row["model"]
        if model not in MODEL_NAMES:
            raise AnalysisError(f"Unknown model in predictions: {model!r}")
        label = finite_int(row["label_asd"], "prediction label_asd")
        predicted = finite_int(row["predicted_asd"], "prediction predicted_asd")
        probability = finite_float(row["probability_asd"], "prediction probability_asd")
        if label not in {0, 1} or predicted not in {0, 1}:
            raise AnalysisError("Prediction labels must be binary")
        if not 0.0 <= probability <= 1.0:
            raise AnalysisError("Prediction probability_asd is outside [0, 1]")
        if predicted != int(probability >= threshold):
            raise AnalysisError("predicted_asd does not match the frozen decision threshold")
        key = (model, row["subject_id"])
        if key in participant_model_keys:
            raise AnalysisError(f"Duplicate participant/model prediction: {key!r}")
        participant_model_keys.add(key)
        grouped[(model, row["held_out_site"])].append(row)

    models = {model for model, _site in grouped}
    if models != set(MODEL_NAMES):
        raise AnalysisError(f"Prediction models differ from frozen models: {sorted(models)!r}")
    sites_by_model = {
        model: {site for candidate_model, site in grouped if candidate_model == model}
        for model in models
    }
    reference_sites = sites_by_model[MODEL_NAMES[0]]
    if not reference_sites or any(sites != reference_sites for sites in sites_by_model.values()):
        raise AnalysisError("Every model must cover exactly the same non-empty held-out site set")
    for (model, site), site_rows in grouped.items():
        labels = {finite_int(row["label_asd"], "prediction label_asd") for row in site_rows}
        if labels != {0, 1}:
            raise AnalysisError(f"{model}/{site} lacks one diagnostic class")
    return grouped


def site_metric(model: str, site: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    labels = np.asarray([finite_int(row["label_asd"], "prediction label_asd") for row in rows], dtype=int)
    probabilities = np.asarray(
        [finite_float(row["probability_asd"], "prediction probability_asd") for row in rows],
        dtype=float,
    )
    predicted = np.asarray(
        [finite_int(row["predicted_asd"], "prediction predicted_asd") for row in rows],
        dtype=int,
    )
    positives = labels == 1
    negatives = labels == 0
    return {
        "model": model,
        "held_out_site": site,
        "participants": len(rows),
        "asd": int(positives.sum()),
        "control": int(negatives.sum()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "sensitivity": float((predicted[positives] == 1).mean()),
        "specificity": float((predicted[negatives] == 0).mean()),
    }


def recompute_site_metrics(
    grouped: dict[tuple[str, str], list[dict[str, str]]]
) -> list[dict[str, Any]]:
    return [
        site_metric(model, site, grouped[(model, site)])
        for model, site in sorted(grouped, key=lambda value: (value[0], value[1]))
    ]


def verify_runner_metrics(
    runner_rows: list[dict[str, str]], computed: list[dict[str, Any]]
) -> None:
    expected_keys = {(row["model"], row["held_out_site"]) for row in computed}
    runner_keys = {(row["model"], row["held_out_site"]) for row in runner_rows}
    if runner_keys != expected_keys or len(runner_rows) != len(expected_keys):
        raise AnalysisError("Runner metric coverage does not match recomputed per-site metrics")
    computed_by_key = {(row["model"], row["held_out_site"]): row for row in computed}
    for row in runner_rows:
        key = (row["model"], row["held_out_site"])
        source = computed_by_key[key]
        for field in ("participants", "asd", "control"):
            if finite_int(row[field], f"runner {field}") != source[field]:
                raise AnalysisError(f"Runner {field} differs from recomputed value for {key!r}")
        for field in ("balanced_accuracy", "auroc", "sensitivity", "specificity"):
            if not math.isclose(
                finite_float(row[field], f"runner {field}"), source[field], rel_tol=0.0, abs_tol=1e-12
            ):
                raise AnalysisError(f"Runner {field} differs from recomputed value for {key!r}")


def pooled_metric(model: str, grouped: dict[tuple[str, str], list[dict[str, str]]]) -> dict[str, float]:
    rows = [row for (candidate_model, _site), values in grouped.items() if candidate_model == model for row in values]
    labels = np.asarray([finite_int(row["label_asd"], "prediction label_asd") for row in rows], dtype=int)
    probabilities = np.asarray([finite_float(row["probability_asd"], "prediction probability_asd") for row in rows], dtype=float)
    predicted = np.asarray([finite_int(row["predicted_asd"], "prediction predicted_asd") for row in rows], dtype=int)
    positives = labels == 1
    negatives = labels == 0
    return {
        "pooled_balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "pooled_auroc": float(roc_auc_score(labels, probabilities)),
        "pooled_sensitivity": float((predicted[positives] == 1).mean()),
        "pooled_specificity": float((predicted[negatives] == 0).mean()),
    }


def model_summary(
    site_rows: list[dict[str, Any]], grouped: dict[tuple[str, str], list[dict[str, str]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        rows = [row for row in site_rows if row["model"] == model]
        pooled = pooled_metric(model, grouped)
        output.append(
            {
                "model": model,
                "held_out_sites": len(rows),
                "participants": sum(int(row["participants"]) for row in rows),
                "unweighted_mean_site_balanced_accuracy": float(
                    np.mean([float(row["balanced_accuracy"]) for row in rows])
                ),
                **pooled,
            }
        )
    return output


def paired_bootstrap(
    site_rows: list[dict[str, Any]], resamples: int, seed: int
) -> list[dict[str, Any]]:
    by_model_site = {(row["model"], row["held_out_site"]): row for row in site_rows}
    sites = sorted({row["held_out_site"] for row in site_rows})
    models = list(MODEL_NAMES)
    if any((model, site) not in by_model_site for model in models for site in sites):
        raise AnalysisError("Cannot form paired bootstrap: model/site coverage is incomplete")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(sites), size=(resamples, len(sites)))
    output: list[dict[str, Any]] = []
    for left, right in itertools.combinations(models, 2):
        differences = np.asarray(
            [
                float(by_model_site[(left, site)]["balanced_accuracy"])
                - float(by_model_site[(right, site)]["balanced_accuracy"])
                for site in sites
            ],
            dtype=float,
        )
        bootstrap_means = differences[indices].mean(axis=1)
        output.append(
            {
                "left_model": left,
                "right_model": right,
                "estimand": "unweighted mean held-out-site balanced accuracy (left minus right)",
                "held_out_sites": len(sites),
                "observed_mean_difference": float(differences.mean()),
                "bootstrap_ci_low": float(np.quantile(bootstrap_means, 0.025)),
                "bootstrap_ci_high": float(np.quantile(bootstrap_means, 0.975)),
                "bootstrap_resamples": resamples,
                "bootstrap_seed": seed,
            }
        )
    return output


def selected_tuning_rows(rows: list[dict[str, str]], expected_sites: set[str]) -> list[dict[str, Any]]:
    allowed_flags = {"0", "1", "false", "true"}
    invalid_flags = sorted({row["selected"].strip().lower() for row in rows} - allowed_flags)
    if invalid_flags:
        raise AnalysisError(f"Unexpected tuning selected flag values: {invalid_flags!r}")
    # The runner writes integer CSV flags (0/1); true/false remains accepted
    # for explicitly documented external/synthetic fixtures.
    selected = [
        row for row in rows if row["selected"].strip().lower() in {"1", "true"}
    ]
    keys = {(row["model"], row["held_out_site"]) for row in selected}
    expected = {(model, site) for model in MODEL_NAMES for site in expected_sites}
    if keys != expected or len(selected) != len(expected):
        raise AnalysisError("Tuning artifacts do not contain exactly one selected candidate per model/site")
    return [
        {
            "model": row["model"],
            "held_out_site": row["held_out_site"],
            "C": finite_float(row["C"], "selected C"),
            "l1_ratio": "" if row["l1_ratio"] == "" else finite_float(row["l1_ratio"], "selected l1_ratio"),
            "inner_mean_site_balanced_accuracy": finite_float(
                row["inner_mean_site_balanced_accuracy"], "selected inner score"
            ),
            "inner_sites_scored": finite_int(row["inner_sites_scored"], "selected inner sites"),
        }
        for row in sorted(selected, key=lambda row: (row["model"], row["held_out_site"]))
    ]


def warning_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    counts = Counter((row["model"], row["fit_phase"], row["warning_category"]) for row in rows)
    return [
        {
            "model": model,
            "fit_phase": fit_phase,
            "warning_category": category,
            "warning_rows": count,
        }
        for (model, fit_phase, category), count in sorted(counts.items())
    ]


def plot_per_site_balanced_accuracy(rows: list[dict[str, Any]], output: Path) -> None:
    sites = sorted({row["held_out_site"] for row in rows})
    colors = {model: color for model, color in zip(MODEL_NAMES, ("#4C78A8", "#F58518", "#54A24B"), strict=True)}
    fig, axis = plt.subplots(figsize=(10, max(4.5, 0.35 * len(sites) + 1.5)))
    offsets = np.linspace(-0.22, 0.22, len(MODEL_NAMES))
    for offset, model in zip(offsets, MODEL_NAMES, strict=True):
        values = {
            row["held_out_site"]: float(row["balanced_accuracy"])
            for row in rows
            if row["model"] == model
        }
        axis.scatter(
            [values[site] for site in sites],
            np.arange(len(sites)) + offset,
            label=model,
            color=colors[model],
            s=28,
        )
    axis.axvline(0.5, color="#777777", linestyle="--", linewidth=1, label="chance BA")
    axis.set(yticks=np.arange(len(sites)), yticklabels=sites, xlabel="Held-out-site balanced accuracy")
    axis.set_xlim(0.0, 1.0)
    axis.legend(fontsize=8, loc="lower right")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_paired_differences(rows: list[dict[str, Any]], output: Path) -> None:
    labels = [f"{row['left_model']} -\n{row['right_model']}" for row in rows]
    values = np.asarray([float(row["observed_mean_difference"]) for row in rows])
    lowers = np.asarray([float(row["bootstrap_ci_low"]) for row in rows])
    uppers = np.asarray([float(row["bootstrap_ci_high"]) for row in rows])
    fig, axis = plt.subplots(figsize=(9, 4.5))
    positions = np.arange(len(rows))
    axis.errorbar(
        positions,
        values,
        yerr=np.vstack((values - lowers, uppers - values)),
        fmt="o",
        color="#4C78A8",
        capsize=4,
    )
    axis.axhline(0.0, color="#777777", linestyle="--", linewidth=1)
    axis.set(xticks=positions, xticklabels=labels, ylabel="Mean site balanced-accuracy difference")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def analyze_run(run_dir: Path, protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run the frozen analysis and return only metadata about generated outputs."""
    if output_dir.exists():
        raise AnalysisError(f"Refusing to overwrite existing analysis directory: {output_dir}")
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.is_file():
        raise AnalysisError(f"Missing run metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") not in {"complete", None}:
        raise AnalysisError("Run metadata does not identify a completed run")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    evaluation = protocol["evaluation"]
    threshold = float(evaluation["decision_threshold"])
    resamples = int(evaluation["bootstrap_resamples"])
    seed = int(evaluation["seed"])
    if resamples != 10000:
        raise AnalysisError("Frozen protocol must use exactly 10,000 bootstrap resamples")

    predictions_path = run_dir / "predictions.csv"
    metrics_path = run_dir / "test_metrics.csv"
    tuning_path = run_dir / "tuning_scores.csv"
    warnings_path = run_dir / "fit_warnings.csv"
    predictions = read_csv(predictions_path, PREDICTION_FIELDS)
    grouped = validate_predictions(predictions, threshold)
    recomputed = recompute_site_metrics(grouped)
    observed_sites = {row["held_out_site"] for row in recomputed}
    if observed_sites != set(metadata.get("held_out_sites", [])):
        raise AnalysisError("Prediction held-out sites differ from the run metadata")
    if set(metadata.get("models", [])) != set(MODEL_NAMES):
        raise AnalysisError("Run metadata models differ from the frozen analysis models")
    verify_runner_metrics(read_csv(metrics_path, METRIC_FIELDS), recomputed)
    summaries = model_summary(recomputed, grouped)
    comparisons = paired_bootstrap(recomputed, resamples, seed)
    selected = selected_tuning_rows(
        read_csv(tuning_path, TUNING_FIELDS), {row["held_out_site"] for row in recomputed}
    )
    warnings = warning_summary(read_csv(warnings_path, WARNING_FIELDS))

    output_dir.mkdir(parents=True)
    write_csv(output_dir / "per_site_metrics.csv", recomputed, SITE_METRIC_FIELDS)
    write_csv(output_dir / "model_summary.csv", summaries, SUMMARY_FIELDS)
    write_csv(output_dir / "paired_balanced_accuracy_differences.csv", comparisons, PAIRWISE_FIELDS)
    write_csv(output_dir / "selected_hyperparameters.csv", selected, SELECTED_TUNING_FIELDS)
    write_csv(output_dir / "fit_warning_summary.csv", warnings, WARNING_SUMMARY_FIELDS)
    plot_per_site_balanced_accuracy(recomputed, output_dir / "per_site_balanced_accuracy.png")
    plot_paired_differences(comparisons, output_dir / "paired_balanced_accuracy_differences.png")

    source_files = [metadata_path, protocol_path, predictions_path, metrics_path, tuning_path, warnings_path]
    generated_files = sorted(path.name for path in output_dir.iterdir()) + ["analysis_manifest.json"]
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "run_id": metadata.get("run_id"),
        "primary_estimand": "unweighted mean held-out-site balanced accuracy",
        "paired_contrasts": "all frozen-model pairs; left minus right",
        "bootstrap": {"resamples": resamples, "seed": seed, "unit": "held-out site", "interval": "percentile 95%"},
        "source_sha256": {str(path): sha256_file(path) for path in source_files},
        "generated_files": generated_files,
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "run_id": metadata.get("run_id"),
        "output_dir": str(output_dir),
        "generated_files": sorted(path.name for path in output_dir.iterdir()),
    }


def main() -> None:
    args = parse_args()
    metadata_path = args.run_dir / "metadata.json"
    if not metadata_path.is_file():
        raise SystemExit(f"Missing run metadata: {metadata_path}")
    run_id = json.loads(metadata_path.read_text(encoding="utf-8")).get("run_id")
    if args.confirm_unblind_run_id != run_id:
        raise SystemExit(
            "Refusing to read a completed result run without explicit acknowledgement: "
            "pass --confirm-unblind-run-id with the exact run ID from metadata.json."
        )
    result = analyze_run(args.run_dir, args.protocol, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
