"""One-time frozen analysis of complete, audited E2 test predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.extensions.e2_analysis import classify_evidence, frozen_contrasts, load_replicate_metrics, sha256_file
from src.extensions.e2_training import write_json_atomic


ACK = "I acknowledge that this command opens the audited E2 scientific values under analysis protocol v1."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/extensions/e2_synthetic_geometry_v1.json")
    parser.add_argument("--analysis", type=Path, default=ROOT / "configs/extensions/e2_analysis_v1.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--unblind-acknowledgement", required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for key, values in sorted(groups.items()):
        result = {name: value for name, value in zip(keys, key)}
        for endpoint in ("balanced_accuracy", "auroc", "brier_score", "mean_transport_error", "runtime_seconds", "best_epoch"):
            array = np.asarray([row[endpoint] for row in values], dtype=float)
            result[f"mean_{endpoint}"] = float(array.mean())
            result[f"sd_{endpoint}"] = float(array.std(ddof=1)) if len(array) > 1 else 0.0
        result["parameter_count"] = values[0]["parameter_count"]
        output.append(result)
    return output


def render_figures(output: Path, summaries: list[dict], contrasts: list[dict]) -> None:
    primary_names = ("primary_conditional_advantage", "oracle_positive_control", "matched_capacity_anchor", "S0_capacity_warning")
    primary = [next(row for row in contrasts if row["contrast"] == name) for name in primary_names]
    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    y = np.arange(len(primary))
    means = np.asarray([row["mean_difference_pp"] for row in primary])
    lower = np.asarray([row["ci95_lower_pp"] for row in primary])
    upper = np.asarray([row["ci95_upper_pp"] for row in primary])
    axis.errorbar(means, y, xerr=np.vstack((means - lower, upper - means)), fmt="o", color="#30343B", capsize=3)
    axis.axvline(0, color="#8A8A8A", linewidth=0.8)
    axis.set_yticks(y, [name.replace("_", " ") for name in primary_names])
    axis.set_xlabel("Paired balanced-accuracy difference (percentage points)")
    axis.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output / "e2_primary_forest.png", dpi=220)
    plt.close(fig)

    families = ["S0_no_geometry", "S1_recoverable_geometry", "S2_incorrect_topology", "S3_shuffled_geometry", "S5_unlearnable_subject_frames", "S6_global_feature_analogue"]
    operators = ["gcn", "learned_local", "learned_bunn", "oracle_true_map"]
    fig, axis = plt.subplots(figsize=(8.0, 4.4))
    for operator in operators:
        values = [next(row["mean_balanced_accuracy"] for row in summaries if row["family"] == family and row["transport_noise_degrees"] == 0 and row["operator"] == operator) for family in families]
        axis.plot(range(len(families)), values, marker="o", label=operator.replace("_", " "))
    axis.set_xticks(range(len(families)), [name.split("_")[0] for name in families])
    axis.set_ylabel("Mean test balanced accuracy")
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "e2_condition_operator_plot.png", dpi=220)
    plt.close(fig)

    noise_levels = [0, 15, 30, 60, 120]
    fig, axis = plt.subplots(figsize=(6.6, 3.8))
    for operator in ("gcn", "learned_local", "learned_bunn", "oracle_true_map"):
        values = []
        for noise in noise_levels:
            family = "S1_recoverable_geometry" if noise == 0 else "S4_transport_noise"
            values.append(next(row["mean_balanced_accuracy"] for row in summaries if row["family"] == family and row["transport_noise_degrees"] == noise and row["operator"] == operator))
        axis.plot(noise_levels, values, marker="o", label=operator.replace("_", " "))
    axis.set_xlabel("Frame-marker corruption (degrees)")
    axis.set_ylabel("Mean test balanced accuracy")
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "e2_transport_noise_plot.png", dpi=220)
    plt.close(fig)


def execute(protocol_path: Path, analysis_path: Path, run_dir: Path, output: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    protocol_hash = sha256_file(protocol_path)
    rows, manager = load_replicate_metrics(run_dir, protocol, protocol_hash)
    contrasts = frozen_contrasts(rows, analysis)
    classification = classify_evidence(contrasts, analysis)
    summaries = summarize(rows, ("family", "transport_noise_degrees", "operator"))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "replicate_metrics.csv", rows, list(rows[0]))
    write_csv(output / "primary_contrasts.csv", contrasts, ["contrast", "replicates", "mean_difference_pp", "ci95_lower_pp", "ci95_upper_pp", "exact_sign_flip_p"])
    write_csv(output / "condition_summary.csv", summaries, list(summaries[0]))
    transport = [{key: row[key] for key in ("family", "transport_noise_degrees", "replicate_seed", "operator", "mean_transport_error")} for row in rows]
    write_csv(output / "transport_recovery.csv", transport, list(transport[0]))
    runtime = [{key: row[key] for key in ("family", "transport_noise_degrees", "replicate_seed", "operator", "runtime_seconds", "best_epoch", "parameter_count")} for row in rows]
    write_csv(output / "runtime_summary.csv", runtime, list(runtime[0]))
    render_figures(output, summaries, contrasts)
    metadata = {
        "state": "analysis_complete_unblinded",
        "protocol_sha256": protocol_hash,
        "analysis_protocol_sha256": sha256_file(analysis_path),
        "manager_completion_sha256": sha256_file(run_dir / "manager_complete.json"),
        "source_cells": manager["completed_cells"],
        **classification,
    }
    write_json_atomic(output / "analysis_metadata.json", metadata)
    return metadata


def main() -> int:
    args = parse_args()
    if args.unblind_acknowledgement != ACK:
        raise RuntimeError("Exact E2 unblind acknowledgement is required")
    execute(args.protocol, args.analysis, args.run_dir, args.output_dir)
    print(json.dumps({"state": "analysis_complete", "scientific_values_written": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
