"""Score-blind equivalence check for sequential and site-parallel neural smokes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from scripts.run_baselines import write_json_atomic
from scripts.run_neural_full import SITE_ARTIFACT_FIELDS


class EquivalenceError(ValueError):
    """Raised when execution modes differ scientifically or structurally."""


KEY_FIELDS = {
    "predictions.csv": ["operator", "density", "seed", "outer_fold", "held_out_site", "subject_id"],
    "test_metrics.csv": ["operator", "density", "seed", "outer_fold", "held_out_site"],
    "tuning_scores.csv": ["operator", "density", "outer_fold", "held_out_site", "candidate_index"],
    "inner_site_scores.csv": [
        "operator", "density", "outer_fold", "held_out_site", "candidate_index",
        "inner_validation_fold", "tuning_seed", "site_id",
    ],
    "training_curves.csv": [
        "fit_scope", "operator", "density", "outer_fold", "held_out_site",
        "candidate_index", "inner_validation_fold", "seed", "epoch",
    ],
    "diagnostics.csv": [
        "operator", "density", "seed", "outer_fold", "held_out_site", "subject_id", "layer",
    ],
    "fit_runtime.csv": [
        "fit_scope", "operator", "density", "outer_fold", "held_out_site",
        "candidate_index", "inner_validation_fold", "seed",
    ],
    "fit_warnings.csv": [
        "operator", "density", "outer_fold", "held_out_site", "fit_scope", "failure_type",
    ],
}

RUNTIME_VARIABLE_FIELDS = {"runtime_seconds", "peak_gpu_memory_bytes", "resumed"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path, fields: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            raise EquivalenceError(f"Schema mismatch: {path.name}")
        return list(reader)


def compare_value(left: str, right: str, tolerance: float) -> float:
    if left == right:
        return 0.0
    if left == "" or right == "":
        raise EquivalenceError("Empty/non-empty field mismatch")
    try:
        left_float = float(left)
        right_float = float(right)
    except ValueError as error:
        raise EquivalenceError("Categorical field mismatch") from error
    if not math.isfinite(left_float) or not math.isfinite(right_float):
        raise EquivalenceError("Non-finite equivalence field")
    difference = abs(left_float - right_float)
    if difference > tolerance:
        raise EquivalenceError("Numerical execution difference exceeds tolerance")
    return difference


def compare_runs(
    sequential_dir: Path,
    parallel_dir: Path,
    *,
    tolerance: float,
) -> dict[str, Any]:
    sequential_metadata = load_json(sequential_dir / "metadata.json")
    parallel_metadata = load_json(parallel_dir / "metadata.json")
    if sequential_metadata.get("status") != "complete" or parallel_metadata.get("status") != "complete":
        raise EquivalenceError("Both execution smokes must be complete")
    if parallel_metadata.get("execution_mode") != "site_parallel":
        raise EquivalenceError("Comparison target is not a parallel run")
    if sequential_metadata.get("results_embargoed") is not True or parallel_metadata.get("results_embargoed") is not True:
        raise EquivalenceError("Results embargo is not active")
    scientific_fields = (
        "run_kind", "code_version", "source_hashes", "frozen_input_hashes", "configurations",
        "held_out_sites", "site_to_outer_fold", "protocol", "operator_contract",
        "analysis_protocol", "smoke_override",
    )
    for field in scientific_fields:
        if sequential_metadata.get(field) != parallel_metadata.get(field):
            raise EquivalenceError(f"Scientific metadata mismatch: {field}")

    artifact_report: dict[str, Any] = {}
    for name, fields in SITE_ARTIFACT_FIELDS.items():
        left_rows = read_rows(sequential_dir / name, fields)
        right_rows = read_rows(parallel_dir / name, fields)
        keys = KEY_FIELDS[name]
        left_by_key = {tuple(row[field] for field in keys): row for row in left_rows}
        right_by_key = {tuple(row[field] for field in keys): row for row in right_rows}
        if len(left_by_key) != len(left_rows) or len(right_by_key) != len(right_rows):
            raise EquivalenceError(f"Duplicate equivalence key: {name}")
        if set(left_by_key) != set(right_by_key):
            raise EquivalenceError(f"Artifact coverage mismatch: {name}")
        maximum_difference = 0.0
        for key in left_by_key:
            left = left_by_key[key]
            right = right_by_key[key]
            for field in fields:
                if name == "fit_runtime.csv" and field in RUNTIME_VARIABLE_FIELDS:
                    continue
                maximum_difference = max(
                    maximum_difference,
                    compare_value(left[field], right[field], tolerance),
                )
        artifact_report[name] = {
            "rows_checked": len(left_rows),
            "maximum_absolute_difference": maximum_difference,
            "runtime_resource_fields_excluded": name == "fit_runtime.csv",
        }
    return {
        "state": "passed", "tolerance": tolerance,
        "sequential_run_id": sequential_metadata["run_id"],
        "parallel_run_id": parallel_metadata["run_id"],
        "artifacts": artifact_report, "results_remain_embargoed": True,
        "notice": "Execution-equivalence certificate; no predictive or representation value is reported.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequential-dir", required=True, type=Path)
    parser.add_argument("--parallel-dir", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.tolerance < 0:
        raise ValueError("Tolerance cannot be negative")
    report = compare_runs(args.sequential_dir, args.parallel_dir, tolerance=args.tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
