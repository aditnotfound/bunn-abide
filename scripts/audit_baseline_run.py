"""Audit a completed baseline run without reporting predictive performance.

This is the independent Step 7.5 completion gate. It verifies sealed-fold
hashes, aggregate hashes, schemas, coverage, numerical validity, and frozen
protocol counts. Its report intentionally contains counts and pass/fail checks
only: it never emits a prediction, metric value, or model comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Support both documented invocation forms: ``python -m scripts...`` and
# ``python scripts/audit_baseline_run.py`` from the repository root.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_baselines import (
    FOLD_ARTIFACT_FIELDS,
    INNER_SITE_FIELDS,
    METRIC_FIELDS,
    MODEL_NAMES,
    PREDICTION_FIELDS,
    TUNING_FIELDS,
    WARNING_FIELDS,
    fold_label,
    sha256_file,
    verify_completed_fold,
)


ROOT_ARTIFACT_FIELDS = {
    "predictions.csv": PREDICTION_FIELDS,
    "test_metrics.csv": METRIC_FIELDS,
    "tuning_scores.csv": TUNING_FIELDS,
    "inner_site_scores.csv": INNER_SITE_FIELDS,
    "fit_warnings.csv": WARNING_FIELDS,
}


@dataclass(frozen=True)
class AuditFailure(Exception):
    """A score-blind integrity failure, expressed as stable check codes."""

    codes: tuple[str, ...]

    def __str__(self) -> str:
        return "Baseline integrity audit failed: " + ", ".join(self.codes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Completed baseline run directory.")
    parser.add_argument(
        "--protocol",
        default="configs/baseline_protocol.json",
        help="Frozen baseline protocol used to derive expected candidate counts.",
    )
    parser.add_argument(
        "--baseline-table",
        default="data/processed/abide_i_baseline_table.csv",
        help="Frozen participant table used for subject/site/label coverage checks.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional path for the score-blind JSON report. The completed run is never modified.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path, expected_fields: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        actual_fields = reader.fieldnames or []
        if actual_fields != expected_fields:
            raise ValueError(f"schema:{path.name}")
        return list(reader)


def as_int(value: str, check: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(check) from error


def as_finite_float(value: str, check: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(check) from error
    if not math.isfinite(numeric):
        raise ValueError(check)
    return numeric


def candidate_count(model_spec: dict[str, Any]) -> int:
    c_values = model_spec.get("C_grid", [])
    l1_values = model_spec.get("l1_ratio_grid")
    return len(c_values) * (len(l1_values) if l1_values is not None else 1)


def source_path(path_value: str, project_root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def add_error(errors: set[str], callback: Any) -> None:
    try:
        callback()
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        errors.add("audit.internal")


def validate_frozen_inputs(
    metadata: dict[str, Any], project_root: Path, errors: set[str]
) -> None:
    """Check the files whose hashes were frozen at launch, if they are available."""

    sources = metadata.get("sources", {})
    frozen = metadata.get("frozen_input_hashes", {})
    expected_sources = {
        "baseline table": "baseline_table",
        "outer assignments": "outer_splits",
        "inner assignments": "inner_splits",
    }
    for hash_name, source_name in expected_sources.items():
        expected = frozen.get("expected", {}).get(hash_name)
        source = sources.get(source_name)
        if not expected or not source:
            errors.add("frozen_inputs.metadata")
            continue
        path = source_path(source, project_root)
        if not path.exists() or sha256_file(path) != expected:
            errors.add("frozen_inputs.hash")


def validate_fold_artifacts(
    run_dir: Path,
    held_out_sites: list[str],
    site_to_outer_fold: dict[str, str],
    errors: set[str],
) -> dict[str, list[dict[str, str]]]:
    aggregate: dict[str, list[dict[str, str]]] = {name: [] for name in FOLD_ARTIFACT_FIELDS}
    expected_labels: set[str] = set()

    for site in held_out_sites:
        fold = next((int(key) for key, value in site_to_outer_fold.items() if value == site), None)
        if fold is None:
            errors.add("fold_mapping")
            continue
        label = fold_label(fold, site)
        expected_labels.add(label)
        fold_dir = run_dir / "folds" / label
        if not verify_completed_fold(fold_dir, fold, site):
            errors.add("fold_manifest_or_hash")
            continue
        try:
            completion = read_json(fold_dir / "complete.json")
            for filename, fields in FOLD_ARTIFACT_FIELDS.items():
                rows = read_csv(fold_dir / filename, fields)
                if completion.get("row_counts", {}).get(filename) != len(rows):
                    errors.add("fold_row_count")
                aggregate[filename].extend(rows)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            errors.add("fold_schema_or_read")

    actual_labels = {
        path.parent.name for path in (run_dir / "folds").glob("*/complete.json")
    }
    if actual_labels != expected_labels:
        errors.add("fold_coverage")
    return aggregate


def validate_root_artifacts(
    run_dir: Path,
    metadata: dict[str, Any],
    expected_rows: dict[str, list[dict[str, str]]],
    errors: set[str],
) -> dict[str, list[dict[str, str]]]:
    root_rows: dict[str, list[dict[str, str]]] = {}
    expected_hashes = metadata.get("artifact_hashes", {})
    for filename, fields in ROOT_ARTIFACT_FIELDS.items():
        path = run_dir / filename
        try:
            rows = read_csv(path, fields)
            root_rows[filename] = rows
            if expected_hashes.get(filename) != sha256_file(path):
                errors.add("root_artifact_hash")
            if rows != expected_rows[filename]:
                errors.add("root_aggregate_matches_folds")
        except (OSError, ValueError, KeyError):
            errors.add("root_artifact_schema_or_read")
    return root_rows


def validate_predictions(
    rows: list[dict[str, str]],
    expected_subjects: dict[str, tuple[str, int]],
    held_out_sites: set[str],
    models: tuple[str, ...],
    errors: set[str],
) -> None:
    seen: set[tuple[str, str]] = set()
    models_by_subject: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        try:
            subject_id = row["subject_id"]
            model = row["model"]
            held_out_site = row["held_out_site"]
            source_site = row["site_id"]
            label = as_int(row["label_asd"], "prediction_numeric")
            predicted = as_int(row["predicted_asd"], "prediction_numeric")
            probability = as_finite_float(row["probability_asd"], "prediction_numeric")
            if not 0.0 <= probability <= 1.0 or label not in {0, 1} or predicted not in {0, 1}:
                errors.add("prediction_range")
            if model not in models or held_out_site not in held_out_sites:
                errors.add("prediction_model_or_site")
            if source_site != held_out_site:
                errors.add("prediction_site_alignment")
            expected = expected_subjects.get(subject_id)
            if expected is None or expected != (source_site, label):
                errors.add("prediction_subject_alignment")
            key = (subject_id, model)
            if key in seen:
                errors.add("prediction_duplicate")
            seen.add(key)
            models_by_subject[subject_id].add(model)
        except (KeyError, ValueError):
            errors.add("prediction_schema_or_numeric")

    if len(rows) != len(expected_subjects) * len(models):
        errors.add("prediction_row_count")
    if set(models_by_subject) != set(expected_subjects):
        errors.add("prediction_subject_coverage")
    if any(subject_models != set(models) for subject_models in models_by_subject.values()):
        errors.add("prediction_model_coverage")


def validate_metrics_and_selection(
    metric_rows: list[dict[str, str]],
    tuning_rows: list[dict[str, str]],
    held_out_sites: set[str],
    models: tuple[str, ...],
    protocol: dict[str, Any],
    errors: set[str],
) -> None:
    selected: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    tuning_by_pair: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in tuning_rows:
        try:
            pair = (row["held_out_site"], row["model"])
            as_finite_float(row["C"], "tuning_numeric")
            as_finite_float(row["inner_mean_site_balanced_accuracy"], "tuning_numeric")
            as_int(row["inner_sites_scored"], "tuning_numeric")
            if row["model"] not in models or row["held_out_site"] not in held_out_sites:
                errors.add("tuning_model_or_site")
            if row["selected"] not in {"0", "1"}:
                errors.add("tuning_selected_value")
            tuning_by_pair[pair].append(row)
            if row["selected"] == "1":
                selected[pair].append(row)
        except (KeyError, ValueError):
            errors.add("tuning_schema_or_numeric")

    expected_pairs = {(site, model) for site in held_out_sites for model in models}
    if set(tuning_by_pair) != expected_pairs:
        errors.add("tuning_pair_coverage")
    for pair in expected_pairs:
        model_spec = protocol.get("models", {}).get(pair[1], {})
        if len(tuning_by_pair[pair]) != candidate_count(model_spec):
            errors.add("tuning_candidate_count")
        if len(selected[pair]) != 1:
            errors.add("tuning_selected_count")

    metric_pairs: set[tuple[str, str]] = set()
    for row in metric_rows:
        try:
            pair = (row["held_out_site"], row["model"])
            if pair in metric_pairs:
                errors.add("metric_duplicate")
            metric_pairs.add(pair)
            participants = as_int(row["participants"], "metric_numeric")
            asd = as_int(row["asd"], "metric_numeric")
            control = as_int(row["control"], "metric_numeric")
            if participants <= 0 or participants != asd + control:
                errors.add("metric_class_counts")
            for field in ("balanced_accuracy", "auroc", "sensitivity", "specificity", "C"):
                as_finite_float(row[field], "metric_numeric")
            if pair not in expected_pairs:
                errors.add("metric_model_or_site")
            chosen = selected.get(pair, [])
            if len(chosen) == 1 and (row["C"], row["l1_ratio"]) != (chosen[0]["C"], chosen[0]["l1_ratio"]):
                errors.add("metric_selected_parameters")
        except (KeyError, ValueError):
            errors.add("metric_schema_or_numeric")
    if metric_pairs != expected_pairs:
        errors.add("metric_pair_coverage")


def validate_inner_scores(
    rows: list[dict[str, str]],
    held_out_sites: set[str],
    models: tuple[str, ...],
    protocol: dict[str, Any],
    errors: set[str],
) -> None:
    expected_count = sum(candidate_count(protocol["models"][model]) for model in models)
    expected_count *= len(held_out_sites) * (len(held_out_sites) - 1)
    if len(rows) != expected_count:
        errors.add("inner_site_row_count")
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for row in rows:
        try:
            key = (
                row["held_out_site"], row["model"], row["inner_validation_fold"],
                row["site_id"], row["C"], row["l1_ratio"],
            )
            if key in seen:
                errors.add("inner_site_duplicate")
            seen.add(key)
            if row["held_out_site"] not in held_out_sites or row["model"] not in models:
                errors.add("inner_site_model_or_site")
            if row["site_id"] == row["held_out_site"]:
                errors.add("inner_site_leakage")
            as_int(row["participants"], "inner_site_numeric")
            as_finite_float(row["balanced_accuracy"], "inner_site_numeric")
            as_finite_float(row["C"], "inner_site_numeric")
        except (KeyError, ValueError):
            errors.add("inner_site_schema_or_numeric")


def validate_warning_rows(rows: Iterable[dict[str, str]], errors: set[str]) -> None:
    for row in rows:
        if set(row) != set(WARNING_FIELDS):
            errors.add("warning_schema")


def audit_run(
    run_dir: Path,
    protocol_path: Path,
    baseline_table_path: Path,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Return a score-blind integrity report or raise ``AuditFailure``."""

    run_dir = run_dir.resolve()
    project_root = (project_root or Path.cwd()).resolve()
    errors: set[str] = set()
    try:
        metadata = read_json(run_dir / "metadata.json")
        status = read_json(run_dir / "status.json")
        protocol = read_json(protocol_path)
    except (OSError, json.JSONDecodeError):
        raise AuditFailure(("run_metadata_or_status",)) from None

    held_out_sites = tuple(metadata.get("held_out_sites", []))
    models = tuple(metadata.get("models", []))
    site_to_outer_fold = metadata.get("site_to_outer_fold", {})
    participant_count = metadata.get("participants_in_dataset")
    if (
        metadata.get("status") != "complete"
        or status.get("state") != "complete"
        or status.get("completed_site_count") != len(held_out_sites)
        or set(status.get("completed_sites", [])) != set(held_out_sites)
        or not held_out_sites
        or not models
        or any(model not in MODEL_NAMES for model in models)
        or set(models) - set(protocol.get("models", {}))
    ):
        errors.add("run_completion_contract")

    validate_frozen_inputs(metadata, project_root, errors)
    fold_rows = validate_fold_artifacts(run_dir, list(held_out_sites), site_to_outer_fold, errors)
    root_rows = validate_root_artifacts(run_dir, metadata, fold_rows, errors)

    try:
        table_rows = read_csv(baseline_table_path, [
            "connectome_row", "subject_id", "site_id", "label_asd", "age_at_scan", "sex_code",
            "mean_framewise_displacement", "scan_length_timepoints",
        ])
        expected_subjects = {
            row["subject_id"]: (row["site_id"], as_int(row["label_asd"], "table_numeric"))
            for row in table_rows
        }
        if len(expected_subjects) != len(table_rows) or len(table_rows) != participant_count:
            errors.add("baseline_table_subject_contract")
    except (OSError, ValueError, KeyError):
        expected_subjects = {}
        errors.add("baseline_table_schema_or_read")

    if root_rows:
        validate_predictions(
            root_rows.get("predictions.csv", []), expected_subjects, set(held_out_sites), models, errors
        )
        validate_metrics_and_selection(
            root_rows.get("test_metrics.csv", []), root_rows.get("tuning_scores.csv", []),
            set(held_out_sites), models, protocol, errors
        )
        validate_inner_scores(
            root_rows.get("inner_site_scores.csv", []), set(held_out_sites), models, protocol, errors
        )
        validate_warning_rows(root_rows.get("fit_warnings.csv", []), errors)

    report = {
        "audit": "baseline_run_integrity_v1",
        "run_id": metadata.get("run_id"),
        "status": "passed" if not errors else "failed",
        "participants": participant_count,
        "held_out_sites": len(held_out_sites),
        "models": list(models),
        "row_counts": {name: len(rows) for name, rows in root_rows.items()},
        "warning_rows": len(root_rows.get("fit_warnings.csv", [])),
        "checks": {
            "fold_manifests": len(held_out_sites),
            "root_artifacts": len(ROOT_ARTIFACT_FIELDS),
            "participant_model_units": (participant_count or 0) * len(models),
        },
        "failures": sorted(errors),
    }
    if errors:
        raise AuditFailure(tuple(sorted(errors)))
    return report


def main() -> None:
    args = parse_args()
    try:
        report = audit_run(
            Path(args.run_dir), Path(args.protocol), Path(args.baseline_table), Path.cwd()
        )
    except AuditFailure as error:
        report = {
            "audit": "baseline_run_integrity_v1",
            "status": "failed",
            "failures": list(error.codes),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(1) from None

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
