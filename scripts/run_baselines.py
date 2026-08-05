"""Run the frozen, leakage-safe non-graph ABIDE-I baseline protocol.

This runner implements Step 7.3.  It never creates new splits: it consumes the
frozen outer leave-one-site-out and inner grouped-validation assignments made
in Steps 7.0--7.2.  Every imputer, encoder, scaler, hyperparameter decision,
and fitted classifier is restricted to the appropriate training partition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import signal
import sys
import warnings
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


COVARIATE_NUMERIC = [
    "age_at_scan",
    "mean_framewise_displacement",
    "scan_length_timepoints",
]
COVARIATE_CATEGORICAL = ["sex_code"]
MODEL_NAMES = (
    "covariates_l2_logistic",
    "connectome_elastic_net_logistic",
    "combined_elastic_net_logistic",
)
PREDICTION_FIELDS = [
    "model", "outer_fold", "held_out_site", "subject_id", "site_id", "label_asd",
    "probability_asd", "predicted_asd", "C", "l1_ratio",
]
METRIC_FIELDS = [
    "model", "outer_fold", "held_out_site", "participants", "asd", "control",
    "balanced_accuracy", "auroc", "sensitivity", "specificity", "C", "l1_ratio",
]
TUNING_FIELDS = [
    "model", "outer_fold", "held_out_site", "C", "l1_ratio",
    "inner_mean_site_balanced_accuracy", "inner_sites_scored", "selected",
]
INNER_SITE_FIELDS = [
    "model", "outer_fold", "held_out_site", "inner_validation_fold", "site_id",
    "participants", "balanced_accuracy", "C", "l1_ratio",
]
WARNING_FIELDS = [
    "model", "outer_fold", "held_out_site", "inner_validation_fold", "fit_scope",
    "fit_phase", "C", "l1_ratio", "warning_category", "warning_message",
]
FOLD_ARTIFACT_FIELDS = {
    "predictions.csv": PREDICTION_FIELDS,
    "test_metrics.csv": METRIC_FIELDS,
    "tuning_scores.csv": TUNING_FIELDS,
    "inner_site_scores.csv": INNER_SITE_FIELDS,
    "fit_warnings.csv": WARNING_FIELDS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/baseline_protocol.json")
    parser.add_argument(
        "--inputs", default="configs/baseline_inputs_and_splits.json"
    )
    parser.add_argument(
        "--table", default="data/processed/abide_i_baseline_table.csv"
    )
    parser.add_argument(
        "--connectomes", default="data/processed/abide_i_connectomes_fisher_z.npz"
    )
    parser.add_argument(
        "--outer-splits", default="data/processed/splits/outer_loso_assignments.csv"
    )
    parser.add_argument(
        "--inner-splits", default="data/processed/splits/inner_grouped_assignments.csv"
    )
    parser.add_argument(
        "--output-root", default="outputs/runs/baselines"
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Stable artifact directory name; defaults to a UTC timestamped name.",
    )
    parser.add_argument(
        "--run-kind",
        choices=("smoke", "full"),
        default="full",
        help="A smoke run is explicitly marked as an engineering check, not a result.",
    )
    parser.add_argument(
        "--held-out-sites",
        nargs="+",
        default=None,
        help="Optional subset of frozen held-out sites. Required for smoke runs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
    )
    parser.add_argument(
        "--fast-smoke",
        action="store_true",
        help=(
            "Engineering-only override: retain the frozen split mechanics but use "
            "one lowest-regularization candidate per model and 100 saga iterations "
            "without a retry. Never allowed for a full run."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a named run after verifying its immutable inputs and completed folds.",
    )
    parser.add_argument(
        "--notification-topic-arn",
        default=os.environ.get("BUNN_SNS_TOPIC_ARN"),
        help="Optional SNS topic ARN. Defaults to BUNN_SNS_TOPIC_ARN when set.",
    )
    parser.add_argument(
        "--require-notification",
        action="store_true",
        help="Fail before fitting unless an SNS topic ARN is configured and publishable.",
    )
    parser.add_argument(
        "--code-version",
        default="unknown",
        help="Immutable Git commit or other code identifier recorded in run metadata.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_table(table_path: Path, connectome_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    table = pd.read_csv(table_path, dtype={"subject_id": str, "site_id": str, "sex_code": str})
    required = {
        "connectome_row",
        "subject_id",
        "site_id",
        "label_asd",
        *COVARIATE_NUMERIC,
        *COVARIATE_CATEGORICAL,
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Baseline table lacks columns: {sorted(missing)}")
    table["connectome_row"] = pd.to_numeric(table["connectome_row"], errors="raise")
    table["label_asd"] = pd.to_numeric(table["label_asd"], errors="raise")
    for column in COVARIATE_NUMERIC:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.sort_values("connectome_row", kind="stable").reset_index(drop=True)
    expected_rows = np.arange(len(table), dtype=int)
    if not np.array_equal(table["connectome_row"].to_numpy(dtype=int), expected_rows):
        raise ValueError("connectome_row must be a complete zero-based sequence")
    if table["subject_id"].duplicated().any():
        raise ValueError("Baseline table contains duplicate subject IDs")
    if set(table["label_asd"].unique()) != {0, 1}:
        raise ValueError("Baseline table must contain binary ASD labels 0 and 1")

    with np.load(connectome_path, allow_pickle=False) as arrays:
        edge_features = np.asarray(arrays["edge_features_fisher_z"], dtype=np.float64)
        subject_ids = np.asarray(arrays["subject_id"]).astype(str)
        site_ids = np.asarray(arrays["site_id"]).astype(str)
        labels = np.asarray(arrays["label_asd"], dtype=int)
    if edge_features.ndim != 2 or edge_features.shape[0] != len(table):
        raise ValueError("Connectome edge feature matrix does not align with baseline table")
    if not np.isfinite(edge_features).all():
        raise ValueError("Connectome edge feature matrix contains non-finite values")
    if not np.array_equal(table["subject_id"].to_numpy(), subject_ids):
        raise ValueError("Baseline table subject order differs from connectome artifact")
    if not np.array_equal(table["site_id"].to_numpy(), site_ids):
        raise ValueError("Baseline table site order differs from connectome artifact")
    if not np.array_equal(table["label_asd"].to_numpy(dtype=int), labels):
        raise ValueError("Baseline table labels differ from connectome artifact")
    return table, edge_features


def verify_frozen_hashes(
    inputs_path: Path,
    table_path: Path,
    outer_path: Path,
    inner_path: Path,
) -> dict[str, Any]:
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    expected = {
        "baseline table": inputs["baseline_table"]["sha256"],
        "outer assignments": inputs["splits"]["outer_assignments_sha256"],
        "inner assignments": inputs["splits"]["inner_assignments_sha256"],
    }
    actual = {
        "baseline table": sha256_file(table_path),
        "outer assignments": sha256_file(outer_path),
        "inner assignments": sha256_file(inner_path),
    }
    mismatches = [name for name in expected if expected[name] != actual[name]]
    if mismatches:
        raise ValueError(f"Frozen input hash mismatch: {', '.join(mismatches)}")
    return {"expected": expected, "actual": actual}


def make_feature_frame(
    table: pd.DataFrame, edge_features: np.ndarray
) -> tuple[pd.DataFrame, list[str]]:
    edge_columns = [f"edge_{index:04d}" for index in range(edge_features.shape[1])]
    covariates = table[COVARIATE_NUMERIC + COVARIATE_CATEGORICAL].copy()
    edges = pd.DataFrame(edge_features, columns=edge_columns, index=table.index)
    return pd.concat([covariates, edges], axis=1), edge_columns


def build_pipeline(
    model_name: str,
    parameters: dict[str, float],
    edge_columns: list[str],
    protocol: dict[str, Any],
    random_state: int,
) -> Pipeline:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_name}")
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if model_name in {"covariates_l2_logistic", "combined_elastic_net_logistic"}:
        transformers.extend(
            [
                (
                    "covariate_numeric",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    COVARIATE_NUMERIC,
                ),
                (
                    "covariate_categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ),
                        ]
                    ),
                    COVARIATE_CATEGORICAL,
                ),
            ]
        )
    if model_name in {"connectome_elastic_net_logistic", "combined_elastic_net_logistic"}:
        transformers.append(
            (
                "connectome",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                edge_columns,
            )
        )
    model_spec = protocol["models"][model_name]
    classifier_args: dict[str, Any] = {
        "C": parameters["C"],
        "class_weight": model_spec["class_weight"],
        "solver": model_spec["solver"],
        "max_iter": model_spec.get("max_iter", 10000),
        "random_state": random_state,
    }
    # scikit-learn 1.8+ deprecates the explicit ``penalty`` argument.  The
    # corresponding supported parameterisation is l1_ratio=0 for L2 and a
    # value strictly between zero and one for elastic net.
    if model_name == "covariates_l2_logistic":
        classifier_args["l1_ratio"] = 0.0
    else:
        classifier_args["l1_ratio"] = parameters["l1_ratio"]
    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers, sparse_threshold=0.0)),
            ("classifier", LogisticRegression(**classifier_args)),
        ]
    )


def candidate_parameters(model_name: str, protocol: dict[str, Any]) -> list[dict[str, float]]:
    specification = protocol["models"][model_name]
    candidates: list[dict[str, float]] = []
    if model_name == "covariates_l2_logistic":
        candidates = [{"C": float(value)} for value in specification["C_grid"]]
    else:
        candidates = [
            {"C": float(c_value), "l1_ratio": float(l1_value)}
            for c_value in specification["C_grid"]
            for l1_value in specification["l1_ratio_grid"]
        ]
    return sorted(candidates, key=lambda item: (item["C"], item.get("l1_ratio", -1.0)))


def apply_fast_smoke_overrides(protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return a clearly recorded lightweight configuration for an engineering smoke test."""
    effective = json.loads(json.dumps(protocol))
    override: dict[str, Any] = {
        "candidate_grid": "lowest C and, where applicable, lowest l1_ratio only",
        "elastic_net_max_iter": 100,
        "elastic_net_retry": "disabled",
    }
    for model_name, specification in effective["models"].items():
        specification["C_grid"] = [min(specification["C_grid"])]
        if "l1_ratio_grid" in specification:
            specification["l1_ratio_grid"] = [min(specification["l1_ratio_grid"])]
            specification["max_iter"] = 100
            specification.pop("retry_max_iter", None)
    return effective, override


def fit_with_retry(
    estimator: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    retry_max_iter: int | None,
) -> tuple[Pipeline, list[dict[str, str]]]:
    warning_rows: list[dict[str, str]] = []

    def _fit(phase: str) -> bool:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            estimator.fit(X_train, y_train)
        converged = True
        for warning in captured:
            category = warning.category.__name__
            warning_rows.append(
                {
                    "fit_phase": phase,
                    "warning_category": category,
                    "warning_message": str(warning.message),
                }
            )
            if issubclass(warning.category, ConvergenceWarning):
                converged = False
        return converged

    converged = _fit("initial")
    if not converged and retry_max_iter is not None:
        estimator.set_params(classifier__max_iter=retry_max_iter)
        _fit("retry")
    return estimator, warning_rows


def mean_site_balanced_accuracy(
    labels: np.ndarray, predictions: np.ndarray, sites: np.ndarray
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for site in sorted(set(sites.astype(str))):
        mask = sites.astype(str) == site
        site_labels = labels[mask]
        if set(site_labels.tolist()) != {0, 1}:
            raise ValueError(f"Site {site} does not have both classes for balanced accuracy")
        rows.append(
            {
                "site_id": site,
                "participants": int(mask.sum()),
                "balanced_accuracy": float(balanced_accuracy_score(site_labels, predictions[mask])),
            }
        )
    return float(np.mean([row["balanced_accuracy"] for row in rows])), rows


def select_candidate(
    candidate_rows: list[dict[str, Any]], model_name: str
) -> dict[str, Any]:
    if not candidate_rows:
        raise ValueError(f"No tuning rows for {model_name}")
    ranked = sorted(
        candidate_rows,
        key=lambda row: (
            -round(float(row["inner_mean_site_balanced_accuracy"]), 12),
            float(row["C"]),
            float(row.get("l1_ratio") or -1.0),
        ),
    )
    return ranked[0]


def validate_split_contract(
    table: pd.DataFrame,
    outer_rows: list[dict[str, str]],
    inner_rows: list[dict[str, str]],
    held_out_site: str,
) -> tuple[int, np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    table_ids = set(table["subject_id"])
    outer_by_subject: dict[str, dict[str, str]] = {}
    for row in outer_rows:
        subject_id = row["subject_id"]
        if subject_id in outer_by_subject:
            raise ValueError("A subject appears in multiple outer assignments")
        outer_by_subject[subject_id] = row
    if set(outer_by_subject) != table_ids:
        raise ValueError("Outer assignments do not match baseline table participants")
    outer_rows_for_site = [row for row in outer_rows if row["held_out_site"] == held_out_site]
    if not outer_rows_for_site:
        raise ValueError(f"No outer assignment for held-out site {held_out_site}")
    outer_folds = {int(row["outer_fold"]) for row in outer_rows_for_site}
    if len(outer_folds) != 1:
        raise ValueError(f"Held-out site {held_out_site} maps to multiple outer folds")
    outer_fold = next(iter(outer_folds))
    test_ids = {row["subject_id"] for row in outer_rows_for_site}
    test_mask = table["subject_id"].isin(test_ids).to_numpy()
    if not np.array_equal(test_mask, (table["site_id"] == held_out_site).to_numpy()):
        raise ValueError(f"Outer assignment and site IDs disagree for {held_out_site}")
    test_indices = np.flatnonzero(test_mask)
    train_indices = np.flatnonzero(~test_mask)
    if set(table.loc[test_indices, "label_asd"].astype(int)) != {0, 1}:
        raise ValueError(f"Held-out site {held_out_site} lacks a class")

    rows_for_outer = [row for row in inner_rows if int(row["outer_fold"]) == outer_fold]
    inner_by_subject: dict[str, dict[str, str]] = {}
    for row in rows_for_outer:
        subject_id = row["subject_id"]
        if subject_id in inner_by_subject:
            raise ValueError(f"Duplicate inner assignment for {subject_id} in outer fold {outer_fold}")
        inner_by_subject[subject_id] = row
    train_ids = set(table.loc[train_indices, "subject_id"])
    if set(inner_by_subject) != train_ids:
        raise ValueError(f"Inner assignments do not cover exactly the training set for {held_out_site}")
    if test_ids & set(inner_by_subject):
        raise ValueError(f"Held-out subjects appear in inner assignments for {held_out_site}")

    inner_validation_indices: dict[int, np.ndarray] = {}
    for inner_fold in sorted({int(row["inner_validation_fold"]) for row in rows_for_outer}):
        validation_ids = {
            subject_id
            for subject_id, row in inner_by_subject.items()
            if int(row["inner_validation_fold"]) == inner_fold
        }
        indices = np.flatnonzero(table["subject_id"].isin(validation_ids).to_numpy())
        validation_sites = set(table.loc[indices, "site_id"])
        fitting_sites = set(table.loc[np.setdiff1d(train_indices, indices), "site_id"])
        if validation_sites & fitting_sites:
            raise ValueError(f"Inner fold {inner_fold} leaks a site for {held_out_site}")
        if held_out_site in validation_sites or held_out_site in fitting_sites:
            raise ValueError(f"Outer test site appears in inner split for {held_out_site}")
        if set(table.loc[indices, "label_asd"].astype(int)) != {0, 1}:
            raise ValueError(f"Inner validation fold {inner_fold} lacks a class")
        inner_validation_indices[inner_fold] = indices
    return outer_fold, train_indices, test_indices, inner_validation_indices


def tune_model(
    model_name: str,
    X: pd.DataFrame,
    labels: np.ndarray,
    sites: np.ndarray,
    train_indices: np.ndarray,
    inner_validation_indices: dict[int, np.ndarray],
    edge_columns: list[str],
    protocol: dict[str, Any],
    outer_fold: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows: list[dict[str, Any]] = []
    site_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    retry_max_iter = protocol["models"][model_name].get("retry_max_iter")
    for candidate in candidate_parameters(model_name, protocol):
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "inner_tuning",
                    "model": model_name,
                    "candidate": candidate,
                    "outer_fold": outer_fold,
                }
            )
        print(
            f"tuning outer_fold={outer_fold} model={model_name} "
            f"C={candidate['C']} l1_ratio={candidate.get('l1_ratio', 'n/a')}",
            flush=True,
        )
        all_site_scores: list[float] = []
        for inner_fold, validation_indices in sorted(inner_validation_indices.items()):
            fitting_indices = np.setdiff1d(train_indices, validation_indices)
            estimator = build_pipeline(
                model_name,
                candidate,
                edge_columns,
                protocol,
                random_state=int(protocol["evaluation"]["seed"]) + 1000 * outer_fold + inner_fold,
            )
            estimator, fit_warnings = fit_with_retry(
                estimator,
                X.iloc[fitting_indices],
                labels[fitting_indices],
                retry_max_iter,
            )
            predictions = (estimator.predict_proba(X.iloc[validation_indices])[:, 1] >= protocol["evaluation"]["decision_threshold"]).astype(int)
            mean_score, fold_site_rows = mean_site_balanced_accuracy(
                labels[validation_indices], predictions, sites[validation_indices]
            )
            del mean_score
            for row in fold_site_rows:
                row.update(
                    {
                        "model": model_name,
                        "outer_fold": outer_fold,
                        "inner_validation_fold": inner_fold,
                        "C": candidate["C"],
                        "l1_ratio": candidate.get("l1_ratio", ""),
                    }
                )
                all_site_scores.append(float(row["balanced_accuracy"]))
            site_rows.extend(fold_site_rows)
            for row in fit_warnings:
                row.update(
                    {
                        "model": model_name,
                        "outer_fold": outer_fold,
                        "inner_validation_fold": inner_fold,
                        "fit_scope": "inner",
                        "C": candidate["C"],
                        "l1_ratio": candidate.get("l1_ratio", ""),
                    }
                )
            warning_rows.extend(fit_warnings)
        candidate_rows.append(
            {
                "model": model_name,
                "outer_fold": outer_fold,
                "C": candidate["C"],
                "l1_ratio": candidate.get("l1_ratio", ""),
                "inner_mean_site_balanced_accuracy": float(np.mean(all_site_scores)),
                "inner_sites_scored": len(all_site_scores),
            }
        )
    selected = select_candidate(candidate_rows, model_name)
    for row in candidate_rows:
        row["selected"] = int(row is selected)
    return (
        {"C": float(selected["C"]), **({"l1_ratio": float(selected["l1_ratio"])} if selected.get("l1_ratio", "") != "" else {})},
        candidate_rows,
        site_rows,
        warning_rows,
    )


def test_metric_row(
    model_name: str,
    outer_fold: int,
    held_out_site: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    selected_parameters: dict[str, float],
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "model": model_name,
        "outer_fold": outer_fold,
        "held_out_site": held_out_site,
        "participants": len(labels),
        "asd": int(labels.sum()),
        "control": int((labels == 0).sum()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "sensitivity": float(((predictions == 1) & (labels == 1)).sum() / (labels == 1).sum()),
        "specificity": float(((predictions == 0) & (labels == 0)).sum() / (labels == 0).sum()),
        "C": selected_parameters["C"],
        "l1_ratio": selected_parameters.get("l1_ratio", ""),
    }


def fold_label(outer_fold: int, held_out_site: str) -> str:
    return f"{outer_fold:02d}_{held_out_site}"


def update_status(run_dir: Path, **changes: Any) -> dict[str, Any]:
    path = run_dir / "status.json"
    status = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    status.update(changes)
    status["last_updated_utc"] = utc_now()
    write_json_atomic(path, status)
    return status


def write_fold_artifacts(
    run_dir: Path,
    outer_fold: int,
    held_out_site: str,
    prediction_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    tuning_rows: list[dict[str, Any]],
    inner_site_rows: list[dict[str, Any]],
    warning_rows: list[dict[str, Any]],
) -> Path:
    """Atomically seal a site only after every one of its artifacts is written."""
    label = fold_label(outer_fold, held_out_site)
    folds_dir = run_dir / "folds"
    completed = folds_dir / label
    if completed.exists():
        raise FileExistsError(f"Completed fold directory already exists: {completed}")
    temporary = folds_dir / f".{label}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True, exist_ok=False)
    rows_by_name = {
        "predictions.csv": prediction_rows,
        "test_metrics.csv": metric_rows,
        "tuning_scores.csv": tuning_rows,
        "inner_site_scores.csv": inner_site_rows,
        "fit_warnings.csv": warning_rows,
    }
    for filename, fields in FOLD_ARTIFACT_FIELDS.items():
        write_csv(temporary / filename, rows_by_name[filename], fields)
    completion = {
        "state": "complete",
        "outer_fold": outer_fold,
        "held_out_site": held_out_site,
        "completed_utc": utc_now(),
        "row_counts": {filename: len(rows_by_name[filename]) for filename in FOLD_ARTIFACT_FIELDS},
        "artifact_hashes": {
            filename: sha256_file(temporary / filename) for filename in FOLD_ARTIFACT_FIELDS
        },
    }
    write_json_atomic(temporary / "complete.json", completion)
    os.replace(temporary, completed)
    return completed


def verify_completed_fold(path: Path, outer_fold: int, held_out_site: str) -> bool:
    completion_path = path / "complete.json"
    if not completion_path.exists():
        return False
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if (
        completion.get("state") != "complete"
        or completion.get("outer_fold") != outer_fold
        or completion.get("held_out_site") != held_out_site
    ):
        return False
    hashes = completion.get("artifact_hashes", {})
    for filename in FOLD_ARTIFACT_FIELDS:
        artifact = path / filename
        if not artifact.exists() or hashes.get(filename) != sha256_file(artifact):
            return False
    return True


def read_completed_fold_rows(
    run_dir: Path, outer_fold: int, held_out_site: str
) -> dict[str, list[dict[str, str]]]:
    path = run_dir / "folds" / fold_label(outer_fold, held_out_site)
    if not verify_completed_fold(path, outer_fold, held_out_site):
        raise ValueError(f"Completed fold failed validation: {path}")
    return {filename: read_csv(path / filename) for filename in FOLD_ARTIFACT_FIELDS}


def immutable_run_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metadata[key]
        for key in (
            "run_id",
            "run_kind",
            "code_version",
            "protocol_sha256",
            "frozen_input_hashes",
            "sources",
            "models",
            "smoke_override",
            "held_out_sites",
            "site_to_outer_fold",
            "participants_in_dataset",
            "edge_features",
        )
    }


def initialise_or_resume_run(
    run_dir: Path, metadata: dict[str, Any], resume: bool
) -> list[str]:
    """Create a new run, or validate its immutable contract before resuming."""
    metadata_path = run_dir / "metadata.json"
    if resume:
        if not metadata_path.exists():
            raise FileNotFoundError(f"Cannot resume: missing {metadata_path}")
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if immutable_run_metadata(existing) != immutable_run_metadata(metadata):
            raise ValueError("Cannot resume: immutable run metadata differ from this invocation")
        existing["status"] = "running"
        existing["resumed_utc"] = utc_now()
        write_json_atomic(metadata_path, existing)
        completed: list[str] = []
        for site in metadata["held_out_sites"]:
            outer_fold = next(
                fold for fold, candidate_site in metadata["site_to_outer_fold"].items()
                if candidate_site == site
            )
            fold_path = run_dir / "folds" / fold_label(int(outer_fold), site)
            if verify_completed_fold(fold_path, int(outer_fold), site):
                completed.append(site)
        update_status(
            run_dir,
            state="running",
            resumed_utc=utc_now(),
            pid=os.getpid(),
            completed_sites=completed,
            completed_site_count=len(completed),
            total_sites=len(metadata["held_out_sites"]),
        )
        return completed
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    write_json_atomic(run_dir / "metadata.json", metadata)
    write_json_atomic(
        run_dir / "status.json",
        {
            "state": "running",
            "run_id": metadata["run_id"],
            "pid": os.getpid(),
            "started_utc": metadata["started_utc"],
            "last_updated_utc": utc_now(),
            "completed_sites": [],
            "completed_site_count": 0,
            "total_sites": len(metadata["held_out_sites"]),
            "current_site": None,
            "current_model": None,
            "current_stage": "initialised",
        },
    )
    return []


def publish_sns_notification(
    run_dir: Path,
    topic_arn: str | None,
    subject: str,
    message: str,
) -> dict[str, Any]:
    """Publish a concise alert without embedding AWS credentials in the project."""
    outcome: dict[str, Any] = {
        "attempted_utc": utc_now(),
        "topic_configured": bool(topic_arn),
        "subject": subject,
    }
    if not topic_arn:
        outcome["status"] = "not_configured"
        write_json_atomic(run_dir / "notification.json", outcome)
        return outcome
    try:
        import boto3

        response = boto3.client("sns", region_name=sns_region_from_topic_arn(topic_arn)).publish(
            TopicArn=topic_arn,
            Subject=subject,
            Message=message,
        )
        outcome.update({"status": "published", "message_id": response.get("MessageId")})
    except Exception as error:  # Notification failure must not erase valid results.
        outcome.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
    write_json_atomic(run_dir / "notification.json", outcome)
    return outcome


def sns_region_from_topic_arn(topic_arn: str) -> str:
    """Return the SNS region encoded in a standard topic ARN.

    The managed runner should not depend on a shell-level AWS region variable:
    the notification topic already identifies the correct regional endpoint.
    """
    parts = topic_arn.split(":", maxsplit=5)
    if len(parts) != 6 or parts[0] != "arn" or parts[2] != "sns" or not parts[3]:
        raise ValueError("notification topic must be a standard SNS topic ARN")
    return parts[3]


def notify_terminal_state(args: argparse.Namespace, run_dir: Path, state: str) -> dict[str, Any]:
    if state == "complete":
        subject = f"BuNN baseline COMPLETE: {args.run_id}"
        message = f"Run {args.run_id} completed all requested held-out sites. Check its completion manifest before interpretation."
    else:
        subject = f"BuNN baseline {state.upper()}: {args.run_id}"
        message = f"Run {args.run_id} ended with state {state}. Check status.json and metadata.json before resuming."
    return publish_sns_notification(run_dir, args.notification_topic_arn, subject, message)


def run(args: argparse.Namespace) -> Path:
    if args.run_kind == "smoke" and not args.held_out_sites:
        raise ValueError("Smoke runs require explicit --held-out-sites")
    if args.fast_smoke and args.run_kind != "smoke":
        raise ValueError("--fast-smoke is permitted only with --run-kind smoke")
    if args.resume and not args.run_id:
        raise ValueError("--resume requires an explicit --run-id")
    if args.require_notification and not args.notification_topic_arn:
        raise ValueError("--require-notification requires --notification-topic-arn or BUNN_SNS_TOPIC_ARN")
    protocol_path = Path(args.protocol)
    inputs_path = Path(args.inputs)
    table_path = Path(args.table)
    connectome_path = Path(args.connectomes)
    outer_path = Path(args.outer_splits)
    inner_path = Path(args.inner_splits)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != 1:
        raise ValueError("Unsupported baseline protocol version")
    smoke_override: dict[str, Any] | None = None
    if args.fast_smoke:
        protocol, smoke_override = apply_fast_smoke_overrides(protocol)
    hashes = verify_frozen_hashes(inputs_path, table_path, outer_path, inner_path)
    table, edge_features = parse_table(table_path, connectome_path)
    X, edge_columns = make_feature_frame(table, edge_features)
    labels = table["label_asd"].to_numpy(dtype=int)
    sites = table["site_id"].to_numpy(dtype=str)
    outer_rows = read_csv(outer_path)
    inner_rows = read_csv(inner_path)
    available_sites = sorted({row["held_out_site"] for row in outer_rows})
    selected_sites = args.held_out_sites or available_sites
    if len(set(selected_sites)) != len(selected_sites):
        raise ValueError("Held-out sites must not be repeated")
    unknown_sites = sorted(set(selected_sites) - set(available_sites))
    if unknown_sites:
        raise ValueError(f"Unknown held-out sites: {unknown_sites}")
    if args.run_kind == "full" and set(selected_sites) != set(available_sites):
        raise ValueError("A full run must include every frozen held-out site")
    site_to_outer_fold = {
        str(int(next(row["outer_fold"] for row in outer_rows if row["held_out_site"] == site))): site
        for site in selected_sites
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"baseline_{args.run_kind}_{timestamp}"
    run_dir = Path(args.output_root) / run_id
    metadata = {
        "run_id": run_id,
        "run_kind": args.run_kind,
        "status": "running",
        "started_utc": utc_now(),
        "code_version": args.code_version,
        "protocol_sha256": sha256_file(protocol_path),
        "frozen_input_hashes": hashes,
        "sources": {
            "protocol": str(protocol_path),
            "inputs": str(inputs_path),
            "baseline_table": str(table_path),
            "connectomes": str(connectome_path),
            "outer_splits": str(outer_path),
            "inner_splits": str(inner_path),
        },
        "models": args.models,
        "smoke_override": smoke_override,
        "held_out_sites": selected_sites,
        "site_to_outer_fold": site_to_outer_fold,
        "participants_in_dataset": len(table),
        "edge_features": len(edge_columns),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "interpretation": (
            "Engineering smoke test only; these outputs must not be presented as "
            "the pre-specified full-study result."
            if args.run_kind == "smoke"
            else "Pre-specified full baseline evaluation."
        ),
    }
    completed_sites = initialise_or_resume_run(run_dir, metadata, args.resume)
    if args.require_notification:
        start_alert = publish_sns_notification(
            run_dir,
            args.notification_topic_arn,
            f"BuNN baseline STARTED: {run_id}",
            f"Run {run_id} started with {len(selected_sites)} held-out sites.",
        )
        if start_alert["status"] != "published":
            raise RuntimeError("Required SNS start notification could not be published")

    for held_out_site in selected_sites:
        outer_fold, train_indices, test_indices, inner_validation_indices = validate_split_contract(
            table, outer_rows, inner_rows, held_out_site
        )
        if held_out_site in completed_sites:
            print(f"verified completed held_out_site={held_out_site}; skipping on resume", flush=True)
            continue
        update_status(
            run_dir,
            state="running",
            current_site=held_out_site,
            current_outer_fold=outer_fold,
            current_model=None,
            current_stage="starting_site",
            current_candidate=None,
        )
        print(
            f"starting held_out_site={held_out_site} outer_fold={outer_fold} "
            f"train={len(train_indices)} test={len(test_indices)}",
            flush=True,
        )
        site_prediction_rows: list[dict[str, Any]] = []
        site_metric_rows: list[dict[str, Any]] = []
        site_tuning_rows: list[dict[str, Any]] = []
        site_inner_site_rows: list[dict[str, Any]] = []
        site_warning_rows: list[dict[str, Any]] = []

        def record_tuning_progress(progress: dict[str, Any]) -> None:
            update_status(
                run_dir,
                state="running",
                current_site=held_out_site,
                current_outer_fold=outer_fold,
                current_model=progress["model"],
                current_stage=progress["stage"],
                current_candidate=progress["candidate"],
            )

        for model_name in args.models:
            update_status(
                run_dir,
                state="running",
                current_site=held_out_site,
                current_outer_fold=outer_fold,
                current_model=model_name,
                current_stage="tuning",
                current_candidate=None,
            )
            print(f"starting final model={model_name} held_out_site={held_out_site}", flush=True)
            parameters, model_tuning, model_inner_sites, model_warnings = tune_model(
                model_name,
                X,
                labels,
                sites,
                train_indices,
                inner_validation_indices,
                edge_columns,
                protocol,
                outer_fold,
                progress_callback=record_tuning_progress,
            )
            for row in model_tuning:
                row["held_out_site"] = held_out_site
            for row in model_inner_sites:
                row["held_out_site"] = held_out_site
            for row in model_warnings:
                row["held_out_site"] = held_out_site
            site_tuning_rows.extend(model_tuning)
            site_inner_site_rows.extend(model_inner_sites)
            site_warning_rows.extend(model_warnings)

            update_status(
                run_dir,
                state="running",
                current_site=held_out_site,
                current_outer_fold=outer_fold,
                current_model=model_name,
                current_stage="outer_final_fit",
                current_candidate=parameters,
            )
            estimator = build_pipeline(
                model_name,
                parameters,
                edge_columns,
                protocol,
                random_state=int(protocol["evaluation"]["seed"]) + 1000 * outer_fold + 999,
            )
            estimator, fit_warnings = fit_with_retry(
                estimator,
                X.iloc[train_indices],
                labels[train_indices],
                protocol["models"][model_name].get("retry_max_iter"),
            )
            for row in fit_warnings:
                row.update(
                    {
                        "model": model_name,
                        "outer_fold": outer_fold,
                        "inner_validation_fold": "",
                        "fit_scope": "outer_final",
                        "held_out_site": held_out_site,
                        "C": parameters["C"],
                        "l1_ratio": parameters.get("l1_ratio", ""),
                    }
                )
            site_warning_rows.extend(fit_warnings)
            probabilities = estimator.predict_proba(X.iloc[test_indices])[:, 1]
            site_metric_rows.append(
                test_metric_row(
                    model_name,
                    outer_fold,
                    held_out_site,
                    labels[test_indices],
                    probabilities,
                    float(protocol["evaluation"]["decision_threshold"]),
                    parameters,
                )
            )
            predictions = (probabilities >= protocol["evaluation"]["decision_threshold"]).astype(int)
            for row_index, probability, prediction in zip(test_indices, probabilities, predictions, strict=True):
                site_prediction_rows.append(
                    {
                        "model": model_name,
                        "outer_fold": outer_fold,
                        "held_out_site": held_out_site,
                        "subject_id": table.iloc[row_index]["subject_id"],
                        "site_id": table.iloc[row_index]["site_id"],
                        "label_asd": int(labels[row_index]),
                        "probability_asd": float(probability),
                        "predicted_asd": int(prediction),
                        "C": parameters["C"],
                        "l1_ratio": parameters.get("l1_ratio", ""),
                    }
                )
        write_fold_artifacts(
            run_dir,
            outer_fold,
            held_out_site,
            site_prediction_rows,
            site_metric_rows,
            site_tuning_rows,
            site_inner_site_rows,
            site_warning_rows,
        )
        completed_sites.append(held_out_site)
        update_status(
            run_dir,
            state="running",
            completed_sites=completed_sites,
            completed_site_count=len(completed_sites),
            total_sites=len(selected_sites),
            current_site=None,
            current_outer_fold=None,
            current_model=None,
            current_stage="site_checkpointed",
            current_candidate=None,
        )

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    inner_site_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    for held_out_site in selected_sites:
        outer_fold = int(next(fold for fold, site in site_to_outer_fold.items() if site == held_out_site))
        rows = read_completed_fold_rows(run_dir, outer_fold, held_out_site)
        prediction_rows.extend(rows["predictions.csv"])
        metric_rows.extend(rows["test_metrics.csv"])
        tuning_rows.extend(rows["tuning_scores.csv"])
        inner_site_rows.extend(rows["inner_site_scores.csv"])
        warning_rows.extend(rows["fit_warnings.csv"])

    write_csv(run_dir / "predictions.csv", prediction_rows, PREDICTION_FIELDS)
    write_csv(run_dir / "test_metrics.csv", metric_rows, METRIC_FIELDS)
    write_csv(run_dir / "tuning_scores.csv", tuning_rows, TUNING_FIELDS)
    write_csv(run_dir / "inner_site_scores.csv", inner_site_rows, INNER_SITE_FIELDS)
    write_csv(run_dir / "fit_warnings.csv", warning_rows, WARNING_FIELDS)
    summary = {
        "run_id": run_id,
        "run_kind": args.run_kind,
        "held_out_sites": selected_sites,
        "models": args.models,
        "test_metric_rows": len(metric_rows),
        "prediction_rows": len(prediction_rows),
        "tuning_rows": len(tuning_rows),
        "inner_site_score_rows": len(inner_site_rows),
        "warning_rows": len(warning_rows),
        "selected_parameters": [
            {
                "model": row["model"],
                "held_out_site": row["held_out_site"],
                "C": row["C"],
                "l1_ratio": row["l1_ratio"],
            }
            for row in metric_rows
        ],
        "notice": (
            "Smoke-run artifacts are an engineering validation only and are not "
            "the primary full-study result."
            if args.run_kind == "smoke"
            else "Full frozen protocol completed."
        ),
    }
    write_json_atomic(run_dir / "summary.json", summary)
    metadata["status"] = "complete"
    metadata["completed_utc"] = utc_now()
    metadata["artifact_hashes"] = {
        path.name: sha256_file(path) for path in sorted(run_dir.glob("*.csv"))
    }
    write_json_atomic(run_dir / "metadata.json", metadata)
    update_status(
        run_dir,
        state="complete",
        completed_sites=completed_sites,
        completed_site_count=len(completed_sites),
        total_sites=len(selected_sites),
        current_site=None,
        current_outer_fold=None,
        current_model=None,
        current_stage="completion_audited",
        current_candidate=None,
    )
    notification = notify_terminal_state(args, run_dir, "complete")
    metadata["notification_status"] = notification["status"]
    write_json_atomic(run_dir / "metadata.json", metadata)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return run_dir


def mark_explicit_run_failed(args: argparse.Namespace, error: Exception) -> None:
    """Avoid leaving an explicitly named interrupted run labelled ``running``."""
    if not args.run_id:
        return
    metadata_path = Path(args.output_root) / args.run_id / "metadata.json"
    if not metadata_path.exists():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["status"] = "interrupted" if isinstance(error, InterruptedError) else "failed"
    metadata["failed_utc"] = datetime.now(UTC).isoformat()
    metadata["failure_type"] = type(error).__name__
    metadata["failure_message"] = str(error)
    write_json_atomic(metadata_path, metadata)
    state = metadata["status"]
    update_status(
        metadata_path.parent,
        state=state,
        current_stage="terminal_error",
        failure_type=type(error).__name__,
        failure_message=str(error),
    )
    notification = notify_terminal_state(args, metadata_path.parent, state)
    metadata["notification_status"] = notification["status"]
    write_json_atomic(metadata_path, metadata)


def handle_sigterm(_signal: int, _frame: Any) -> None:
    raise InterruptedError("received SIGTERM")


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        run(args)
    except Exception as error:
        mark_explicit_run_failed(args, error)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
