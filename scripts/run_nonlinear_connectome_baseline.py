"""Run the frozen post-hoc nonlinear connectome baseline.

The runner consumes the existing ABIDE-I cohort and split files. It seals one
outer site at a time and never prints held-out scores, which preserves the
score-blind audit boundary used by the earlier experiments.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:  # Package import in tests; direct import when executed as a script.
    from scripts.run_baselines import (
        parse_table,
        read_csv,
        sha256_file,
        validate_split_contract,
        verify_frozen_hashes,
        publish_sns_notification,
        write_csv,
        write_json_atomic,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by command-line use
    from run_baselines import (
        parse_table,
        read_csv,
        sha256_file,
        validate_split_contract,
        verify_frozen_hashes,
        publish_sns_notification,
        write_csv,
        write_json_atomic,
    )


PREDICTION_FIELDS = [
    "model", "outer_fold", "held_out_site", "subject_id", "site_id",
    "label_asd", "decision_score", "predicted_asd", "C",
    "gamma_multiplier", "gamma",
]
METRIC_FIELDS = [
    "model", "outer_fold", "held_out_site", "participants", "asd", "control",
    "balanced_accuracy", "auroc", "sensitivity", "specificity", "C",
    "gamma_multiplier", "gamma",
]
TUNING_FIELDS = [
    "model", "outer_fold", "held_out_site", "C", "gamma_multiplier", "gamma",
    "inner_mean_site_balanced_accuracy", "inner_sites_scored", "selected",
]
INNER_SITE_FIELDS = [
    "model", "outer_fold", "held_out_site", "inner_validation_fold", "site_id",
    "participants", "balanced_accuracy", "C", "gamma_multiplier", "gamma",
]
FOLD_FILES = {
    "predictions.csv": PREDICTION_FIELDS,
    "test_metrics.csv": METRIC_FIELDS,
    "tuning_scores.csv": TUNING_FIELDS,
    "inner_site_scores.csv": INNER_SITE_FIELDS,
}
MODEL_NAME = "connectome_rbf_svm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", default="configs/extensions/nonlinear_connectome_baseline_v1.json"
    )
    parser.add_argument("--inputs", default="configs/baseline_inputs_and_splits.json")
    parser.add_argument("--table", default="data/processed/abide_i_baseline_table.csv")
    parser.add_argument(
        "--connectomes", default="data/processed/abide_i_connectomes_fisher_z.npz"
    )
    parser.add_argument(
        "--outer-splits", default="data/processed/splits/outer_loso_assignments.csv"
    )
    parser.add_argument(
        "--inner-splits", default="data/processed/splits/inner_grouped_assignments.csv"
    )
    parser.add_argument("--output-root", default="outputs/runs/nonlinear_baseline")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--held-out-sites", nargs="+", default=None)
    parser.add_argument("--run-kind", choices=("timing_smoke", "full"), default="full")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--code-version", default="unknown")
    parser.add_argument(
        "--notification-topic-arn", default=os.environ.get("BUNN_SNS_TOPIC_ARN")
    )
    parser.add_argument("--require-notification", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def candidates(protocol: dict[str, Any]) -> list[dict[str, float]]:
    model = protocol["model"]
    feature_count = int(model["feature_count"])
    return [
        {
            "C": float(c_value),
            "gamma_multiplier": float(multiplier),
            "gamma": float(multiplier) / feature_count,
        }
        for c_value in model["C_grid"]
        for multiplier in model["gamma_multipliers_over_feature_count"]
    ]


def build_estimator(candidate: dict[str, float], protocol: dict[str, Any]) -> Pipeline:
    model = protocol["model"]
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    C=candidate["C"],
                    gamma=candidate["gamma"],
                    kernel=model["kernel"],
                    class_weight=model["class_weight"],
                ),
            ),
        ]
    )


def site_balanced_accuracy(
    labels: np.ndarray, predictions: np.ndarray, sites: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for site in sorted(set(sites.astype(str))):
        mask = sites.astype(str) == site
        site_labels = labels[mask]
        if set(site_labels.tolist()) != {0, 1}:
            raise ValueError(f"Site {site} does not contain both classes")
        rows.append(
            {
                "site_id": site,
                "participants": int(mask.sum()),
                "balanced_accuracy": float(
                    balanced_accuracy_score(site_labels, predictions[mask])
                ),
            }
        )
    return rows


def select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No tuning candidates were evaluated")
    return sorted(
        rows,
        key=lambda row: (
            -round(float(row["inner_mean_site_balanced_accuracy"]), 12),
            float(row["C"]),
            float(row["gamma_multiplier"]),
        ),
    )[0]


def tune(
    features: np.ndarray,
    labels: np.ndarray,
    sites: np.ndarray,
    train_indices: np.ndarray,
    inner_validation_indices: dict[int, np.ndarray],
    protocol: dict[str, Any],
    outer_fold: int,
    held_out_site: str,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    tuning_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    for candidate in candidates(protocol):
        print(
            f"outer_fold={outer_fold} site={held_out_site} stage=inner_tuning "
            f"C={candidate['C']} gamma_multiplier={candidate['gamma_multiplier']}",
            flush=True,
        )
        candidate_site_scores: list[float] = []
        for inner_fold, validation_indices in sorted(inner_validation_indices.items()):
            fitting_indices = np.setdiff1d(train_indices, validation_indices)
            estimator = build_estimator(candidate, protocol)
            estimator.fit(features[fitting_indices], labels[fitting_indices])
            scores = estimator.decision_function(features[validation_indices])
            predictions = (scores >= 0.0).astype(int)
            rows = site_balanced_accuracy(
                labels[validation_indices], predictions, sites[validation_indices]
            )
            for row in rows:
                row.update(
                    {
                        "model": MODEL_NAME,
                        "outer_fold": outer_fold,
                        "held_out_site": held_out_site,
                        "inner_validation_fold": inner_fold,
                        **candidate,
                    }
                )
                candidate_site_scores.append(float(row["balanced_accuracy"]))
            inner_rows.extend(rows)
        tuning_rows.append(
            {
                "model": MODEL_NAME,
                "outer_fold": outer_fold,
                "held_out_site": held_out_site,
                **candidate,
                "inner_mean_site_balanced_accuracy": float(np.mean(candidate_site_scores)),
                "inner_sites_scored": len(candidate_site_scores),
                "selected": 0,
            }
        )
    selected = select_candidate(tuning_rows)
    selected["selected"] = 1
    return (
        {
            "C": float(selected["C"]),
            "gamma_multiplier": float(selected["gamma_multiplier"]),
            "gamma": float(selected["gamma"]),
        },
        tuning_rows,
        inner_rows,
    )


def evaluate_site(
    features: np.ndarray,
    labels: np.ndarray,
    sites: np.ndarray,
    subject_ids: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    candidate: dict[str, float],
    protocol: dict[str, Any],
    outer_fold: int,
    held_out_site: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    estimator = build_estimator(candidate, protocol)
    estimator.fit(features[train_indices], labels[train_indices])
    scores = np.asarray(estimator.decision_function(features[test_indices]), dtype=float)
    predictions = (scores >= 0.0).astype(int)
    test_labels = labels[test_indices]
    prediction_rows = [
        {
            "model": MODEL_NAME,
            "outer_fold": outer_fold,
            "held_out_site": held_out_site,
            "subject_id": str(subject_ids[index]),
            "site_id": str(sites[index]),
            "label_asd": int(labels[index]),
            "decision_score": float(score),
            "predicted_asd": int(prediction),
            **candidate,
        }
        for index, score, prediction in zip(test_indices, scores, predictions, strict=True)
    ]
    metric_rows = [
        {
            "model": MODEL_NAME,
            "outer_fold": outer_fold,
            "held_out_site": held_out_site,
            "participants": len(test_indices),
            "asd": int(test_labels.sum()),
            "control": int((test_labels == 0).sum()),
            "balanced_accuracy": float(balanced_accuracy_score(test_labels, predictions)),
            "auroc": float(roc_auc_score(test_labels, scores)),
            "sensitivity": float(predictions[test_labels == 1].mean()),
            "specificity": float((predictions[test_labels == 0] == 0).mean()),
            **candidate,
        }
    ]
    return prediction_rows, metric_rows


def fold_label(outer_fold: int, held_out_site: str) -> str:
    return f"{outer_fold:02d}_{held_out_site}"


def seal_fold(
    run_dir: Path,
    outer_fold: int,
    held_out_site: str,
    rows: dict[str, list[dict[str, Any]]],
) -> None:
    completed = run_dir / "folds" / fold_label(outer_fold, held_out_site)
    if completed.exists():
        raise FileExistsError(f"Fold already exists: {completed}")
    temporary = completed.parent / f".{completed.name}.tmp-{os.getpid()}"
    temporary.mkdir(parents=True, exist_ok=False)
    for filename, fields in FOLD_FILES.items():
        write_csv(temporary / filename, rows[filename], fields)
    certificate = {
        "state": "complete",
        "outer_fold": outer_fold,
        "held_out_site": held_out_site,
        "completed_utc": utc_now(),
        "row_counts": {name: len(rows[name]) for name in FOLD_FILES},
        "artifact_hashes": {
            name: sha256_file(temporary / name) for name in FOLD_FILES
        },
    }
    write_json_atomic(temporary / "complete.json", certificate)
    os.replace(temporary, completed)


def verified_completed_fold(path: Path, outer_fold: int, held_out_site: str) -> bool:
    certificate_path = path / "complete.json"
    if not certificate_path.exists():
        return False
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    if (
        certificate.get("state") != "complete"
        or certificate.get("outer_fold") != outer_fold
        or certificate.get("held_out_site") != held_out_site
    ):
        return False
    return all(
        (path / name).exists()
        and certificate["artifact_hashes"].get(name) == sha256_file(path / name)
        for name in FOLD_FILES
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = Path(args.protocol)
    inputs_path = Path(args.inputs)
    table_path = Path(args.table)
    connectome_path = Path(args.connectomes)
    outer_path = Path(args.outer_splits)
    inner_path = Path(args.inner_splits)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    frozen_hashes = verify_frozen_hashes(inputs_path, table_path, outer_path, inner_path)
    table, features = parse_table(table_path, connectome_path)
    if features.shape != (
        int(protocol["cohort"]["participants"]),
        int(protocol["model"]["feature_count"]),
    ):
        raise ValueError("Connectome feature shape differs from the frozen protocol")
    outer_rows = read_csv(outer_path)
    inner_rows = read_csv(inner_path)
    all_sites = sorted(table["site_id"].astype(str).unique())
    selected_sites = list(args.held_out_sites or all_sites)
    if args.run_kind == "timing_smoke" and len(selected_sites) != 1:
        raise ValueError("A timing smoke must contain exactly one held-out site")
    if not set(selected_sites).issubset(all_sites):
        raise ValueError("Unknown held-out site requested")

    site_to_fold: dict[str, int] = {}
    for site in selected_sites:
        rows = [row for row in outer_rows if row["held_out_site"] == site]
        site_to_fold[site] = int(rows[0]["outer_fold"])
    run_dir = Path(args.output_root) / args.run_id
    metadata = {
        "run_id": args.run_id,
        "run_kind": args.run_kind,
        "status": "running",
        "started_utc": utc_now(),
        "code_version": args.code_version,
        "protocol_sha256": sha256_file(protocol_path),
        "protocol_path": str(protocol_path),
        "frozen_input_hashes": frozen_hashes,
        "held_out_sites": selected_sites,
        "site_to_outer_fold": site_to_fold,
        "participants": len(table),
        "feature_count": features.shape[1],
    }
    metadata_path = run_dir / "metadata.json"
    if args.resume:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        immutable = [
            "run_id", "run_kind", "code_version", "protocol_sha256",
            "frozen_input_hashes", "held_out_sites", "site_to_outer_fold",
            "participants", "feature_count",
        ]
        if any(existing.get(key) != metadata.get(key) for key in immutable):
            raise ValueError("Resume metadata differs from the immutable run contract")
    else:
        if run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite run: {run_dir}")
        (run_dir / "folds").mkdir(parents=True)
        write_json_atomic(metadata_path, metadata)

    start_notification = publish_sns_notification(
        run_dir,
        args.notification_topic_arn,
        f"BuNN nonlinear baseline STARTED: {args.run_id}",
        f"Run {args.run_id} started with {len(selected_sites)} held-out site(s).",
    )
    if args.require_notification and start_notification["status"] != "published":
        raise RuntimeError("Required SNS start notification could not be published")

    labels = table["label_asd"].to_numpy(dtype=int)
    sites = table["site_id"].astype(str).to_numpy()
    subject_ids = table["subject_id"].astype(str).to_numpy()
    completed_sites: list[str] = []
    for held_out_site in selected_sites:
        outer_fold = site_to_fold[held_out_site]
        fold_dir = run_dir / "folds" / fold_label(outer_fold, held_out_site)
        if args.resume and verified_completed_fold(fold_dir, outer_fold, held_out_site):
            completed_sites.append(held_out_site)
            continue
        outer_fold, train_indices, test_indices, inner_validation = validate_split_contract(
            table, outer_rows, inner_rows, held_out_site
        )
        candidate, tuning_rows, inner_site_rows = tune(
            features, labels, sites, train_indices, inner_validation, protocol,
            outer_fold, held_out_site,
        )
        prediction_rows, metric_rows = evaluate_site(
            features, labels, sites, subject_ids, train_indices, test_indices,
            candidate, protocol, outer_fold, held_out_site,
        )
        seal_fold(
            run_dir,
            outer_fold,
            held_out_site,
            {
                "predictions.csv": prediction_rows,
                "test_metrics.csv": metric_rows,
                "tuning_scores.csv": tuning_rows,
                "inner_site_scores.csv": inner_site_rows,
            },
        )
        completed_sites.append(held_out_site)
        write_json_atomic(
            run_dir / "status.json",
            {
                "state": "running",
                "completed_sites": completed_sites,
                "completed_site_count": len(completed_sites),
                "expected_site_count": len(selected_sites),
                "last_updated_utc": utc_now(),
            },
        )
        print(
            f"sealed outer_fold={outer_fold} site={held_out_site} "
            f"completed={len(completed_sites)}/{len(selected_sites)}",
            flush=True,
        )
    metadata["status"] = "awaiting_score_blind_audit"
    metadata["completed_utc"] = utc_now()
    write_json_atomic(metadata_path, metadata)
    write_json_atomic(
        run_dir / "status.json",
        {
            "state": "awaiting_score_blind_audit",
            "completed_sites": completed_sites,
            "completed_site_count": len(completed_sites),
            "expected_site_count": len(selected_sites),
            "last_updated_utc": utc_now(),
        },
    )
    terminal_notification = publish_sns_notification(
        run_dir,
        args.notification_topic_arn,
        f"BuNN nonlinear baseline AWAITING AUDIT: {args.run_id}",
        f"Run {args.run_id} sealed all {len(completed_sites)} site(s) and is awaiting its score-blind audit.",
    )
    if args.require_notification and terminal_notification["status"] != "published":
        raise RuntimeError("Required SNS terminal notification could not be published")
    return {
        "run_id": args.run_id,
        "state": "awaiting_score_blind_audit",
        "sealed_sites": len(completed_sites),
    }


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
