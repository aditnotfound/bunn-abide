"""Score-blind integrity audit for the nonlinear connectome baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

try:  # Package import in tests; direct import when executed as a script.
    from scripts.run_baselines import read_csv, sha256_file, write_json_atomic
    from scripts.run_nonlinear_connectome_baseline import verified_completed_fold
except ModuleNotFoundError:  # pragma: no cover - exercised by command-line use
    from run_baselines import read_csv, sha256_file, write_json_atomic
    from run_nonlinear_connectome_baseline import verified_completed_fold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/extensions/nonlinear_connectome_baseline_v1.json"),
    )
    return parser.parse_args()


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=tolerance))


def audit(run_dir: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["protocol_sha256"] != sha256_file(protocol_path):
        raise ValueError("Protocol hash differs from the run metadata")
    sites = list(metadata["held_out_sites"])
    if metadata["run_kind"] == "full" and len(sites) != int(protocol["cohort"]["sites"]):
        raise ValueError("A full run does not contain every frozen site")

    prediction_count = 0
    metric_count = 0
    tuning_count = 0
    inner_fit_count = 0
    subject_ids: set[str] = set()
    fold_hashes: dict[str, str] = {}
    for site in sites:
        outer_fold = int(metadata["site_to_outer_fold"][site])
        fold_dir = run_dir / "folds" / f"{outer_fold:02d}_{site}"
        if not verified_completed_fold(fold_dir, outer_fold, site):
            raise ValueError(f"Fold seal failed validation: {site}")
        predictions = read_csv(fold_dir / "predictions.csv")
        metrics = read_csv(fold_dir / "test_metrics.csv")
        tuning = read_csv(fold_dir / "tuning_scores.csv")
        inner = read_csv(fold_dir / "inner_site_scores.csv")
        if len(metrics) != 1:
            raise ValueError(f"Expected one test metric row for {site}")
        if len(tuning) != int(protocol["expected_counts"]["candidates_per_outer_site"]):
            raise ValueError(f"Candidate coverage differs for {site}")
        if sum(int(row["selected"]) for row in tuning) != 1:
            raise ValueError(f"Expected one selected candidate for {site}")
        if not predictions or any(row["held_out_site"] != site for row in predictions):
            raise ValueError(f"Prediction site mismatch for {site}")
        ids = [row["subject_id"] for row in predictions]
        if len(ids) != len(set(ids)) or subject_ids.intersection(ids):
            raise ValueError(f"Duplicate held-out participant in {site}")
        subject_ids.update(ids)

        labels = np.asarray([int(row["label_asd"]) for row in predictions])
        scores = np.asarray([float(row["decision_score"]) for row in predictions])
        predicted = np.asarray([int(row["predicted_asd"]) for row in predictions])
        if not np.array_equal(predicted, (scores >= 0.0).astype(int)):
            raise ValueError(f"Fixed decision threshold was not followed for {site}")
        metric = metrics[0]
        recomputed = {
            "balanced_accuracy": balanced_accuracy_score(labels, predicted),
            "auroc": roc_auc_score(labels, scores),
            "sensitivity": predicted[labels == 1].mean(),
            "specificity": (predicted[labels == 0] == 0).mean(),
        }
        if any(not close(float(metric[name]), float(value)) for name, value in recomputed.items()):
            raise ValueError(f"Recorded metric could not be reproduced for {site}")
        selected = next(row for row in tuning if int(row["selected"]) == 1)
        for name in ("C", "gamma_multiplier", "gamma"):
            if not close(float(metric[name]), float(selected[name])):
                raise ValueError(f"Selected parameter mismatch for {site}: {name}")

        expected_folds = len({row["inner_validation_fold"] for row in inner})
        if expected_folds != 4:
            raise ValueError(f"Inner fold coverage differs for {site}")
        prediction_count += len(predictions)
        metric_count += len(metrics)
        tuning_count += len(tuning)
        inner_fit_count += len(tuning) * expected_folds
        certificate = fold_dir / "complete.json"
        fold_hashes[site] = sha256_file(certificate)

    if metadata["run_kind"] == "full":
        expected = protocol["expected_counts"]
        if prediction_count != int(expected["held_out_prediction_rows"]):
            raise ValueError("Held-out participant coverage differs from the protocol")
        if metric_count != int(expected["held_out_metric_rows"]):
            raise ValueError("Held-out metric coverage differs from the protocol")
        if inner_fit_count != int(expected["inner_fits"]):
            raise ValueError("Inner fit count differs from the protocol")
        if len(subject_ids) != int(protocol["cohort"]["participants"]):
            raise ValueError("Unique participant coverage differs from the protocol")

    certificate = {
        "audit_version": 1,
        "run_id": metadata["run_id"],
        "run_kind": metadata["run_kind"],
        "score_blind": True,
        "passed": True,
        "sealed_sites": len(sites),
        "held_out_prediction_rows": prediction_count,
        "held_out_metric_rows": metric_count,
        "tuning_candidate_rows": tuning_count,
        "inner_fits_reconstructed": inner_fit_count,
        "unique_held_out_participants": len(subject_ids),
        "metric_values_disclosed": False,
        "fold_certificate_sha256": fold_hashes,
    }
    write_json_atomic(run_dir / "score_blind_audit.json", certificate)
    return certificate


def main() -> None:
    args = parse_args()
    result = audit(args.run_dir, args.protocol)
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "passed": result["passed"],
                "sealed_sites": result["sealed_sites"],
                "metric_values_disclosed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
