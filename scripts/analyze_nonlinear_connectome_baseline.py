"""Analyze the audited post-hoc RBF-SVM comparator once."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-per-site",
        type=Path,
        default=Path("outputs/analysis/step7_6_full_baselines_v2/per_site_metrics.csv"),
    )
    parser.add_argument(
        "--neural-curves",
        type=Path,
        default=Path("outputs/analysis/step11_neural_full_parallel_v1/site_predictive_curves.csv"),
    )
    parser.add_argument(
        "--run-archive", type=Path, default=Path("outputs/archives/rbf_svm_full_v1.tar.gz")
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/extensions/nonlinear_baseline_analysis_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/nonlinear_baseline_analysis_v1"),
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


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def exact_sign_flip_p(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    signs = np.asarray(list(product((-1.0, 1.0), repeat=len(values))), dtype=float)
    permuted = np.abs((signs * values).mean(axis=1))
    return float(np.mean(permuted >= observed - 1e-15))


def contrast_summary(
    name: str,
    values: np.ndarray,
    counts: np.ndarray,
    bootstrap_indices: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    bootstrapped = values[bootstrap_indices].mean(axis=1)
    return {
        "contrast": name,
        "held_out_sites": len(values),
        "equal_site_mean_difference": float(values.mean()),
        "bootstrap_ci_low": float(np.quantile(bootstrapped, 0.025)),
        "bootstrap_ci_high": float(np.quantile(bootstrapped, 0.975)),
        "exact_sign_flip_p": exact_sign_flip_p(values),
        "participant_weighted_mean_difference": float(np.average(values, weights=counts)),
        "median_site_difference": float(np.median(values)),
        "positive_site_differences": int((values > 0).sum()),
        "negative_site_differences": int((values < 0).sum()),
        "bootstrap_resamples": len(bootstrap_indices),
        "bootstrap_seed": seed,
    }


def analyze(
    run_dir: Path,
    baseline_path: Path,
    neural_path: Path,
    archive_path: Path,
    contract_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite analysis: {output_dir}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected_hashes = contract["source_sha256"]
    actual_hashes = {
        "run_archive": sha256_file(archive_path),
        "classical_per_site_metrics": sha256_file(baseline_path),
        "neural_site_predictive_curves": sha256_file(neural_path),
    }
    if actual_hashes != expected_hashes:
        raise ValueError("One or more frozen analysis inputs changed")
    audit = json.loads((run_dir / "score_blind_audit.json").read_text(encoding="utf-8"))
    if not audit.get("passed") or audit.get("metric_values_disclosed"):
        raise ValueError("Run did not pass the score-blind audit")
    if audit["run_id"] != contract["input_run_id"]:
        raise ValueError("Run ID differs from the analysis contract")

    metrics: list[dict[str, str]] = []
    predictions: list[dict[str, str]] = []
    for fold in sorted((run_dir / "folds").iterdir()):
        if fold.is_dir() and not fold.name.startswith("."):
            metrics.extend(read_csv(fold / "test_metrics.csv"))
            predictions.extend(read_csv(fold / "predictions.csv"))
    sites = sorted({row["held_out_site"] for row in metrics})
    if len(sites) != int(contract["expected_sites"]):
        raise ValueError("RBF-SVM site coverage differs from the contract")
    if len(predictions) != int(contract["expected_participants"]):
        raise ValueError("RBF-SVM participant coverage differs from the contract")
    rbf = {row["held_out_site"]: float(row["balanced_accuracy"]) for row in metrics}
    counts = {row["held_out_site"]: int(row["participants"]) for row in metrics}

    baseline_rows = read_csv(baseline_path)
    elastic = {
        row["held_out_site"]: float(row["balanced_accuracy"])
        for row in baseline_rows
        if row["model"] == "connectome_elastic_net_logistic"
    }
    neural_rows = read_csv(neural_path)
    curves = {
        (row["held_out_site"], row["curve_operator"]):
            float(row["normalized_auc_balanced_accuracy"])
        for row in neural_rows
    }
    if set(rbf) != set(elastic) or any((site, operator) not in curves for site in sites for operator in ("gcn", "learned_bunn")):
        raise ValueError("Paired site coverage differs across comparator sources")

    references = {
        "rbf_svm_minus_connectome_elastic_net": elastic,
        "rbf_svm_minus_gcn_curve": {site: curves[(site, "gcn")] for site in sites},
        "rbf_svm_minus_learned_bunn_curve": {
            site: curves[(site, "learned_bunn")] for site in sites
        },
    }
    seed = int(contract["inference"]["bootstrap_seed"])
    resamples = int(contract["inference"]["bootstrap_resamples"])
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(0, len(sites), size=(resamples, len(sites)))
    count_array = np.asarray([counts[site] for site in sites], dtype=float)
    contrast_rows = []
    site_rows = []
    for contrast in contract["contrasts"]:
        name = contrast["name"]
        reference = references[name]
        values = np.asarray([rbf[site] - reference[site] for site in sites], dtype=float)
        contrast_rows.append(
            contrast_summary(name, values, count_array, bootstrap_indices, seed)
        )
        for site, value in zip(sites, values, strict=True):
            site_rows.append(
                {
                    "contrast": name,
                    "held_out_site": site,
                    "participants": counts[site],
                    "rbf_svm_balanced_accuracy": rbf[site],
                    "reference_balanced_accuracy": reference[site],
                    "paired_difference": float(value),
                }
            )

    labels = np.asarray([int(row["label_asd"]) for row in predictions])
    scores = np.asarray([float(row["decision_score"]) for row in predictions])
    predicted = np.asarray([int(row["predicted_asd"]) for row in predictions])
    pooled = {
        "participants": len(labels),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "auroc": float(roc_auc_score(labels, scores)),
        "sensitivity": float(predicted[labels == 1].mean()),
        "specificity": float((predicted[labels == 0] == 0).mean()),
        "equal_site_balanced_accuracy": float(np.mean(list(rbf.values()))),
    }
    output_dir.mkdir(parents=True)
    contrast_fields = list(contrast_rows[0])
    site_fields = list(site_rows[0])
    write_csv(output_dir / "paired_contrasts.csv", contrast_rows, contrast_fields)
    write_csv(output_dir / "site_differences.csv", site_rows, site_fields)
    (output_dir / "rbf_svm_summary.json").write_text(
        json.dumps(pooled, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    generated = sorted(output_dir.iterdir())
    manifest = {
        "analysis_version": contract["analysis_version"],
        "input_run_id": audit["run_id"],
        "post_hoc": True,
        "confirmatory_override_allowed": False,
        "source_sha256": actual_hashes,
        "generated_sha256": {path.name: sha256_file(path) for path in generated},
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = analyze(
        args.run_dir, args.baseline_per_site, args.neural_curves, args.run_archive,
        args.contract, args.output_dir,
    )
    print(json.dumps({"analysis_version": manifest["analysis_version"], "status": "complete"}, indent=2))


if __name__ == "__main__":
    main()
