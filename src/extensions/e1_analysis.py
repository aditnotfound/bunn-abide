"""Deterministic analysis primitives for the frozen E1 intervention audit."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


INTERVENTIONS = (
    "identity_maps",
    "node_map_shuffle",
    "random_orthogonal_maps",
    "degree_preserving_topology",
)
RANDOM_INTERVENTIONS = {
    "node_map_shuffle", "random_orthogonal_maps", "degree_preserving_topology"
}
DIAGNOSTICS = (
    "normalized_effective_rank",
    "normalized_dispersion",
    "mean_pairwise_cosine",
    "invariant_edge_transport_distance",
)
SECONDARY_ENDPOINTS = (
    "mean_absolute_probability_change",
    "classification_flip_fraction",
    "auroc_change_pp",
    "sensitivity_change_pp",
    "specificity_change_pp",
    "normalized_effective_rank_change",
    "normalized_dispersion_change",
    "mean_pairwise_cosine_change",
    "invariant_edge_transport_distance_change",
)


class E1AnalysisError(ValueError):
    """Raised when an E1 result artifact or estimand is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics along the final subject axis for any leading dimensions."""
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape[-1] != len(labels) or set(labels.tolist()) != {0, 1}:
        raise E1AnalysisError("Metric inputs require aligned probabilities and both classes")
    positive = labels == 1
    negative = labels == 0
    predicted = probabilities >= 0.5
    sensitivity = predicted[..., positive].mean(axis=-1)
    specificity = (~predicted[..., negative]).mean(axis=-1)
    balanced_accuracy = 0.5 * (sensitivity + specificity)

    # Mann-Whitney form of AUROC, with average ranks for ties.
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    auroc = np.empty(len(flat), dtype=np.float64)
    positives = int(positive.sum())
    negatives = int(negative.sum())
    for index, row in enumerate(flat):
        ranks = rankdata(row, method="average")
        rank_sum = ranks[positive].sum()
        auroc[index] = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    auroc = auroc.reshape(probabilities.shape[:-1])
    return {
        "balanced_accuracy": balanced_accuracy,
        "auroc": auroc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def site_density_rows(
    *,
    site: str,
    density: float,
    labels: np.ndarray,
    probabilities: np.ndarray,
    diagnostics: np.ndarray,
    intervention_names: list[str],
    diagnostic_names: list[str],
) -> list[dict[str, Any]]:
    """Compute the frozen seed/permutation aggregation for one site-density."""
    if intervention_names[0] != "unaltered":
        raise E1AnalysisError("The first intervention must be unaltered")
    if diagnostic_names != list(DIAGNOSTICS):
        raise E1AnalysisError("Diagnostic order differs from the frozen contract")
    if probabilities.ndim != 4 or diagnostics.ndim != 5:
        raise E1AnalysisError("Unexpected E1 array rank")
    if probabilities.shape[:3] != diagnostics.shape[:3] or probabilities.shape[-1] != diagnostics.shape[-2]:
        raise E1AnalysisError("Prediction and diagnostic arrays are misaligned")

    reference_probability = probabilities[0, :, 0, :].astype(np.float64)
    reference_diagnostic = diagnostics[0, :, 0, :, :].astype(np.float64)
    if not np.isfinite(reference_probability).all() or not np.isfinite(reference_diagnostic).all():
        raise E1AnalysisError("Unaltered reference contains non-finite values")
    reference_metrics = binary_metrics(labels, reference_probability)
    reference_predictions = reference_probability >= 0.5
    rows: list[dict[str, Any]] = []

    for intervention in INTERVENTIONS:
        intervention_index = intervention_names.index(intervention)
        permutation_count = probabilities.shape[2] if intervention in RANDOM_INTERVENTIONS else 1
        intervention_probability = probabilities[intervention_index, :, :permutation_count, :].astype(np.float64)
        intervention_diagnostic = diagnostics[intervention_index, :, :permutation_count, :, :].astype(np.float64)
        if not np.isfinite(intervention_probability).all() or not np.isfinite(intervention_diagnostic).all():
            raise E1AnalysisError(f"{site} {density} {intervention} has non-finite expected cells")
        metrics = binary_metrics(labels, intervention_probability)
        row: dict[str, Any] = {
            "site": site,
            "density": float(density),
            "intervention": intervention,
            "participants": int(len(labels)),
            "reference_balanced_accuracy": float(reference_metrics["balanced_accuracy"].mean()),
            "intervention_balanced_accuracy": float(metrics["balanced_accuracy"].mean()),
            "balanced_accuracy_change_pp": float(
                100.0 * (metrics["balanced_accuracy"].mean() - reference_metrics["balanced_accuracy"].mean())
            ),
            "mean_absolute_probability_change": float(
                np.abs(intervention_probability - reference_probability[:, None, :]).mean()
            ),
            "classification_flip_fraction": float(
                ((intervention_probability >= 0.5) != reference_predictions[:, None, :]).mean()
            ),
            "auroc_change_pp": float(100.0 * (metrics["auroc"].mean() - reference_metrics["auroc"].mean())),
            "sensitivity_change_pp": float(
                100.0 * (metrics["sensitivity"].mean() - reference_metrics["sensitivity"].mean())
            ),
            "specificity_change_pp": float(
                100.0 * (metrics["specificity"].mean() - reference_metrics["specificity"].mean())
            ),
        }
        diagnostic_change = intervention_diagnostic - reference_diagnostic[:, None, :, :]
        for metric_index, metric in enumerate(DIAGNOSTICS):
            row[f"{metric}_change"] = float(diagnostic_change[..., metric_index].mean())
        rows.append(row)
    return rows


def paired_bootstrap(values: np.ndarray, resamples: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise E1AnalysisError("Paired bootstrap requires a finite one-dimensional site vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    estimates = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or len(values) > 20 or not np.isfinite(values).all():
        raise E1AnalysisError("Exact sign-flip test requires 1--20 finite paired site effects")
    observed = abs(float(values.mean()))
    assignments = np.arange(1 << len(values), dtype=np.uint32)[:, None]
    bits = (assignments >> np.arange(len(values), dtype=np.uint32)) & 1
    signs = 1.0 - 2.0 * bits
    permuted = np.abs((signs * values[None, :]).mean(axis=1))
    return float(np.mean(permuted >= observed - 1e-12))


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or np.any((p < 0) | (p > 1)):
        raise E1AnalysisError("Holm adjustment requires valid p-values")
    order = np.argsort(p, kind="stable")
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted.tolist()


def validate_site_density_coverage(table: pd.DataFrame, sites: list[str], densities: list[float]) -> None:
    expected = set(itertools.product(sites, densities, INTERVENTIONS))
    observed = set(
        (str(row.site), float(row.density), str(row.intervention))
        for row in table[["site", "density", "intervention"]].itertuples(index=False)
    )
    if observed != expected or len(table) != len(expected):
        raise E1AnalysisError("Site-density intervention coverage is incomplete or duplicated")


def aggregate_tables(
    site_density: pd.DataFrame,
    *,
    sites: list[str],
    densities: list[float],
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, pd.DataFrame]:
    validate_site_density_coverage(site_density, sites, densities)
    primary_rows: list[dict[str, Any]] = []
    raw_p: list[float] = []
    density_rows: list[dict[str, Any]] = []
    secondary_rows: list[dict[str, Any]] = []

    for intervention_index, intervention in enumerate(INTERVENTIONS):
        subset = site_density[site_density.intervention == intervention]
        per_site = subset.groupby("site", sort=True)["balanced_accuracy_change_pp"].mean().reindex(sites)
        estimate, lower, upper = paired_bootstrap(
            per_site.to_numpy(), bootstrap_resamples, bootstrap_seed + intervention_index
        )
        p_value = exact_sign_flip_p(per_site.to_numpy())
        raw_p.append(p_value)
        primary_rows.append({
            "intervention": intervention, "site_count": len(sites),
            "estimate_pp": estimate, "ci_lower_pp": lower, "ci_upper_pp": upper,
            "exact_sign_flip_p": p_value,
        })

        for density_index, density in enumerate(densities):
            values = subset[subset.density == density].set_index("site")["balanced_accuracy_change_pp"].reindex(sites)
            estimate_d, lower_d, upper_d = paired_bootstrap(
                values.to_numpy(), bootstrap_resamples,
                bootstrap_seed + 100 + intervention_index * len(densities) + density_index,
            )
            density_rows.append({
                "intervention": intervention, "density": density, "site_count": len(sites),
                "estimate_pp": estimate_d, "ci_lower_pp": lower_d, "ci_upper_pp": upper_d,
            })

        for endpoint_index, endpoint in enumerate(SECONDARY_ENDPOINTS):
            values = subset.groupby("site", sort=True)[endpoint].mean().reindex(sites)
            estimate_s, lower_s, upper_s = paired_bootstrap(
                values.to_numpy(), bootstrap_resamples,
                bootstrap_seed + 1_000 + intervention_index * len(SECONDARY_ENDPOINTS) + endpoint_index,
            )
            secondary_rows.append({
                "intervention": intervention, "endpoint": endpoint, "site_count": len(sites),
                "estimate": estimate_s, "ci_lower": lower_s, "ci_upper": upper_s,
            })

    adjusted = holm_adjust(raw_p)
    for row, adjusted_p in zip(primary_rows, adjusted, strict=True):
        row["holm_adjusted_p"] = adjusted_p

    association_rows: list[dict[str, Any]] = []
    for intervention in INTERVENTIONS:
        subset = site_density[site_density.intervention == intervention].sort_values(["site", "density"])
        for diagnostic in DIAGNOSTICS:
            rho, _ = spearmanr(
                subset["balanced_accuracy_change_pp"].to_numpy(),
                subset[f"{diagnostic}_change"].to_numpy(),
            )
            association_rows.append({
                "intervention": intervention, "diagnostic": diagnostic,
                "site_density_cells": len(subset), "spearman_rho": float(rho),
                "interpretation": "descriptive_noncausal",
            })

    return {
        "site_density_effects": site_density.sort_values(["site", "density", "intervention"]).reset_index(drop=True),
        "primary_contrasts": pd.DataFrame(primary_rows),
        "density_contrasts": pd.DataFrame(density_rows),
        "secondary_endpoints": pd.DataFrame(secondary_rows),
        "representation_prediction_associations": pd.DataFrame(association_rows),
    }


def load_and_compute(source_root: Path, protocol: dict[str, Any]) -> dict[str, pd.DataFrame]:
    manager_dir = source_root / "e1_full_v1"
    manager_path = manager_dir / "manager_complete.json"
    if sha256_file(manager_path) != protocol["source"]["manager_completion_sha256"]:
        raise E1AnalysisError("Manager completion certificate hash mismatch")
    manager = json.loads(manager_path.read_text(encoding="utf-8"))
    if manager.get("state") != "complete_all_sites_score_blind_audited":
        raise E1AnalysisError("Full E1 manager is not complete and audited")
    sites = [str(value) for value in manager["sites"]]
    densities = [float(value) for value in protocol["coverage"]["densities"]]
    rows: list[dict[str, Any]] = []

    for site in sites:
        site_dir = source_root / f"e1_full_v1__{site}"
        audit_path = site_dir / "score_blind_audit.json"
        if sha256_file(audit_path) != manager["audit_hashes"][site]:
            raise E1AnalysisError(f"{site} audit hash differs from manager certificate")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("state") != "passed" or not audit.get("score_blind"):
            raise E1AnalysisError(f"{site} did not pass its score-blind audit")
        for density in densities:
            name = f"density_{density:.2f}.npz"
            path = site_dir / name
            if audit["artifact_hashes"].get(name) != sha256_file(path):
                raise E1AnalysisError(f"{site} {name} hash mismatch")
            with np.load(path, allow_pickle=False) as arrays:
                rows.extend(site_density_rows(
                    site=site, density=density, labels=arrays["labels"],
                    probabilities=arrays["probabilities"], diagnostics=arrays["diagnostics"],
                    intervention_names=arrays["interventions"].astype(str).tolist(),
                    diagnostic_names=arrays["metrics"].astype(str).tolist(),
                ))
    table = pd.DataFrame(rows)
    return aggregate_tables(
        table, sites=sites, densities=densities,
        bootstrap_resamples=int(protocol["uncertainty"]["paired_site_bootstrap_resamples"]),
        bootstrap_seed=int(protocol["uncertainty"]["bootstrap_seed"]),
    )
