"""Run the frozen, revision-only analyses for the standalone YHSA report."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
E1_INTERVENTIONS = (
    "identity_maps",
    "node_map_shuffle",
    "random_orthogonal_maps",
    "degree_preserving_topology",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_atomic(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False, float_format="%.12g", lineterminator="\n")
    os.replace(temporary, path)


def write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def bootstrap_mean(
    values: np.ndarray, *, resamples: int, rng: np.random.Generator,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    draws = values[indices].mean(axis=1)
    low, high = percentile_interval(draws)
    return float(values.mean()), low, high


def bootstrap_weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    sampled_values = values[indices]
    sampled_weights = weights[indices]
    draws = (sampled_values * sampled_weights).sum(axis=1) / sampled_weights.sum(axis=1)
    low, high = percentile_interval(draws)
    estimate = float(np.average(values, weights=weights))
    return estimate, low, high


def exact_two_sided_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    permuted = np.abs((signs * values).mean(axis=1))
    return float(np.mean(permuted >= observed - 1e-15))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted_sorted = np.maximum.accumulate(
        (len(p_values) - np.arange(len(p_values))) * p_values[order]
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def verify_inputs(protocol: dict) -> None:
    for record in protocol["inputs"].values():
        path = PROJECT_ROOT / record["path"]
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise ValueError(f"Frozen input hash mismatch for {path}: {actual}")


def qc_tables(protocol: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(PROJECT_ROOT / protocol["inputs"]["raw_phenotype"]["path"], low_memory=False)
    parent = pd.read_csv(PROJECT_ROOT / protocol["inputs"]["parent_manifest"]["path"])
    final = pd.read_csv(PROJECT_ROOT / protocol["inputs"]["analysis_manifest"]["path"])
    dx_valid = raw["DX_GROUP"].isin([1, 2])
    file_valid = raw["FILE_ID"].notna() & ~raw["FILE_ID"].astype(str).str.lower().isin(
        ["", "no_filename", "nan"]
    )
    manual_ok = raw["qc_rater_1"].astype(str).str.upper().eq("OK")
    functional_ok = raw["qc_func_rater_2"].astype(str).str.upper().eq("OK")
    eligible_sites = set(parent["site_id"].astype(str))
    site_ok = raw["SITE_ID"].astype(str).isin(eligible_sites)
    final_ids = set(final["subject_id"].astype(str))
    parent_ids = set(parent["subject_id"].astype(str))
    subject_ids = raw["SUB_ID"].astype("Int64").astype(str)

    masks = [
        ("All phenotype rows", np.ones(len(raw), dtype=bool)),
        ("Valid diagnosis", dx_valid),
        ("Available FILE_ID", dx_valid & file_valid),
        ("Manual anatomical QC retained", dx_valid & file_valid & manual_ok),
        ("Manual functional QC retained", dx_valid & file_valid & manual_ok & functional_ok),
        ("Eligible 18-site set", dx_valid & file_valid & manual_ok & functional_ok & site_ok),
        ("Final nonzero-variance cohort", subject_ids.isin(final_ids)),
    ]
    cascade_rows = []
    for order, (stage, mask) in enumerate(masks, start=1):
        subset = raw.loc[mask]
        cascade_rows.append(
            {
                "order": order,
                "stage": stage,
                "participants": len(subset),
                "asd": int((subset["DX_GROUP"] == 1).sum()),
                "control": int((subset["DX_GROUP"] == 2).sum()),
                "removed_from_previous": 0 if order == 1 else cascade_rows[-1]["participants"] - len(subset),
            }
        )
    cascade = pd.DataFrame(cascade_rows)

    functional_pass = dx_valid & file_valid & manual_ok & functional_ok
    site_rows = []
    for site in sorted(set(raw.loc[functional_pass, "SITE_ID"].astype(str)) | eligible_sites):
        stage = raw.loc[functional_pass & raw["SITE_ID"].astype(str).eq(site)]
        parent_site = parent[parent["site_id"].astype(str).eq(site)]
        final_site = final[final["site_id"].astype(str).eq(site)]
        site_rows.append(
            {
                "site": site,
                "after_manual_functional_qc": len(stage),
                "eligible_parent_manifest": len(parent_site),
                "final_analysis": len(final_site),
                "final_asd": int((final_site["dx_group"] == 1).sum()),
                "final_control": int((final_site["dx_group"] == 2).sum()),
                "excluded_by_site_rule": len(stage) if site not in eligible_sites else 0,
                "excluded_zero_variance": len(parent_site) - len(final_site),
            }
        )
    site_attrition = pd.DataFrame(site_rows)

    outcome = np.full(len(raw), "retained_final", dtype=object)
    outcome[~subject_ids.isin(final_ids)] = "zero_variance_or_later_exclusion"
    outcome[~subject_ids.isin(parent_ids)] = "ineligible_site_or_earlier"
    outcome[~functional_ok] = "failed_functional_qc"
    outcome[~manual_ok] = "failed_manual_qc"
    outcome[~file_valid] = "unavailable_file"
    outcome[~dx_valid] = "invalid_diagnosis"
    fd = pd.to_numeric(raw["func_mean_fd"], errors="coerce")
    fd_rows = []
    for label in (
        "retained_final",
        "zero_variance_or_later_exclusion",
        "ineligible_site_or_earlier",
        "failed_functional_qc",
        "failed_manual_qc",
        "unavailable_file",
        "invalid_diagnosis",
    ):
        values = fd[outcome == label].dropna()
        fd_rows.append(
            {
                "outcome": label,
                "participants": int(np.sum(outcome == label)),
                "fd_available": len(values),
                "mean_fd": values.mean() if len(values) else np.nan,
                "median_fd": values.median() if len(values) else np.nan,
                "q25_fd": values.quantile(0.25) if len(values) else np.nan,
                "q75_fd": values.quantile(0.75) if len(values) else np.nan,
            }
        )
    return cascade, site_attrition, pd.DataFrame(fd_rows)


def build_site_model_table(protocol: dict) -> pd.DataFrame:
    neural = pd.read_csv(PROJECT_ROOT / protocol["inputs"]["site_configuration_metrics"]["path"])
    curves = pd.read_csv(PROJECT_ROOT / protocol["inputs"]["site_predictive_curves"]["path"])
    baseline = pd.read_csv(PROJECT_ROOT / protocol["inputs"]["baseline_site_metrics"]["path"])
    manifest = pd.read_csv(PROJECT_ROOT / protocol["inputs"]["analysis_manifest"]["path"])
    counts = manifest.groupby("site_id").size().rename("participants")
    rows = []
    for row in baseline.itertuples(index=False):
        rows.append(
            {"model": row.model, "site": row.held_out_site, "participants": row.participants,
             "balanced_accuracy": row.balanced_accuracy, "summary": "single_configuration"}
        )
    for row in curves.itertuples(index=False):
        rows.append(
            {"model": f"{row.curve_operator}_curve", "site": row.held_out_site,
             "participants": int(counts[row.held_out_site]),
             "balanced_accuracy": row.normalized_auc_balanced_accuracy, "summary": "normalized_trapezoid_curve"}
        )
    for operator in ("identity", "learned_local"):
        for row in neural[neural.operator.eq(operator)].itertuples(index=False):
            rows.append(
                {"model": operator, "site": row.held_out_site, "participants": row.participants,
                 "balanced_accuracy": row.mean_balanced_accuracy, "summary": "single_configuration"}
            )
    rbf = pd.read_csv(PROJECT_ROOT / "outputs/analysis/nonlinear_baseline_analysis_v1/site_differences.csv")
    rbf = rbf[rbf.contrast.eq("rbf_svm_minus_connectome_elastic_net")]
    for row in rbf.itertuples(index=False):
        rows.append(
            {"model": "rbf_svm", "site": row.held_out_site, "participants": row.participants,
             "balanced_accuracy": row.rbf_svm_balanced_accuracy, "summary": "single_configuration"}
        )
    return pd.DataFrame(rows).sort_values(["model", "site"]).reset_index(drop=True)


def predictive_tables(protocol: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(protocol["inference"]["bootstrap_seed"])
    resamples = protocol["inference"]["site_bootstrap_resamples"]
    site_models = build_site_model_table(protocol)
    summary_rows = []
    for model, group in site_models.groupby("model", sort=True):
        estimate, low, high = bootstrap_mean(group.balanced_accuracy.to_numpy(), resamples=resamples, rng=rng)
        weighted, weighted_low, weighted_high = bootstrap_weighted_mean(
            group.balanced_accuracy.to_numpy(), group.participants.to_numpy(), resamples=resamples, rng=rng
        )
        summary_rows.append(
            {"model": model, "sites": len(group), "equal_site_estimate": estimate,
             "equal_site_ci_low": low, "equal_site_ci_high": high,
             "participant_weighted_estimate": weighted,
             "participant_weighted_ci_low": weighted_low, "participant_weighted_ci_high": weighted_high}
        )
    model_summary = pd.DataFrame(summary_rows)

    contrasts = (
        ("learned_bunn_curve", "gcn_curve"),
        ("learned_bunn_curve", "connectome_elastic_net_logistic"),
        ("rbf_svm", "connectome_elastic_net_logistic"),
    )
    contrast_rows = []
    for left, right in contrasts:
        pair = site_models[site_models.model.eq(left)].merge(
            site_models[site_models.model.eq(right)], on="site", suffixes=("_left", "_right"), validate="one_to_one"
        )
        values = pair.balanced_accuracy_left.to_numpy() - pair.balanced_accuracy_right.to_numpy()
        weights = pair.participants_left.to_numpy()
        estimate, low, high = bootstrap_mean(values, resamples=resamples, rng=rng)
        weighted, weighted_low, weighted_high = bootstrap_weighted_mean(
            values, weights, resamples=resamples, rng=rng
        )
        contrast_rows.append(
            {"contrast": f"{left}_minus_{right}", "sites": len(pair),
             "equal_site_estimate": estimate, "equal_site_ci_low": low, "equal_site_ci_high": high,
             "participant_weighted_estimate": weighted,
             "participant_weighted_ci_low": weighted_low, "participant_weighted_ci_high": weighted_high}
        )
    contrast_summary = pd.DataFrame(contrast_rows)

    metrics = pd.read_csv(PROJECT_ROOT / protocol["inputs"]["site_configuration_metrics"]["path"])
    anchors = {"gcn": "identity", "trivial_bundle": "identity", "learned_bunn": "learned_local"}
    uniform_rows = []
    uniform_site = {}
    for operator, anchor in anchors.items():
        anchor_table = metrics[metrics.operator.eq(anchor)].set_index("held_out_site")
        operator_table = metrics[metrics.operator.eq(operator)]
        site_values = {}
        for site, group in operator_table.groupby("held_out_site"):
            values = [float(anchor_table.loc[site, "mean_balanced_accuracy"])] + group.sort_values("density").mean_balanced_accuracy.tolist()
            site_values[site] = float(np.mean(values))
        uniform_site[operator] = site_values
        vals = np.asarray(list(site_values.values()))
        estimate, low, high = bootstrap_mean(vals, resamples=resamples, rng=rng)
        uniform_rows.append({"operator": operator, "sites": len(vals), "uniform_grid_estimate": estimate,
                             "ci_low": low, "ci_high": high})
    for left, right in (("learned_bunn", "gcn"), ("learned_bunn", "trivial_bundle")):
        sites = sorted(uniform_site[left])
        vals = np.asarray([uniform_site[left][site] - uniform_site[right][site] for site in sites])
        estimate, low, high = bootstrap_mean(vals, resamples=resamples, rng=rng)
        uniform_rows.append({"operator": f"{left}_minus_{right}", "sites": len(vals),
                             "uniform_grid_estimate": estimate, "ci_low": low, "ci_high": high})
    return site_models, model_summary, contrast_summary, pd.DataFrame(uniform_rows)


def e1_pairwise_table(protocol: dict) -> pd.DataFrame:
    table = pd.read_csv(PROJECT_ROOT / protocol["inputs"]["e1_site_density_effects"]["path"])
    site_means = table.groupby(["site", "intervention"], as_index=False).balanced_accuracy_change_pp.mean()
    rng = np.random.default_rng(protocol["inference"]["bootstrap_seed"] + 1)
    rows = []
    for left, right in itertools.combinations(E1_INTERVENTIONS, 2):
        pair = site_means[site_means.intervention.eq(left)].merge(
            site_means[site_means.intervention.eq(right)], on="site", suffixes=("_left", "_right"), validate="one_to_one"
        )
        values = pair.balanced_accuracy_change_pp_left.to_numpy() - pair.balanced_accuracy_change_pp_right.to_numpy()
        estimate, low, high = bootstrap_mean(
            values, resamples=protocol["inference"]["site_bootstrap_resamples"], rng=rng
        )
        rows.append({"contrast": f"{left}_minus_{right}", "sites": len(values), "estimate_pp": estimate,
                     "ci_low_pp": low, "ci_high_pp": high, "exact_sign_flip_p": exact_two_sided_sign_flip_p(values)})
    result = pd.DataFrame(rows)
    result["holm_adjusted_p"] = holm_adjust(result.exact_sign_flip_p.to_numpy())
    return result


def positive_topk_adjacency(connectome: np.ndarray, density: float) -> np.ndarray:
    nodes = connectome.shape[0]
    edge_i, edge_j = np.triu_indices(nodes, 1)
    edge_count = int(np.ceil(density * len(edge_i)))
    scores = connectome[edge_i, edge_j]
    if int(np.sum(scores > 0)) < edge_count:
        raise ValueError("Insufficient positive edges for frozen graph rule")
    order = np.argsort(-scores, kind="stable")[:edge_count]
    adjacency = np.zeros_like(connectome, dtype=float)
    adjacency[edge_i[order], edge_j[order]] = 1.0
    adjacency[edge_j[order], edge_i[order]] = 1.0
    return adjacency


def heat_spectrum_table(protocol: dict) -> pd.DataFrame:
    archive = np.load(PROJECT_ROOT / protocol["inputs"]["connectomes"]["path"])
    connectomes = archive["connectomes_fisher_z"]
    rows = []
    for density in protocol["heat_spectrum"]["densities"]:
        participant_rows = []
        for connectome in connectomes:
            adjacency = positive_topk_adjacency(connectome, density)
            degree = adjacency.sum(axis=1)
            active = degree > 0
            inverse_sqrt = np.zeros_like(degree, dtype=float)
            inverse_sqrt[active] = 1.0 / np.sqrt(degree[active])
            symmetric_laplacian = np.diag(active.astype(float)) - inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]
            eigenvalues = np.linalg.eigvalsh(symmetric_laplacian)
            eigenvalues = np.clip(eigenvalues, 0.0, 2.0)
            response = np.exp(-protocol["heat_spectrum"]["diffusion_time"] * eigenvalues)
            participant_rows.append(
                [eigenvalues[1], np.median(eigenvalues), eigenvalues[-1], response.min(),
                 np.median(response), response.mean(), np.mean(eigenvalues < 1e-8)]
            )
        values = np.asarray(participant_rows)
        names = protocol["heat_spectrum"]["summaries"]
        for column, name in enumerate(names):
            rows.append(
                {"density": density, "metric": name, "participants": len(values),
                 "mean": values[:, column].mean(), "sd": values[:, column].std(ddof=1),
                 "median": np.median(values[:, column]), "q25": np.quantile(values[:, column], 0.25),
                 "q75": np.quantile(values[:, column], 0.75)}
            )
    return pd.DataFrame(rows)


def latex_escape(value: str) -> str:
    return value.replace("_", "\\_").replace("%", "\\%")


def generate_latex_tables(
    cascade: pd.DataFrame,
    model_summary: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    e1: pd.DataFrame,
    heat: pd.DataFrame,
    output_dir: Path,
) -> None:
    qc_lines = ["% Generated by analyze_final_revision.py; do not edit by hand.", "\\begin{tabular}{lrrrr}",
                "\\toprule", "Stage & $n$ & ASD & Control & Removed \\\\", "\\midrule"]
    for row in cascade.itertuples(index=False):
        qc_lines.append(f"{latex_escape(row.stage)} & {row.participants} & {row.asd} & {row.control} & {row.removed_from_previous} \\\\")
    qc_lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    write_text_atomic("\n".join(qc_lines), output_dir / "qc_cascade.tex")

    selected_models = ["connectome_elastic_net_logistic", "rbf_svm", "identity", "gcn_curve", "learned_local", "learned_bunn_curve"]
    labels = {"connectome_elastic_net_logistic": "Connectome elastic net", "rbf_svm": "RBF-SVM",
              "identity": "Identity", "gcn_curve": "GCN curve", "learned_local": "Learned-local",
              "learned_bunn_curve": "Learned BuNN curve"}
    lines = ["% Generated by analyze_final_revision.py; do not edit by hand.", "\\begin{tabularx}{\\textwidth}{@{}Ycc@{}}", "\\toprule",
             "Model & Equal-site mean [95\\% CI] & Participant-weighted mean [95\\% CI] \\\\", "\\midrule"]
    indexed = model_summary.set_index("model")
    for model in selected_models:
        row = indexed.loc[model]
        lines.append(f"{labels[model]} & {row.equal_site_estimate:.3f} [{row.equal_site_ci_low:.3f}, {row.equal_site_ci_high:.3f}] & "
                     f"{row.participant_weighted_estimate:.3f} [{row.participant_weighted_ci_low:.3f}, {row.participant_weighted_ci_high:.3f}] \\\\")
    lines.extend(["\\bottomrule", "\\end{tabularx}", ""])
    write_text_atomic("\n".join(lines), output_dir / "model_weighting_intervals.tex")

    contrast_labels = {
        "learned_bunn_curve_minus_gcn_curve": "BuNN curve $-$ GCN curve",
        "learned_bunn_curve_minus_connectome_elastic_net_logistic": "BuNN curve $-$ elastic net",
        "rbf_svm_minus_connectome_elastic_net_logistic": "RBF-SVM $-$ elastic net",
    }
    lines = ["% Generated by analyze_final_revision.py; do not edit by hand.", "\\begin{tabularx}{\\textwidth}{@{}Ycc@{}}", "\\toprule",
             "Contrast & Equal-site estimate [95\\% CI] & Participant-weighted estimate [95\\% CI] \\\\", "\\midrule"]
    for row in contrast_summary.itertuples(index=False):
        lines.append(f"{contrast_labels[row.contrast]} & {row.equal_site_estimate:+.3f} [{row.equal_site_ci_low:+.3f}, {row.equal_site_ci_high:+.3f}] & "
                     f"{row.participant_weighted_estimate:+.3f} [{row.participant_weighted_ci_low:+.3f}, {row.participant_weighted_ci_high:+.3f}] \\\\")
    lines.extend(["\\bottomrule", "\\end{tabularx}", ""])
    write_text_atomic("\n".join(lines), output_dir / "weighting_contrasts.tex")

    e1_labels = {
        "identity_maps_minus_node_map_shuffle": "Identity maps $-$ shuffled learned maps",
        "identity_maps_minus_random_orthogonal_maps": "Identity maps $-$ random orthogonal maps",
        "identity_maps_minus_degree_preserving_topology": "Identity maps $-$ topology rewiring",
        "node_map_shuffle_minus_random_orthogonal_maps": "Shuffled learned maps $-$ random orthogonal maps",
        "node_map_shuffle_minus_degree_preserving_topology": "Shuffled learned maps $-$ topology rewiring",
        "random_orthogonal_maps_minus_degree_preserving_topology": "Random orthogonal maps $-$ topology rewiring",
    }
    lines = ["% Generated by analyze_final_revision.py; do not edit by hand.", "\\begin{tabularx}{\\textwidth}{@{}Yrrr@{}}", "\\toprule",
             "Exploratory E1 contrast & Difference (pp) & 95\\% CI & Holm $p$ \\\\", "\\midrule"]
    for row in e1.itertuples(index=False):
        lines.append(f"{e1_labels[row.contrast]} & {row.estimate_pp:+.2f} & [{row.ci_low_pp:+.2f}, {row.ci_high_pp:+.2f}] & {row.holm_adjusted_p:.3f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabularx}", ""])
    write_text_atomic("\n".join(lines), output_dir / "e1_pairwise.tex")

    heat_wide = heat.pivot(index="density", columns="metric", values="mean")
    lines = ["% Generated by analyze_final_revision.py; do not edit by hand.", "\\begin{tabular}{rrrrr}", "\\toprule",
             "Density & $\\lambda_2$ & Median $\\lambda$ & Mean $e^{-\\lambda}$ & Stationary-mode fraction \\\\", "\\midrule"]
    for density, row in heat_wide.iterrows():
        lines.append(f"{100*density:.0f}\\% & {row.lambda_2:.3f} & {row.lambda_median:.3f} & {row.mean_heat_response:.3f} & {row.stationary_mode_fraction:.3f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    write_text_atomic("\n".join(lines), output_dir / "heat_spectrum.tex")


def run(protocol_path: Path, output_dir: Path, latex_dir: Path) -> Path:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    verify_inputs(protocol)
    cascade, site_attrition, fd_summary = qc_tables(protocol)
    site_models, model_summary, contrast_summary, uniform = predictive_tables(protocol)
    e1 = e1_pairwise_table(protocol)
    heat = heat_spectrum_table(protocol)
    outputs = {
        "qc_cascade.csv": cascade,
        "qc_site_attrition.csv": site_attrition,
        "qc_fd_summary.csv": fd_summary,
        "site_model_values.csv": site_models,
        "model_level_intervals.csv": model_summary,
        "participant_weighted_intervals.csv": contrast_summary,
        "uniform_grid_sensitivity.csv": uniform,
        "e1_pairwise_exploratory.csv": e1,
        "heat_spectrum_summary.csv": heat,
    }
    for name, table in outputs.items():
        write_csv_atomic(table, output_dir / name)
    generate_latex_tables(cascade, model_summary, contrast_summary, e1, heat, latex_dir)
    result = {
        "analysis_version": protocol["analysis_version"],
        "protocol_sha256": sha256_file(protocol_path),
        "outputs": {name: {"rows": len(table), "sha256": sha256_file(output_dir / name)} for name, table in outputs.items()},
        "claim_boundary": "Revision-only post-hoc and sensitivity analyses; frozen fitted-model results unchanged.",
    }
    result_path = output_dir / "result.json"
    write_text_atomic(json.dumps(result, indent=2, sort_keys=True) + "\n", result_path)
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs/final_revision_analysis_v1.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/analysis/final_revision_v1")
    parser.add_argument("--latex-dir", type=Path, default=PROJECT_ROOT / "paper/yhsa-submission/generated/tables")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(run(arguments.protocol, arguments.output_dir, arguments.latex_dir))
