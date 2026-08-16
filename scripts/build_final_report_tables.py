"""Generate reproducibility tables for the final report from frozen records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("configs/baseline_protocol.json"))
    parser.add_argument("--neural", type=Path, default=Path("configs/neural_full_protocol.json"))
    parser.add_argument(
        "--operator", type=Path, default=Path("configs/neural_operator_contract_v2.json")
    )
    parser.add_argument(
        "--snapshot", type=Path, default=Path("reproducibility/result_snapshot.json")
    )
    parser.add_argument(
        "--nonlinear",
        type=Path,
        default=Path("reproducibility/nonlinear_baseline_result.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/yhsa-final/generated/tables")
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def number_list(values: list[float]) -> str:
    return ", ".join(f"{float(value):g}" for value in values)


def build_hyperparameters(
    baseline: dict[str, Any], neural: dict[str, Any], operator: dict[str, Any]
) -> str:
    models = baseline["models"]
    training = neural["training"]
    tuning = neural["tuning"]
    backbone = operator["shared_backbone"]
    candidate_pairs = "; ".join(
        f"({float(row['learning_rate']):g}, {float(row['weight_decay']):g})"
        for row in tuning["candidates"]
    )
    rows = [
        (
            "Covariates logistic",
            f"C: {number_list(models['covariates_l2_logistic']['C_grid'])}",
            "L2; balanced classes",
        ),
        (
            "Connectome elastic net",
            f"C: {number_list(models['connectome_elastic_net_logistic']['C_grid'])}; "
            f"l1 ratio: {number_list(models['connectome_elastic_net_logistic']['l1_ratio_grid'])}",
            "SAGA; balanced classes",
        ),
        (
            "Combined elastic net",
            f"C: {number_list(models['combined_elastic_net_logistic']['C_grid'])}; "
            f"l1 ratio: {number_list(models['combined_elastic_net_logistic']['l1_ratio_grid'])}",
            "SAGA; balanced classes",
        ),
        (
            "All neural operators",
            f"(learning rate, weight decay): {candidate_pairs}",
            f"AdamW; batch {training['batch_size']}; at most {training['maximum_epochs']} epochs; "
            f"patience {training['early_stopping_patience']}",
        ),
        (
            "Shared neural backbone",
            f"{backbone['layers']} layers; width {backbone['hidden_dimension']}; "
            f"dropout {float(backbone['dropout']):g}; {backbone['bundles']} bundles",
            f"{backbone['activation']}; node-mean pooling; {len(tuning['seeds'])} tuning and "
            f"{len(neural['final_fit']['seeds'])} final seeds",
        ),
    ]
    body = "\n".join(f"{name} & {search} & {fixed} \\\\" for name, search, fixed in rows)
    return """% Generated from frozen JSON contracts. Do not edit by hand.
\\begin{table}[htbp]
\\centering
\\caption{Complete Study 1 model-selection and training settings.}
\\label{tab:hyperparameters}
\\small
\\begin{tabularx}{\\textwidth}{@{}p{3.0cm}Y Y@{}}
\\toprule
\\textbf{Model or component} & \\textbf{Tuned values} & \\textbf{Fixed settings} \\\\
\\midrule
""" + body + """
\\bottomrule
\\end{tabularx}
\\end{table}
"""


def build_weighting(snapshot: dict[str, Any]) -> str:
    labels = {
        "learned_bunn_curve_minus_gcn_curve": "BuNN curve $-$ GCN curve",
        "learned_bunn_curve_minus_connectome_elastic_net": "BuNN curve $-$ elastic net",
        "learned_bunn_minus_gcn_matched_anchor_effective_rank_change":
            "BuNN $-$ GCN effective-rank change",
    }
    by_contrast: dict[str, dict[str, float]] = {}
    for row in snapshot["robustness"]["alternative_summaries"]:
        by_contrast.setdefault(row["contrast"], {})[row["summary"]] = float(row["estimate"])
    rows = []
    for contrast, label in labels.items():
        values = by_contrast[contrast]
        rows.append(
            f"{label} & {values['equal_site_mean']:+.4f} & "
            f"{values['participant_weighted_mean']:+.4f} & "
            f"{values['median_site_difference']:+.4f} \\\\"
        )
    return """% Generated from the frozen result snapshot. Do not edit by hand.
\\begin{table}[htbp]
\\centering
\\caption{Sensitivity of the three core contrasts to site weighting.}
\\label{tab:weighting-sensitivity}
\\small
\\begin{tabularx}{\\textwidth}{@{}Y r r r@{}}
\\toprule
\\textbf{Contrast} & \\textbf{Equal-site mean} & \\textbf{Participant-weighted} & \\textbf{Site median} \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabularx}
\\end{table}
"""


def build_nonlinear(result: dict[str, Any]) -> str:
    labels = {
        "rbf_svm_minus_connectome_elastic_net": "RBF-SVM $-$ elastic net",
        "rbf_svm_minus_gcn_curve": "RBF-SVM $-$ GCN curve",
        "rbf_svm_minus_learned_bunn_curve": "RBF-SVM $-$ learned BuNN curve",
    }
    rows = []
    for contrast in result["contrasts"]:
        low, high = contrast["bootstrap_ci_95"]
        rows.append(
            f"{labels[contrast['name']]} & "
            f"{float(contrast['equal_site_mean_difference']):+.4f} & "
            f"$[{float(low):+.4f}, {float(high):+.4f}]$ & "
            f"{float(contrast['exact_sign_flip_p']):.3f} \\\\"
        )
    return """% Generated from the audited nonlinear-baseline result. Do not edit by hand.
\\begin{table}[htbp]
\\centering
\\caption{Post-hoc RBF-SVM comparisons using paired held-out sites.}
\\label{tab:nonlinear-baseline}
\\small
\\begin{tabularx}{\\textwidth}{@{}Y r r r@{}}
\\toprule
\\textbf{Contrast} & \\textbf{Difference} & \\textbf{95\\% CI} & \\textbf{Exact $p$} \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabularx}
\\end{table}
"""


def build_tables(
    baseline_path: Path,
    neural_path: Path,
    operator_path: Path,
    snapshot_path: Path,
    output_dir: Path,
    nonlinear_path: Path | None = None,
) -> list[Path]:
    baseline = read_json(baseline_path)
    neural = read_json(neural_path)
    operator = read_json(operator_path)
    snapshot = read_json(snapshot_path)
    if baseline["analysis_manifest_sha256"] != neural["analysis_manifest_sha256"]:
        raise ValueError("Baseline and neural protocols do not share one analysis manifest")
    if neural["analysis_manifest_sha256"] != operator["analysis_manifest_sha256"]:
        raise ValueError("Neural protocol and operator contract do not share one manifest")
    if int(snapshot["cohort"]["participants"]) != int(operator["cohort"]["participants"]):
        raise ValueError("Frozen result snapshot and operator cohort differ")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        output_dir / "hyperparameters.tex": build_hyperparameters(baseline, neural, operator),
        output_dir / "weighting_robustness.tex": build_weighting(snapshot),
    }
    if nonlinear_path is not None:
        nonlinear = read_json(nonlinear_path)
        if nonlinear.get("post_hoc") is not True or nonlinear.get("confirmatory_override_allowed"):
            raise ValueError("Nonlinear result does not preserve its post-hoc evidence boundary")
        outputs[output_dir / "nonlinear_baseline.tex"] = build_nonlinear(nonlinear)
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8", newline="\n")
    return list(outputs)


def main() -> None:
    args = parse_args()
    outputs = build_tables(
        args.baseline, args.neural, args.operator, args.snapshot, args.output_dir,
        args.nonlinear,
    )
    print(json.dumps({"generated": [str(path) for path in outputs]}, indent=2))


if __name__ == "__main__":
    main()
