"""Generate deterministic LaTeX inputs for the Step 13.2 manuscript.

All scientific values come from the frozen Step 13.1 snapshot. The script
validates the manuscript contract and the complete paper-asset manifest before
writing macros, tables, or the study-design diagram.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


class ManuscriptInputError(ValueError):
    """Raised when frozen manuscript inputs or generated values are invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManuscriptInputError(f"Unreadable JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ManuscriptInputError(f"Expected a JSON object: {path}")
    return payload


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                # OneDrive or antivirus indexing can briefly hold the old file
                # open on Windows. Preserve atomic replacement and retry with a
                # deterministic bounded backoff instead of writing in place.
                time.sleep(0.05 * (attempt + 1))
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_text(path: Path, text: str) -> None:
    atomic_write(path, text.encode("utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def finite_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ManuscriptInputError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(parsed):
        raise ManuscriptInputError(f"{label} is not finite")
    return parsed


def finite_int(value: Any, label: str) -> int:
    parsed = finite_float(value, label)
    if not parsed.is_integer():
        raise ManuscriptInputError(f"{label} is not an integer: {value!r}")
    return int(parsed)


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ManuscriptInputError(f"Unreadable CSV: {path}") from error
    if not rows:
        raise ManuscriptInputError(f"Empty CSV: {path}")
    return rows


def validate_contract(repo_root: Path, contract: dict[str, Any]) -> list[dict[str, Any]]:
    frozen = contract.get("frozen_inputs")
    if not isinstance(frozen, dict) or not frozen:
        raise ManuscriptInputError("Manuscript contract has no frozen inputs")
    records = []
    for relative, expected in sorted(frozen.items()):
        path = repo_root / relative
        if not path.is_file():
            raise ManuscriptInputError(f"Missing manuscript input: {relative}")
        observed = sha256_file(path)
        if observed != expected:
            raise ManuscriptInputError(
                f"Manuscript input digest mismatch for {relative}: {observed} != {expected}"
            )
        records.append({"path": relative, "sha256": observed, "bytes": path.stat().st_size})
    return records


def validate_paper_asset_manifest(repo_root: Path) -> None:
    manifest = load_json(repo_root / "paper/generated/paper_asset_manifest.json")
    assets = manifest.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ManuscriptInputError("Paper-asset manifest contains no assets")
    for record in assets:
        path = repo_root / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ManuscriptInputError(f"Paper asset failed digest validation: {path}")


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def format_number(value: float, digits: int) -> str:
    rendered = f"{value:.{digits}f}"
    return "0" if rendered in {"-0", "-0.0", "-0.00", "-0.000", "-0.0000", "-0.00000"} else rendered


def macro(name: str, value: str) -> str:
    return rf"\providecommand{{\{name}}}{{{value}}}"


def build_numbers(snapshot: dict[str, Any]) -> str:
    cohort = snapshot["cohort"]
    baselines = snapshot["classical_baselines"]
    predictive = snapshot["confirmatory_predictive_contrasts"]
    primary = predictive["learned_bunn_curve_minus_gcn_curve"]
    elastic = predictive["learned_bunn_curve_minus_connectome_elastic_net"]
    representation = snapshot["primary_representation_contrast"]
    robustness = snapshot["robustness"]
    efficiency = snapshot["operator_efficiency"]
    commands = [
        "% Generated by scripts/build_manuscript_inputs.py; do not edit manually.",
        macro("StudyParentRows", str(finite_int(cohort["parent_rows"], "parent_rows"))),
        macro("StudyTechnicalExclusions", str(finite_int(cohort["technical_exclusions"], "technical_exclusions"))),
        macro("StudyParticipants", str(finite_int(cohort["participants"], "participants"))),
        macro("StudySites", str(finite_int(cohort["held_out_sites"], "held_out_sites"))),
        macro("StudyASD", str(finite_int(cohort["asd"], "asd"))),
        macro("StudyControls", str(finite_int(cohort["control"], "control"))),
        macro("StudyROIs", str(finite_int(cohort["roi_count"], "roi_count"))),
        macro("StudyEdges", f"{finite_int(cohort['edge_features'], 'edge_features'):,}"),
        macro("CovariatesBA", format_number(baselines["covariates_l2_logistic"]["equal_site_balanced_accuracy"], 4)),
        macro("ConnectomeBA", format_number(baselines["connectome_elastic_net_logistic"]["equal_site_balanced_accuracy"], 4)),
        macro("CombinedBA", format_number(baselines["combined_elastic_net_logistic"]["equal_site_balanced_accuracy"], 4)),
        macro("BunnGcnEstimate", format_number(primary["estimate"], 5)),
        macro("BunnGcnCILow", format_number(primary["bootstrap_ci_95"][0], 5)),
        macro("BunnGcnCIHigh", format_number(primary["bootstrap_ci_95"][1], 5)),
        macro("BunnGcnP", format_number(primary["exact_sign_flip_p"], 3)),
        macro("PracticalMargin", format_number(primary["practical_margin"], 2)),
        macro("BunnElasticEstimate", format_number(elastic["estimate"], 5)),
        macro("BunnElasticCILow", format_number(elastic["bootstrap_ci_95"][0], 5)),
        macro("BunnElasticCIHigh", format_number(elastic["bootstrap_ci_95"][1], 5)),
        macro("BunnElasticP", format_number(elastic["exact_sign_flip_p"], 4)),
        macro("RankEstimate", format_number(representation["estimate"], 5)),
        macro("RankCILow", format_number(representation["bootstrap_ci_95"][0], 5)),
        macro("RankCIHigh", format_number(representation["bootstrap_ci_95"][1], 5)),
        macro("LeaveOneSitePositive", str(robustness["primary_leave_one_site_out"]["positive_estimates"])),
        macro("SeedPositive", str(robustness["primary_seed_specific"]["positive_estimates"])),
        macro("FavorableIntervals", str(robustness["primary_seed_specific"]["positive_intervals_excluding_zero"])),
        macro("UnfavorableSeedIntervals", str(robustness["primary_seed_specific"]["negative_intervals_excluding_zero"])),
        macro("GcnParameters", f"{efficiency['gcn']['parameter_count']:,}"),
        macro("BunnParameters", f"{efficiency['learned_bunn']['parameter_count']:,}"),
        macro("GcnFitHours", format_number(efficiency["gcn"]["total_runtime_hours"], 2)),
        macro("BunnFitHours", format_number(efficiency["learned_bunn"]["total_runtime_hours"], 2)),
        macro("GcnCurveBA", format_number(efficiency["gcn"]["equal_site_curve_balanced_accuracy"], 4)),
        macro("BunnCurveBA", format_number(efficiency["learned_bunn"]["equal_site_curve_balanced_accuracy"], 4)),
    ]
    return "\n".join(commands) + "\n"


def table_cohort(snapshot: dict[str, Any]) -> str:
    cohort = snapshot["cohort"]
    rows = [
        ("Downloaded AAL time-series files", cohort["parent_rows"]),
        ("Technical exclusions", cohort["technical_exclusions"]),
        ("Retained participants", cohort["participants"]),
        ("ASD / control", f"{cohort['asd']} / {cohort['control']}"),
        ("Held-out sites", cohort["held_out_sites"]),
        ("ROIs / lower-triangle features", f"{cohort['roi_count']} / {cohort['edge_features']:,}"),
    ]
    body = "\n".join(rf"{latex_escape(str(label))} & {value} \\" for label, value in rows)
    return rf"""\begin{{table}}[t]
\centering
\caption{{Frozen cohort and connectome dimensions after technical quality control.}}
\label{{tab:cohort}}
\begin{{tabular}}{{lr}}
\toprule
Item & Value \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def table_baselines(snapshot: dict[str, Any]) -> str:
    order = [
        "covariates_l2_logistic",
        "connectome_elastic_net_logistic",
        "combined_elastic_net_logistic",
    ]
    body = []
    for key in order:
        row = snapshot["classical_baselines"][key]
        body.append(
            rf"{latex_escape(row['label'])} & {row['equal_site_balanced_accuracy']:.4f} & "
            rf"{row['pooled_balanced_accuracy']:.4f} & {row['pooled_auroc']:.4f} \\"
        )
    return """\\begin{table}[t]
\\centering
\\caption{Classical held-out-site baselines. Equal-site balanced accuracy is the primary summary.}
\\label{tab:baselines}
\\begin{tabular}{lccc}
\\toprule
Model & Equal-site BA & Pooled BA & Pooled AUROC \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""


def table_confirmatory(snapshot: dict[str, Any]) -> str:
    labels = {
        "learned_bunn_curve_minus_gcn_curve": "BuNN curve - GCN curve",
        "learned_bunn_curve_minus_trivial_bundle_curve": "BuNN curve - trivial-bundle curve",
        "learned_bunn_nonzero_minus_learned_local": "BuNN nonzero mean - learned-local",
        "learned_bunn_nonzero_minus_identity": "BuNN nonzero mean - identity",
        "learned_bunn_curve_minus_connectome_elastic_net": "BuNN curve - connectome elastic net",
    }
    rows = snapshot["confirmatory_predictive_contrasts"]
    body = []
    for key in labels:
        row = rows[key]
        body.append(
            rf"{labels[key]} & {row['estimate']:.4f} & "
            rf"[{row['bootstrap_ci_95'][0]:.4f}, {row['bootstrap_ci_95'][1]:.4f}] & "
            rf"{row['exact_sign_flip_p']:.4f} \\"
        )
    return """\\begin{table}[t]
\\centering
\\caption{Frozen predictive contrasts. Differences are BuNN minus the named comparator.}
\\label{tab:confirmatory}
\\small
\\begin{tabularx}{\\textwidth}{Xrrr}
\\toprule
Contrast & Estimate & 95\\% site-bootstrap CI & Exact $p$ \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabularx}
\\end{table}
"""


def table_efficiency(snapshot: dict[str, Any]) -> str:
    order = ["gcn", "trivial_bundle", "learned_bunn"]
    body = []
    for key in order:
        row = snapshot["operator_efficiency"][key]
        body.append(
            rf"{latex_escape(row['label'])} & {row['parameter_count']:,} & {row['total_runtime_hours']:.2f} & "
            rf"{row['mean_runtime_seconds']:.2f} & {row['maximum_peak_gpu_memory_gib']:.3f} & "
            rf"{row['equal_site_curve_balanced_accuracy']:.4f} \\"
        )
    return """\\begin{table}[!h]
\\centering
\\caption{Observed execution cost and predictive curve summary for the three diffusion operators. Runtime is implementation- and hardware-specific.}
\\label{tab:efficiency}
\\small
\\begin{tabular}{lrrrrr}
\\toprule
Operator & Params. & Fit h & s/fit & Peak GiB & Curve BA \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""


def table_representation(repo_root: Path) -> str:
    rows = read_csv(repo_root / "paper/generated/tables/representation_contrasts.csv")
    labels = {
        "normalized_effective_rank": "Normalized effective rank",
        "normalized_dispersion": "Normalized dispersion",
        "mean_pairwise_cosine": "Mean pairwise cosine similarity",
        "invariant_edge_transport_distance": "Invariant edge-transport distance",
        "encoder_to_layer_2_normalized_effective_rank_change": "Encoder-to-layer-2 effective-rank change",
    }
    body = []
    for row in rows:
        body.append(
            rf"{labels[row['endpoint']]} & {float(row['estimate']):.4f} & "
            rf"[{float(row['ci_95_low']):.4f}, {float(row['ci_95_high']):.4f}] \\"
        )
    return """\\begin{table}[t]
\\centering
\\caption{BuNN-minus-GCN matched-anchor representation contrasts at layer 2.}
\\label{tab:representation}
\\small
\\begin{tabularx}{\\textwidth}{Xrr}
\\toprule
Endpoint & Estimate & 95\\% site-bootstrap CI \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabularx}
\\end{table}
"""


def supplement_representation_tabular(repo_root: Path) -> str:
    rows = read_csv(repo_root / "paper/generated/tables/representation_contrasts.csv")
    labels = {
        "normalized_effective_rank": "Normalized effective rank",
        "normalized_dispersion": "Normalized dispersion",
        "mean_pairwise_cosine": "Mean pairwise cosine similarity",
        "invariant_edge_transport_distance": "Invariant edge-transport distance",
        "encoder_to_layer_2_normalized_effective_rank_change": "Encoder-to-layer-2 effective-rank change",
    }
    body = []
    for row in rows:
        body.append(
            rf"{labels[row['endpoint']]} & {float(row['estimate']):.4f} & "
            rf"[{float(row['ci_95_low']):.4f}, {float(row['ci_95_high']):.4f}] \\"
        )
    return """\\begin{center}
\\captionof{table}{BuNN-minus-GCN matched-anchor representation contrasts at layer 2.}
\\label{tab:supp-representation}
\\small
\\begin{tabularx}{\\textwidth}{Xrr}
\\toprule
Endpoint & Estimate & 95\\% site-bootstrap CI \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabularx}
\\end{center}
"""


def table_robustness(snapshot: dict[str, Any]) -> str:
    labels = {
        "learned_bunn_curve_minus_gcn_curve": "BuNN - GCN curve",
        "learned_bunn_curve_minus_connectome_elastic_net": "BuNN - elastic net",
        "learned_bunn_minus_gcn_matched_anchor_effective_rank_change": "BuNN - GCN effective-rank change",
    }
    body = []
    for row in snapshot["robustness"]["alternative_summaries"]:
        body.append(
            rf"{labels[row['contrast']]} & {latex_escape(row['summary'].replace('_', ' '))} & "
            rf"{row['estimate']:.5f} \\"
        )
    return """\\begin{table}[t]
\\centering
\\caption{Pre-listed alternative summaries. Equal-site means remain confirmatory.}
\\label{tab:robustness-summaries}
\\small
\\begin{tabularx}{\\textwidth}{Xlr}
\\toprule
Contrast & Summary & Estimate \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabularx}
\\end{table}
"""


def supplement_robustness_tabular(snapshot: dict[str, Any]) -> str:
    labels = {
        "learned_bunn_curve_minus_gcn_curve": "BuNN - GCN curve",
        "learned_bunn_curve_minus_connectome_elastic_net": "BuNN - elastic net",
        "learned_bunn_minus_gcn_matched_anchor_effective_rank_change": "BuNN - GCN effective-rank change",
    }
    body = []
    for row in snapshot["robustness"]["alternative_summaries"]:
        body.append(
            rf"{labels[row['contrast']]} & {latex_escape(row['summary'].replace('_', ' '))} & "
            rf"{row['estimate']:.5f} \\"
        )
    return """\\begin{center}
\\captionof{table}{Pre-listed alternative summaries. Equal-site means remain confirmatory.}
\\label{tab:supp-robustness-summaries}
\\small
\\begin{tabularx}{\\textwidth}{Xlr}
\\toprule
Contrast & Summary & Estimate \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabularx}
\\end{center}
"""


def plot_study_design(snapshot: dict[str, Any], output: Path) -> None:
    cohort = snapshot["cohort"]
    boxes = [
        (0.02, 0.58, 0.20, 0.28, "ABIDE-I PCP\nAAL ROI time series", "#DCEAF7"),
        (0.27, 0.58, 0.20, 0.28, f"Technical QC\n{cohort['participants']} participants, {cohort['held_out_sites']} sites", "#E3F2E1"),
        (0.52, 0.58, 0.20, 0.28, f"Fisher-z connectomes\n{cohort['roi_count']} ROIs; {cohort['edge_features']:,} features", "#FFF0D5"),
        (0.77, 0.58, 0.21, 0.28, "Positive-edge graphs\n0%, 1%, 5%, 10%, 20%", "#F5E1EC"),
        (0.10, 0.10, 0.22, 0.28, "Nested held-out-site\nevaluation and tuning", "#E8E2F5"),
        (0.39, 0.10, 0.22, 0.28, "Shared backbone\n5 operators / controls", "#DFF0EE"),
        (0.68, 0.10, 0.22, 0.28, "Frozen prediction,\nrepresentation, robustness", "#F4E7D7"),
    ]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        fig, axis = plt.subplots(figsize=(10.2, 4.2))
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        for x, y, width, height, text, color in boxes:
            patch = FancyBboxPatch(
                (x, y), width, height, boxstyle="round,pad=0.015,rounding_size=0.02",
                linewidth=1.1, edgecolor="#444444", facecolor=color,
            )
            axis.add_patch(patch)
            axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9)
        arrows = [
            ((0.22, 0.72), (0.27, 0.72)), ((0.47, 0.72), (0.52, 0.72)),
            ((0.72, 0.72), (0.77, 0.72)), ((0.875, 0.58), (0.79, 0.38)),
            ((0.68, 0.24), (0.61, 0.24)), ((0.39, 0.24), (0.32, 0.24)),
        ]
        for start, end in arrows:
            axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#555555"))
        axis.text(0.5, 0.96, "Frozen ABIDE-I operator-audit workflow", ha="center", va="center", fontsize=11, weight="bold")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output, dpi=300, bbox_inches="tight", facecolor="white",
            metadata={"Software": "bunn-abide Step 13.2 manuscript-input builder"},
        )
        plt.close(fig)


def build_manuscript_inputs(repo_root: Path, contract_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    contract = load_json(contract_path)
    frozen_records = validate_contract(repo_root, contract)
    validate_paper_asset_manifest(repo_root)
    snapshot = load_json(repo_root / "reproducibility/result_snapshot.json")
    if snapshot.get("evidence_commit") != contract.get("evidence_commit"):
        raise ManuscriptInputError("Snapshot evidence commit does not match manuscript contract")
    if snapshot["step11_decision"].get("all_three_conditions") is not False:
        raise ManuscriptInputError("Manuscript builder requires the frozen negative/null Step 11 decision")

    outputs: dict[Path, bytes] = {
        repo_root / "paper/generated/numbers.tex": build_numbers(snapshot).encode("utf-8"),
        repo_root / "paper/generated/tex/table_cohort.tex": table_cohort(snapshot).encode("utf-8"),
        repo_root / "paper/generated/tex/table_baselines.tex": table_baselines(snapshot).encode("utf-8"),
        repo_root / "paper/generated/tex/table_confirmatory.tex": table_confirmatory(snapshot).encode("utf-8"),
        repo_root / "paper/generated/tex/table_efficiency.tex": table_efficiency(snapshot).encode("utf-8"),
        repo_root / "paper/generated/tex/table_representation.tex": table_representation(repo_root).encode("utf-8"),
        repo_root / "paper/generated/tex/table_robustness.tex": table_robustness(snapshot).encode("utf-8"),
        repo_root / "paper/generated/tex/supplement_representation_tabular.tex": supplement_representation_tabular(repo_root).encode("utf-8"),
        repo_root / "paper/generated/tex/supplement_robustness_tabular.tex": supplement_robustness_tabular(snapshot).encode("utf-8"),
    }
    for path, payload in outputs.items():
        atomic_write(path, payload)
    study_design = repo_root / "paper/generated/figures/study_design.png"
    plot_study_design(snapshot, study_design)

    generated_paths = sorted([*outputs, study_design])
    generated_records = [
        {
            "path": path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in generated_paths
    ]
    manifest = {
        "schema_version": "step13_manuscript_input_manifest_v1",
        "contract_version": contract["contract_version"],
        "evidence_commit": contract["evidence_commit"],
        "paper_assets_commit": contract["paper_assets_commit"],
        "frozen_inputs": frozen_records,
        "generated_inputs": generated_records,
    }
    manifest_path = repo_root / "paper/generated/manuscript_input_manifest.json"
    write_json(manifest_path, manifest)

    produced = {record["path"] for record in generated_records}
    produced.add(manifest_path.relative_to(repo_root).as_posix())
    expected = set(contract["required_generated_outputs"])
    if produced != expected:
        raise ManuscriptInputError(
            f"Generated manuscript-input mismatch: missing={sorted(expected - produced)}, "
            f"unexpected={sorted(produced - expected)}"
        )
    return {
        "contract_version": contract["contract_version"],
        "validated_frozen_inputs": len(frozen_records),
        "validated_paper_assets": len(load_json(repo_root / "paper/generated/paper_asset_manifest.json")["assets"]),
        "generated_outputs": sorted(produced),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--contract", type=Path, default=Path("configs/manuscript_v1.json"))
    arguments = parser.parse_args()
    root = arguments.repo_root.resolve()
    contract = arguments.contract if arguments.contract.is_absolute() else root / arguments.contract
    print(json.dumps(build_manuscript_inputs(root, contract), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
