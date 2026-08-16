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
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


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
    return rf"""\begin{{table}}[!t]
\centering
\caption{{Cohort retained after technical quality control and resulting connectome dimensions.}}
\label{{tab:cohort}}
\footnotesize
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
    return """\\begin{table}[!t]
\\centering
\\caption{Classical held-out-site baselines. Equal-site balanced accuracy is the primary summary.}
\\label{tab:baselines}
\\footnotesize
\\begin{tabularx}{\\columnwidth}{Xccc}
\\toprule
Model & Site BA & Pooled BA & AUROC \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabularx}
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
    return """\\begin{table*}[!t]
\\centering
\\caption{Pre-specified predictive contrasts. Differences are BuNN minus the named comparator.}
\\label{tab:confirmatory}
\\footnotesize
\\begin{tabularx}{\\textwidth}{Xrrr}
\\toprule
Contrast & Estimate & 95\\% site-bootstrap CI & Exact $p$ \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabularx}
\\end{table*}
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
    return """\\begin{table*}[!t]
\\centering
\\caption{Observed execution cost and predictive curve summary for the three diffusion operators. Runtime is implementation- and hardware-specific.}
\\label{tab:efficiency}
\\footnotesize
\\begin{tabular}{lrrrrr}
\\toprule
Operator & Params. & Fit h & s/fit & Peak GiB & Curve BA \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabular}
\\end{table*}
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
    return """\\begin{table*}[!t]
\\centering
\\caption{BuNN-minus-GCN matched-anchor representation contrasts at layer 2.}
\\label{tab:representation}
\\footnotesize
\\begin{tabularx}{\\textwidth}{Xrr}
\\toprule
Endpoint & Estimate & 95\\% site-bootstrap CI \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabularx}
\\end{table*}
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
    return """\\begin{table}[H]
\\centering
\\caption{BuNN-minus-GCN matched-anchor representation contrasts at layer 2.}
\\label{tab:supp-representation}
\\footnotesize
\\begin{tabularx}{\\textwidth}{Xrr}
\\toprule
Endpoint & Estimate & 95\\% site-bootstrap CI \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabularx}
\\end{table}
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
    return """\\begin{table*}[!t]
\\centering
\\caption{Pre-listed alternative summaries. Equal-site means remain confirmatory.}
\\label{tab:robustness-summaries}
\\footnotesize
\\begin{tabularx}{\\textwidth}{Xlr}
\\toprule
Contrast & Summary & Estimate \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabularx}
\\end{table*}
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
    return """\\begin{table}[H]
\\centering
\\caption{Pre-listed alternative summaries. Equal-site means remain confirmatory.}
\\label{tab:supp-robustness-summaries}
\\footnotesize
\\begin{tabularx}{\\textwidth}{Xlr}
\\toprule
Contrast & Summary & Estimate \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabularx}
\\end{table}
"""


def _draw_network(
    axis: Any,
    origin_x: float,
    origin_y: float,
    density: int,
    ink: str,
    accent: str,
    faint: str,
) -> None:
    nodes = [
        (-4.0, 0.0), (-2.0, 4.0), (2.0, 4.6),
        (4.2, 0.5), (2.4, -3.8), (-2.4, -3.6),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
        (0, 2), (1, 3), (2, 4), (3, 5), (4, 0), (5, 1),
    ]
    visible = min(len(edges), max(0, density))
    for index, (left, right) in enumerate(edges):
        x1, y1 = nodes[left]
        x2, y2 = nodes[right]
        axis.plot(
            [origin_x + x1, origin_x + x2],
            [origin_y + y1, origin_y + y2],
            color=accent if index < visible else faint,
            linewidth=1.05 if index < visible else 0.45,
            zorder=1,
        )
    for x, y in nodes:
        axis.add_patch(
            Circle(
                (origin_x + x, origin_y + y),
                0.62,
                facecolor="white",
                edgecolor=ink,
                linewidth=0.85,
                zorder=2,
            )
        )


def plot_study_design(snapshot: dict[str, Any], output: Path) -> None:
    """Draw a publication-style study overview on a plain white field."""
    cohort = snapshot["cohort"]
    background = "white"
    ink = "#333333"
    blue = "#35689B"
    orange = "#E17C05"
    green = "#3B8F3B"
    muted = "#666666"
    faint = "#D9D9D9"

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8.5}):
        fig, axis = plt.subplots(figsize=(10.8, 3.65))
        fig.subplots_adjust(left=0.02, right=0.985, top=0.95, bottom=0.08)
        axis.set_xlim(0, 100)
        axis.set_ylim(0, 32)
        axis.axis("off")
        axis.set_facecolor(background)
        fig.patch.set_facecolor(background)

        stages = [
            (2.0, "(a)", "Cohort"),
            (27.0, "(b)", "Connectomes"),
            (52.0, "(c)", "Operators"),
            (77.0, "(d)", "Evaluation"),
        ]
        for x, letter, label in stages:
            axis.text(
                x,
                28.0,
                f"{letter}  {label}",
                color=ink,
                fontsize=9.2,
                weight="bold",
                va="top",
            )

        # Cohort: site marks above three regional time series.
        for site in range(cohort["held_out_sites"]):
            x = 3.0 + (site % 9) * 2.15
            y = 20.6 - (site // 9) * 2.25
            height = 0.9 + (site % 4) * 0.28
            axis.plot(
                [x, x],
                [y, y + height],
                color=blue if site % 5 == 0 else "#8BA6C0",
                linewidth=1.15,
            )
        waveform_x = [3.0 + index * 0.75 for index in range(25)]
        for row in range(3):
            waveform_y = [
                11.8
                - row * 2.0
                + 0.55 * math.sin(index * (0.64 + row * 0.07) + row)
                + 0.18 * math.cos(index * 1.7)
                for index in range(25)
            ]
            axis.plot(
                waveform_x,
                waveform_y,
                color=ink if row == 0 else muted,
                linewidth=0.8,
            )
        axis.text(
            2.5,
            4.5,
            f"{cohort['participants']} participants / {cohort['held_out_sites']} sites\n"
            "AAL regional time series",
            color=ink,
            fontsize=8.1,
            linespacing=1.55,
            va="bottom",
        )

        # Connectomes: a correlation matrix paired with a thresholded graph.
        matrix_x, matrix_y, cell = 28.4, 17.6, 1.15
        for row in range(6):
            for column in range(6):
                if row == column:
                    shade = background
                else:
                    value = (row * 3 + column * 5) % 7
                    shade = blue if value in {0, 1} else (ink if value == 2 else faint)
                axis.add_patch(
                    Rectangle(
                        (matrix_x + column * cell, matrix_y - row * cell),
                        cell * 0.78,
                        cell * 0.78,
                        facecolor=shade,
                        edgecolor="none",
                    )
                )
        _draw_network(axis, 42.0, 15.0, 8, ink, blue, faint)
        axis.text(
            27.8,
            4.5,
            f"{cohort['roi_count']} regions / {cohort['edge_features']:,} edges\n"
            "positive density: 0, 1, 5, 10, 20%",
            color=ink,
            fontsize=8.1,
            linespacing=1.55,
            va="bottom",
        )

        # Operators: five treatments share the same input and readout.
        axis.plot([54.5, 54.5], [9.0, 21.0], color=ink, linewidth=1.0)
        axis.plot([71.8, 71.8], [9.0, 21.0], color=ink, linewidth=1.0)
        labels = [
            "identity",
            "learned local",
            "GCN",
            "trivial bundle",
            "learned BuNN",
        ]
        operator_colors = [muted, muted, blue, orange, green]
        for index, label in enumerate(labels):
            y = 20.5 - index * 2.75
            operator_color = operator_colors[index]
            axis.add_patch(
                Circle(
                    (54.5, y),
                    0.42,
                    facecolor=background,
                    edgecolor=ink,
                    linewidth=0.8,
                )
            )
            axis.plot([54.9, 60.0], [y, y], color=muted, linewidth=0.75)
            if index >= 2:
                axis.plot([60.0, 63.0], [y, y], color=operator_color, linewidth=1.35)
            if index in {1, 4}:
                axis.plot(
                    [61.3, 62.2], [y - 0.5, y + 0.5], color=green, linewidth=0.9
                )
            axis.plot([63.0, 71.4], [y, y], color=muted, linewidth=0.75)
            axis.add_patch(
                Circle(
                    (71.8, y),
                    0.42,
                    facecolor=ink,
                    edgecolor=ink,
                    linewidth=0.8,
                )
            )
            axis.text(
                63.4,
                y + 0.56,
                label,
                color=ink,
                fontsize=6.8,
                ha="center",
                va="bottom",
            )
        axis.text(
            52.8,
            4.5,
            "one backbone / five controlled variants\n"
            "only propagation and map capacity change",
            color=ink,
            fontsize=8.1,
            linespacing=1.55,
            va="bottom",
        )

        # Evaluation: repeated held-out-site splits and paired density curves.
        block_x = 79.0
        for column in range(6):
            for row in range(3):
                test = column == 4
                axis.add_patch(
                    Rectangle(
                        (block_x + column * 1.55, 19.7 - row * 1.55),
                        1.0,
                        1.0,
                        facecolor=blue if test else "none",
                        edgecolor=blue if test else muted,
                        linewidth=0.7,
                    )
                )
        axis.text(
            89.7,
            17.9,
            "one site held out",
            color=muted,
            fontsize=6.7,
            rotation=90,
            va="center",
        )
        curve_x = [79.0, 82.3, 85.6, 88.9, 92.2, 95.5]
        curve_y_a = [13.6, 13.0, 12.2, 11.4, 10.7, 10.1]
        curve_y_b = [13.1, 12.5, 11.9, 11.3, 10.5, 9.8]
        axis.plot(curve_x, curve_y_a, color=blue, linewidth=1.25)
        axis.plot(curve_x, curve_y_b, color=green, linewidth=1.25)
        for x, y in zip(curve_x, curve_y_b):
            axis.add_patch(
                Circle(
                    (x, y),
                    0.25,
                    facecolor=background,
                    edgecolor=green,
                    linewidth=0.8,
                )
            )
        axis.text(
            77.8,
            3.3,
            "nested site-held-out validation\n"
            "paired density curves\n"
            "common-frame diagnostics",
            color=ink,
            fontsize=8.1,
            linespacing=1.55,
            va="bottom",
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output,
            dpi=300,
            facecolor=background,
            metadata={"Software": "bunn-abide manuscript-input builder"},
        )
        plt.close(fig)


def plot_operator_schematic(output: Path) -> None:
    """Show the operator distinction without implying biological transport."""
    background = "white"
    ink = "#333333"
    gcn = "#35689B"
    bunn = "#3B8F3B"
    muted = "#666666"
    faint = "#D9D9D9"

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8.5}):
        fig, axis = plt.subplots(figsize=(10.8, 3.25))
        fig.subplots_adjust(left=0.02, right=0.985, top=0.94, bottom=0.08)
        axis.set_xlim(0, 100)
        axis.set_ylim(0, 28)
        axis.axis("off")
        axis.set_facecolor(background)
        fig.patch.set_facecolor(background)
        panels = [
            (
                2.5,
                "(a)  GCN",
                "Average neighboring features\nwithout learned transport",
                gcn,
            ),
            (
                36.0,
                "(b)  Learned-local",
                "Learn node-wise maps\nwithout exchanging information",
                muted,
            ),
            (
                69.0,
                "(c)  Learned BuNN",
                "Transport into a common frame,\nthen diffuse across edges",
                bunn,
            ),
        ]
        for left, heading, description, color in panels:
            axis.text(
                left, 25.5, heading, color=color, fontsize=9.2, weight="bold", va="top"
            )
            axis.text(
                left,
                4.2,
                description,
                color=ink,
                fontsize=8.0,
                linespacing=1.5,
                va="top",
            )

        neighbor_positions = [(7, 15.5), (13, 19.0), (13, 12.0)]
        target = (26.5, 15.5)
        for x, y in neighbor_positions:
            axis.add_patch(
                Circle(
                    (x, y),
                    1.05,
                    facecolor=background,
                    edgecolor=ink,
                    linewidth=0.85,
                )
            )
            axis.add_patch(
                FancyArrowPatch(
                    (x + 1.1, y),
                    (target[0] - 1.2, target[1]),
                    arrowstyle="-|>",
                    mutation_scale=8,
                    linewidth=1.0,
                    color=gcn,
                )
            )
        axis.add_patch(Circle(target, 1.25, facecolor=gcn, edgecolor=gcn))
        axis.text(
            target[0],
            target[1] - 0.05,
            r"$\Sigma$",
            color=background,
            fontsize=10,
            ha="center",
            va="center",
        )

        local_nodes = [(42.0, 15.5, 20), (50.5, 18.2, -25), (58.5, 13.3, 42)]
        for x, y, angle in local_nodes:
            axis.add_patch(
                Circle(
                    (x, y),
                    1.05,
                    facecolor=background,
                    edgecolor=ink,
                    linewidth=0.85,
                )
            )
            length = 2.3
            dx = length * math.cos(math.radians(angle))
            dy = length * math.sin(math.radians(angle))
            axis.add_patch(
                FancyArrowPatch(
                    (x, y),
                    (x + dx, y + dy),
                    arrowstyle="-|>",
                    mutation_scale=8,
                    linewidth=1.0,
                    color=bunn,
                )
            )
        axis.text(
            50.2,
            10.0,
            "no cross-node path",
            color=muted,
            fontsize=7.0,
            ha="center",
        )

        source_positions = [(73.5, 19.0, 35), (73.5, 15.5, -30), (73.5, 12.0, 70)]
        align_x = 84.0
        target_x = 94.0
        for x, y, angle in source_positions:
            axis.add_patch(
                Circle(
                    (x, y),
                    0.9,
                    facecolor=background,
                    edgecolor=ink,
                    linewidth=0.8,
                )
            )
            dx = 1.8 * math.cos(math.radians(angle))
            dy = 1.8 * math.sin(math.radians(angle))
            axis.add_patch(
                FancyArrowPatch(
                    (x, y),
                    (x + dx, y + dy),
                    arrowstyle="-|>",
                    mutation_scale=7,
                    linewidth=0.9,
                    color=muted,
                )
            )
            axis.add_patch(
                FancyArrowPatch(
                    (x + 1.0, y),
                    (align_x - 1.0, y),
                    arrowstyle="->",
                    mutation_scale=7,
                    linewidth=0.65,
                    color=faint,
                )
            )
            axis.add_patch(
                FancyArrowPatch(
                    (align_x, y),
                    (align_x + 1.8, y),
                    arrowstyle="-|>",
                    mutation_scale=7,
                    linewidth=1.0,
                    color=bunn,
                )
            )
            axis.add_patch(
                FancyArrowPatch(
                    (align_x + 2.0, y),
                    (target_x - 1.2, 15.5),
                    arrowstyle="-|>",
                    mutation_scale=7,
                    linewidth=0.9,
                    color=bunn,
                )
            )
        axis.plot(
            [align_x - 1.4, align_x - 1.4],
            [10.7, 20.3],
            color=faint,
            linewidth=0.7,
            linestyle=(0, (2, 2)),
        )
        axis.text(
            align_x - 1.4,
            21.0,
            "common frame",
            color=muted,
            fontsize=6.8,
            ha="center",
        )
        axis.add_patch(
            Circle((target_x, 15.5), 1.25, facecolor=bunn, edgecolor=bunn)
        )
        axis.text(
            target_x,
            15.45,
            r"$\Sigma$",
            color=background,
            fontsize=10,
            ha="center",
            va="center",
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output,
            dpi=300,
            facecolor=background,
            metadata={"Software": "bunn-abide manuscript-input builder"},
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
    operator_schematic = repo_root / "paper/generated/figures/operator_schematic.png"
    plot_operator_schematic(operator_schematic)

    generated_paths = sorted([*outputs, study_design, operator_schematic])
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
