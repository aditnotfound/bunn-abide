"""Run the single acknowledged E1 analysis under the frozen protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.run_baselines import write_json_atomic
from src.extensions.e1_analysis import DIAGNOSTICS, INTERVENTIONS, load_and_compute, sha256_file


COLORS = {
    "identity_maps": "#5B6770",
    "node_map_shuffle": "#D55E00",
    "random_orthogonal_maps": "#0072B2",
    "degree_preserving_topology": "#009E73",
}
LABELS = {
    "identity_maps": "Identity maps",
    "node_map_shuffle": "Shuffled learned maps",
    "random_orthogonal_maps": "Random orthogonal maps",
    "degree_preserving_topology": "Degree-preserving rewiring",
}
DIAGNOSTIC_LABELS = {
    "normalized_effective_rank": "Effective rank",
    "normalized_dispersion": "Dispersion",
    "mean_pairwise_cosine": "Pairwise cosine",
    "invariant_edge_transport_distance": "Edge-transport distance",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("configs/extensions/e1_analysis_v1.json"))
    parser.add_argument(
        "--source-root", type=Path,
        default=Path("outputs/extensions/e1_interventions_v1/accepted"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/analysis/e1_interventions_v1"),
    )
    parser.add_argument("--acknowledgement", required=True)
    return parser.parse_args()


def write_csv_atomic(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False, float_format="%.12g", lineterminator="\n")
    os.replace(temporary, path)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path, dpi=300, bbox_inches="tight", facecolor="white",
        metadata={"Software": "bunn-abide E1 analysis v1"},
    )
    plt.close(fig)


def plot_primary(table: pd.DataFrame, path: Path) -> None:
    ordered = table.set_index("intervention").loc[list(INTERVENTIONS)].reset_index()
    fig, axis = plt.subplots(figsize=(7.2, 3.6))
    positions = np.arange(len(ordered))[::-1]
    for position, row in zip(positions, ordered.itertuples(index=False), strict=True):
        axis.errorbar(
            row.estimate_pp, position,
            xerr=[[row.estimate_pp - row.ci_lower_pp], [row.ci_upper_pp - row.estimate_pp]],
            fmt="o", markersize=6, capsize=3, linewidth=1.5,
            color=COLORS[row.intervention],
        )
    axis.axvline(0, color="#202A33", linewidth=0.9, linestyle="--")
    axis.set_yticks(positions, [LABELS[value] for value in ordered.intervention])
    axis.set_xlabel("Change in held-out balanced accuracy (percentage points)")
    axis.set_title("Accepted-checkpoint intervention effects")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", color="#D9DEE3", linewidth=0.6)
    fig.tight_layout()
    save_figure(fig, path)


def plot_density(table: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    for intervention in INTERVENTIONS:
        subset = table[table.intervention == intervention].sort_values("density")
        density_percent = 100.0 * subset.density.to_numpy()
        estimate = subset.estimate_pp.to_numpy()
        axis.plot(
            density_percent, estimate, marker="o", linewidth=1.5,
            color=COLORS[intervention], label=LABELS[intervention],
        )
        axis.fill_between(
            density_percent, subset.ci_lower_pp, subset.ci_upper_pp,
            color=COLORS[intervention], alpha=0.10, linewidth=0,
        )
    axis.axhline(0, color="#202A33", linewidth=0.9, linestyle="--")
    axis.set_xticks([1, 5, 10, 20])
    axis.set_xlabel("Positive-edge density (%)")
    axis.set_ylabel("Balanced-accuracy change (percentage points)")
    axis.set_title("Intervention effects across graph density")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(color="#D9DEE3", linewidth=0.6)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    save_figure(fig, path)


def plot_associations(table: pd.DataFrame, path: Path) -> None:
    matrix = table.pivot(index="intervention", columns="diagnostic", values="spearman_rho").loc[
        list(INTERVENTIONS), list(DIAGNOSTICS)
    ]
    fig, axis = plt.subplots(figsize=(7.2, 3.5))
    image = axis.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(np.arange(len(DIAGNOSTICS)), [DIAGNOSTIC_LABELS[value] for value in DIAGNOSTICS], rotation=22, ha="right")
    axis.set_yticks(np.arange(len(INTERVENTIONS)), [LABELS[value] for value in INTERVENTIONS])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8,
                      color="white" if abs(value) > 0.55 else "#202A33")
    axis.set_title("Representation and prediction changes (descriptive Spearman ρ)")
    fig.colorbar(image, ax=axis, shrink=0.78, label="Spearman ρ")
    fig.tight_layout()
    save_figure(fig, path)


def verify_frozen_sources(protocol: dict[str, object]) -> None:
    source = protocol["source"]
    checks = {
        Path(source["sealed_archive"]): source["sealed_archive_sha256"],
        Path("configs/extensions/e1_checkpoint_interventions_v1.json"): source["intervention_contract_sha256"],
        Path("scripts/run_e1_interventions.py"): source["runner_sha256"],
        Path("scripts/audit_e1_interventions.py"): source["score_blind_auditor_sha256"],
    }
    for path, expected in checks.items():
        if sha256_file(path) != expected:
            raise ValueError(f"Frozen E1 source hash mismatch: {path}")


def run(args: argparse.Namespace) -> Path:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != 1:
        raise ValueError("Unsupported E1 analysis protocol")
    if args.acknowledgement != protocol["unblinding"]["required_acknowledgement"]:
        raise ValueError("Exact E1 unblinding acknowledgement is required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("E1 analysis output directory is not empty")
    verify_frozen_sources(protocol)
    tables = load_and_compute(args.source_root, protocol)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    for name, table in tables.items():
        write_csv_atomic(table, args.output_dir / f"{name}.csv")
    plot_primary(tables["primary_contrasts"], args.output_dir / "primary_intervention_forest.png")
    plot_density(tables["density_contrasts"], args.output_dir / "density_intervention_effects.png")
    plot_associations(
        tables["representation_prediction_associations"],
        args.output_dir / "representation_prediction_associations.png",
    )

    artifact_names = sorted(path.name for path in args.output_dir.iterdir() if path.is_file())
    artifact_hashes = {name: sha256_file(args.output_dir / name) for name in artifact_names}
    metadata = {
        "state": "complete_pending_independent_audit",
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol_path": str(args.protocol), "protocol_sha256": sha256_file(args.protocol),
        "source_root": str(args.source_root), "source_archive_sha256": protocol["source"]["sealed_archive_sha256"],
        "artifact_hashes": artifact_hashes, "scientific_values_printed_to_terminal": False,
        "post_hoc_extension": True, "frozen_study_1_unchanged": True,
    }
    write_json_atomic(args.output_dir / "analysis_metadata.json", metadata)
    print(json.dumps({
        "state": metadata["state"], "artifact_count": len(artifact_hashes),
        "scientific_values_displayed": False,
    }))
    return args.output_dir


if __name__ == "__main__":
    run(parse_args())
