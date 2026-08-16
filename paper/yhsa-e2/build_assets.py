"""Build descriptive, non-inferential figures for the YHSA-format report."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
OUT = Path(__file__).resolve().parent / "generated" / "figures"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": "#263746",
            "axes.linewidth": 0.8,
            "xtick.color": "#263746",
            "ytick.color": "#263746",
            "text.color": "#1f2d38",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def site_composition() -> None:
    frame = pd.read_csv(DATA / "abide_i_baseline_table.csv")
    counts = (
        frame.groupby(["site_id", "label_asd"]).size().unstack(fill_value=0).rename(columns={0: "Control", 1: "ASD"})
    )
    counts = counts.sort_values(counts.columns.tolist(), ascending=False)
    y = np.arange(len(counts))

    fig, ax = plt.subplots(figsize=(7.25, 5.25))
    ax.barh(y, counts["Control"], color="#2F6B8A", label="Control", height=0.68)
    ax.barh(y, counts["ASD"], left=counts["Control"], color="#B4573D", label="ASD", height=0.68)
    ax.set_yticks(y, counts.index)
    ax.invert_yaxis()
    ax.set_xlabel("Technically eligible participants")
    ax.set_ylabel("ABIDE-I site")
    ax.grid(axis="x", color="#d8dee3", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    for i, total in enumerate(counts.sum(axis=1)):
        ax.text(total + 1.2, i, str(int(total)), va="center", fontsize=8, color="#51616e")
    ax.set_xlim(0, counts.sum(axis=1).max() + 17)
    fig.tight_layout()
    fig.savefig(OUT / "site_composition.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def example_connectome() -> None:
    archive = np.load(DATA / "abide_i_connectomes_fisher_z.npz")
    matrices = archive["connectomes_fisher_z"]
    # A deterministic, non-selected example: the first row of the frozen archive.
    matrix = matrices[0]
    bound = float(np.quantile(np.abs(matrix[np.tril_indices_from(matrix, k=-1)]), 0.99))

    fig, ax = plt.subplots(figsize=(6.6, 5.35))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-bound, vmax=bound, interpolation="nearest")
    ax.set_xlabel("AAL region index")
    ax.set_ylabel("AAL region index")
    ax.spines[:].set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    bar.set_label("Fisher-z transformed correlation")
    fig.tight_layout()
    fig.savefig(OUT / "example_connectome.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    site_composition()
    example_connectome()


if __name__ == "__main__":
    main()
