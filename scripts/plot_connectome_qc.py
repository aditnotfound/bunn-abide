"""Render one Fisher-z connectome as a visual QC artifact, not a result figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--connectomes",
        default="data/processed/abide_i_connectomes_fisher_z.npz",
    )
    parser.add_argument("--subject-index", type=int, default=0)
    parser.add_argument(
        "--output", default="outputs/figures/step6_connectome_qc.png"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = np.load(args.connectomes)
    matrix = data["connectomes_fisher_z"][args.subject_index]
    subject_id = data["subject_id"][args.subject_index]
    site_id = data["site_id"][args.subject_index]
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T):
        raise ValueError("Refusing to plot an invalid connectome")

    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    color_limit = float(np.quantile(np.abs(off_diagonal), 0.99))
    figure, axis = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-color_limit, vmax=color_limit)
    axis.set_title(f"Step 6 QC: Fisher-z connectome\nsubject {subject_id}, site {site_id}")
    axis.set_xlabel("AAL region index")
    axis.set_ylabel("AAL region index")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Fisher-z correlation")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
