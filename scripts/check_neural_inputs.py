"""Validate the frozen ABIDE-I graph inputs required before neural training.

This is a data-contract check only. It does not make splits, fit a scaler,
train a model, or calculate any predictive performance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.neural_data import NEURAL_DENSITIES, load_neural_cohort, make_dense_graph_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connectomes", type=Path, default=Path("data/processed/abide_i_connectomes_fisher_z.npz"))
    parser.add_argument("--baseline-table", type=Path, default=Path("data/processed/abide_i_baseline_table.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cohort = load_neural_cohort(args.connectomes, args.baseline_table)
    connectomes = torch.from_numpy(cohort.connectomes)
    density_summary: dict[str, dict[str, int | bool]] = {}
    for density in NEURAL_DENSITIES:
        graph = make_dense_graph_batch(connectomes, density)
        edges = graph.adjacency.sum(dim=(-1, -2)).div(2)
        density_summary[f"{density:.0%}"] = {
            "undirected_edges_per_graph": int(edges[0].item()),
            "minimum_undirected_edges": int(edges.min().item()),
            "maximum_undirected_edges": int(edges.max().item()),
            "symmetric": bool(torch.equal(graph.adjacency, graph.adjacency.transpose(-1, -2))),
            "zero_diagonal": bool(torch.count_nonzero(torch.diagonal(graph.adjacency, dim1=-2, dim2=-1)) == 0),
        }
    report = {
        "status": "passed",
        "participants": cohort.participants,
        "nodes_per_graph": int(cohort.connectomes.shape[1]),
        "node_feature_dimension": int(cohort.connectomes.shape[2]),
        "sites": int(len(set(cohort.site_ids.tolist()))),
        "class_counts": {"asd": int(cohort.labels.sum()), "control": int((cohort.labels == 0).sum())},
        "densities": density_summary,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
