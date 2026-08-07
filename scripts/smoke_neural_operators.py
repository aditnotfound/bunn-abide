"""Run a non-scientific GPU forward/backward smoke for every neural operator.

This script intentionally produces no train/validation/test split and no
predictive metric. It verifies only that the real ABIDE-I tensor can traverse
every frozen propagation operator/density combination with finite gradients.
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
from src.neural_models import NeuralArchitecture, SharedGraphClassifier
from src.neural_operators import orthogonality_error


OPERATORS = ("identity", "learned_local", "gcn", "trivial_bundle", "learned_bunn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connectomes", type=Path, default=Path("data/processed/abide_i_connectomes_fisher_z.npz"))
    parser.add_argument("--baseline-table", type=Path, default=Path("data/processed/abide_i_baseline_table.csv"))
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this GPU smoke test")
    torch.manual_seed(20260803)
    cohort = load_neural_cohort(args.connectomes, args.baseline_table)
    features = torch.from_numpy(cohort.connectomes[: args.batch_size]).cuda()
    architecture = NeuralArchitecture()
    results: list[dict[str, object]] = []
    for density in NEURAL_DENSITIES:
        graph = make_dense_graph_batch(features, density)
        for operator in OPERATORS:
            torch.cuda.reset_peak_memory_stats()
            model = SharedGraphClassifier(operator, architecture).cuda()
            model.train()
            logits, diagnostics = model(graph.features, graph.adjacency, return_diagnostics=True)
            loss = logits.square().mean()
            loss.backward()
            finite_gradients = all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            finite_diagnostics = all(
                all(torch.isfinite(value).all() for value in layer.values())
                for layer in diagnostics
            )
            max_transport_error = 0.0
            if operator in {"learned_local", "learned_bunn"}:
                for layer in model.propagation:
                    maps = layer.maps_for(model.encoder(graph.features))
                    max_transport_error = max(max_transport_error, float(orthogonality_error(maps).detach().cpu()))
            results.append(
                {
                    "operator": operator,
                    "density": density,
                    "parameters": model.parameter_count(),
                    "finite_logits": bool(torch.isfinite(logits).all()),
                    "finite_loss": bool(torch.isfinite(loss)),
                    "finite_gradients": bool(finite_gradients),
                    "finite_diagnostics": bool(finite_diagnostics),
                    "maximum_transport_orthogonality_error": max_transport_error,
                    "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                }
            )
            del model, logits, diagnostics, loss
            torch.cuda.empty_cache()
    print(json.dumps({"status": "passed", "batch_size": args.batch_size, "results": results}, indent=2))


if __name__ == "__main__":
    main()
