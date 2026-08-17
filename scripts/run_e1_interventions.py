"""Run the frozen, score-blind E1 accepted-checkpoint intervention audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.run_baselines import publish_sns_notification, sha256_file, write_json_atomic
from src.extensions.e1_interventions import (
    degree_preserving_rewire,
    deterministic_derangement,
    forward_intervened,
)
from src.neural_data import load_neural_cohort, make_dense_graph_batch
from src.neural_models import SharedGraphClassifier
from src.neural_operators import graph_heat_kernel


INTERVENTIONS = (
    "unaltered", "identity_maps", "node_map_shuffle", "random_orthogonal_maps",
    "degree_preserving_topology", "encoded_node_permutation_equivariance",
)
RANDOM_INTERVENTIONS = {"node_map_shuffle", "random_orthogonal_maps", "degree_preserving_topology"}
METRICS = (
    "normalized_effective_rank", "normalized_dispersion", "mean_pairwise_cosine",
    "invariant_edge_transport_distance",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="CALTECH")
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--contract", type=Path, default=Path("configs/extensions/e1_checkpoint_interventions_v1.json"))
    parser.add_argument("--inputs", type=Path, default=Path("outputs/extensions/e1_interventions_v1/inputs"))
    parser.add_argument("--connectomes", type=Path, default=Path("data/processed/abide_i_connectomes_fisher_z.npz"))
    parser.add_argument("--table", type=Path, default=Path("data/processed/abide_i_baseline_table.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/extensions/e1_interventions_v1/runs"))
    parser.add_argument("--run-id", default="e1_caltech_smoke_v1")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--notification-topic-arn", default=os.environ.get("BUNN_SNS_TOPIC_ARN"))
    parser.add_argument("--require-notification", action="store_true")
    return parser.parse_args()


def update_status(run_dir: Path, **values: object) -> None:
    path = run_dir / "status.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    current.update(values)
    current["updated_utc"] = utc_now()
    write_json_atomic(path, current)


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def diagnostic_array(output, subjects: int) -> np.ndarray:
    values = output.diagnostics[-1]
    result = np.empty((subjects, len(METRICS)), dtype=np.float32)
    for metric_index, metric in enumerate(METRICS):
        result[:, metric_index] = values[metric].detach().cpu().numpy().astype(np.float32)
    return result


def run(args: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("E1 requires CUDA for the full 100-permutation smoke")
    if args.permutations != 100:
        raise ValueError("The frozen promotion smoke requires exactly 100 permutations")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    manifest = json.loads((args.inputs / "input_manifest.json").read_text(encoding="utf-8"))
    if manifest["archive_sha256"] != contract["input_hashes"]["sealed_archive"]:
        raise ValueError("Prepared checkpoint manifest does not match the analysis config")
    if sha256_file(args.connectomes) != contract["input_hashes"]["connectomes"]:
        raise ValueError("Connectome hash does not match the analysis config")
    if sha256_file(args.table) != contract["input_hashes"]["baseline_table"]:
        raise ValueError("Baseline-table hash does not match the analysis config")

    run_dir = args.output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "metadata.json"
    immutable = {
        "run_id": args.run_id, "site": args.site, "permutations": args.permutations,
        "preflight_only": bool(args.preflight_only),
        "contract_sha256": sha256_file(args.contract),
        "input_manifest_sha256": sha256_file(args.inputs / "input_manifest.json"),
        "connectomes_sha256": sha256_file(args.connectomes), "table_sha256": sha256_file(args.table),
        "interventions": list(INTERVENTIONS), "metrics": list(METRICS),
        "results_embargoed": True, "score_blind_terminal": True,
    }
    if metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8"))["immutable"] != immutable:
            raise ValueError("Resume metadata differs from the immutable run contract")
        if not args.resume:
            raise FileExistsError("Run already exists; use --resume after audit")
    else:
        write_json_atomic(metadata_path, {"created_utc": utc_now(), "immutable": immutable})

    if args.require_notification:
        alert = publish_sns_notification(
            run_dir, args.notification_topic_arn, f"BuNN E1 STARTED: {args.run_id}",
            f"Score-blind E1 smoke {args.run_id} started for {args.site}. Results remain embargoed pending audit.",
        )
        if alert.get("status") != "published":
            raise RuntimeError("Required SNS start notification failed")

    cohort = load_neural_cohort(args.connectomes, args.table)
    test_indices = np.flatnonzero(cohort.site_ids == args.site)
    if not len(test_indices):
        raise ValueError(f"Unknown held-out site: {args.site}")
    subject_ids = cohort.subject_ids[test_indices].astype(str)
    labels = cohort.labels[test_indices].astype(np.int8)
    raw = torch.from_numpy(cohort.connectomes[test_indices]).to(torch.device("cuda"))
    predictions = pd.read_csv(args.inputs / "source" / "predictions.csv", dtype={"subject_id": str})
    predictions = predictions[(predictions.operator == "learned_bunn") & (predictions.held_out_site == args.site)]
    checkpoint_rows = [row for row in manifest["checkpoints"] if row["site"] == args.site]
    fold_values = {int(row["fold"]) for row in checkpoint_rows}
    if len(fold_values) != 1 or len(checkpoint_rows) != 20:
        raise ValueError("Smoke site must have exactly 20 accepted checkpoints in one fold")

    device = torch.device("cuda")
    base_seed = int(contract["randomization"]["base_seed"])
    tolerance = float(contract["numerical_gates"]["maximum_absolute_probability_reproduction_error"])
    completed_density_files: list[str] = []
    started = time.monotonic()
    try:
        for density_index, density in enumerate(contract["cohort"]["densities"]):
            density = float(density)
            output_path = run_dir / f"density_{density:.2f}.npz"
            if args.resume and output_path.exists():
                completed_density_files.append(output_path.name)
                continue
            update_status(
                run_dir, state="running", current_density=density, current_intervention="reproduction_gate",
                current_permutation=0, completed_density_files=completed_density_files,
            )
            graph = make_dense_graph_batch(raw, density)
            adjacency = graph.adjacency
            base_kernel = graph_heat_kernel(adjacency, 1.0)
            seeds = [int(value) for value in contract["cohort"]["final_seeds"]]
            subjects = len(test_indices)
            probabilities = np.full(
                (len(INTERVENTIONS), len(seeds), args.permutations, subjects), np.nan, dtype=np.float32
            )
            diagnostics = np.full(
                (len(INTERVENTIONS), len(seeds), args.permutations, subjects, len(METRICS)),
                np.nan, dtype=np.float32,
            )
            orthogonality = np.full((len(INTERVENTIONS), len(seeds), args.permutations), np.nan, dtype=np.float32)
            reproduction_error = np.empty(len(seeds), dtype=np.float64)
            equivariance_logit_error = np.full(len(seeds), np.nan, dtype=np.float64)
            equivariance_diagnostic_error = np.full(len(seeds), np.nan, dtype=np.float64)
            topology_degree_match = np.ones(args.permutations, dtype=np.bool_)
            models: list[SharedGraphClassifier] = []
            feature_batches: list[torch.Tensor] = []
            reference_logits: list[torch.Tensor] = []
            reference_diagnostics: list[np.ndarray] = []

            for seed_index, seed in enumerate(seeds):
                row = next(row for row in checkpoint_rows if float(row["density"]) == density and int(row["seed"]) == seed)
                checkpoint_path = args.inputs / row["path"]
                if sha256_file(checkpoint_path) != row["sha256"]:
                    raise ValueError(f"Checkpoint hash mismatch: {checkpoint_path}")
                payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
                if payload.get("state") != "complete":
                    raise ValueError("An accepted checkpoint is not complete")
                model = SharedGraphClassifier("learned_bunn").to(device)
                model.load_state_dict(payload["model_state"])
                model.eval()
                mean = payload["scaler_mean"].to(device)
                scale = payload["scaler_scale"].to(device)
                features = (raw - mean) / scale
                output = forward_intervened(
                    model, features, adjacency, "unaltered", heat_kernel=base_kernel,
                )
                observed = torch.sigmoid(output.logits).cpu().numpy()
                expected_rows = predictions[(predictions.density == density) & (predictions.seed == seed)]
                expected_by_subject = dict(zip(expected_rows.subject_id, expected_rows.probability_asd, strict=True))
                expected = np.asarray([expected_by_subject[value] for value in subject_ids], dtype=np.float64)
                reproduction_error[seed_index] = float(np.max(np.abs(observed - expected)))
                if reproduction_error[seed_index] > tolerance:
                    raise RuntimeError("Archived probability reproduction gate failed")
                probabilities[0, seed_index, 0] = observed
                diagnostics[0, seed_index, 0] = diagnostic_array(output, subjects)
                orthogonality[0, seed_index, 0] = output.maximum_orthogonality_error
                models.append(model)
                feature_batches.append(features)
                reference_logits.append(output.logits.detach().clone())
                reference_diagnostics.append(diagnostics[0, seed_index, 0].copy())

            if args.preflight_only:
                completed_density_files.append(f"reproduced_density_{density:.2f}")
                update_status(
                    run_dir, state="running", completed_density_files=completed_density_files,
                    current_intervention="reproduction_gate_passed", current_permutation=None,
                )
                continue

            for intervention_index, intervention in enumerate(INTERVENTIONS):
                if intervention in {"unaltered", *RANDOM_INTERVENTIONS}:
                    continue
                update_status(run_dir, current_intervention=intervention, current_permutation=0)
                for seed_index, (model, features) in enumerate(zip(models, feature_batches, strict=True)):
                    kernel = base_kernel
                    if intervention == "encoded_node_permutation_equivariance":
                        permutation = torch.as_tensor(
                            deterministic_derangement(raw.shape[1], base_seed), device=device,
                        )
                        kernel = base_kernel[:, permutation][:, :, permutation]
                    output = forward_intervened(
                        model, features, adjacency, intervention,
                        randomization_seed=base_seed, heat_kernel=kernel,
                    )
                    probabilities[intervention_index, seed_index, 0] = torch.sigmoid(output.logits).cpu().numpy()
                    diagnostics[intervention_index, seed_index, 0] = diagnostic_array(output, subjects)
                    orthogonality[intervention_index, seed_index, 0] = output.maximum_orthogonality_error
                    if intervention == "encoded_node_permutation_equivariance":
                        equivariance_logit_error[seed_index] = float(
                            (output.logits - reference_logits[seed_index]).abs().max().cpu()
                        )
                        equivariance_diagnostic_error[seed_index] = float(
                            np.max(np.abs(
                                diagnostics[intervention_index, seed_index, 0]
                                - reference_diagnostics[seed_index]
                            ))
                        )

            for permutation_index in range(args.permutations):
                randomization_seed = base_seed + permutation_index
                for intervention in ("node_map_shuffle", "random_orthogonal_maps"):
                    intervention_index = INTERVENTIONS.index(intervention)
                    update_status(
                        run_dir, current_intervention=intervention,
                        current_permutation=permutation_index + 1, total_permutations=args.permutations,
                    )
                    for seed_index, (model, features) in enumerate(zip(models, feature_batches, strict=True)):
                        output = forward_intervened(
                            model, features, adjacency, intervention,
                            randomization_seed=randomization_seed, heat_kernel=base_kernel,
                        )
                        probabilities[intervention_index, seed_index, permutation_index] = torch.sigmoid(output.logits).cpu().numpy()
                        diagnostics[intervention_index, seed_index, permutation_index] = diagnostic_array(output, subjects)
                        orthogonality[intervention_index, seed_index, permutation_index] = output.maximum_orthogonality_error

                intervention = "degree_preserving_topology"
                intervention_index = INTERVENTIONS.index(intervention)
                update_status(
                    run_dir, current_intervention=intervention,
                    current_permutation=permutation_index + 1, total_permutations=args.permutations,
                )
                rewired = degree_preserving_rewire(adjacency, seed=randomization_seed)
                topology_degree_match[permutation_index] = bool(
                    torch.equal((rewired > 0).sum(-1), (adjacency > 0).sum(-1))
                )
                rewired_kernel = graph_heat_kernel(rewired, 1.0)
                for seed_index, (model, features) in enumerate(zip(models, feature_batches, strict=True)):
                    output = forward_intervened(
                        model, features, adjacency, intervention,
                        randomization_seed=randomization_seed,
                        adjacency_override=rewired, heat_kernel=rewired_kernel,
                    )
                    probabilities[intervention_index, seed_index, permutation_index] = torch.sigmoid(output.logits).cpu().numpy()
                    diagnostics[intervention_index, seed_index, permutation_index] = diagnostic_array(output, subjects)
                    orthogonality[intervention_index, seed_index, permutation_index] = output.maximum_orthogonality_error

            save_npz_atomic(
                output_path, probabilities=probabilities, diagnostics=diagnostics,
                orthogonality_error=orthogonality, reproduction_error=reproduction_error,
                equivariance_logit_error=equivariance_logit_error,
                equivariance_diagnostic_error=equivariance_diagnostic_error,
                topology_degree_match=topology_degree_match,
                subject_ids=subject_ids, labels=labels, interventions=np.asarray(INTERVENTIONS),
                metrics=np.asarray(METRICS), seeds=np.asarray(seeds), density=np.asarray([density]),
            )
            completed_density_files.append(output_path.name)
            update_status(
                run_dir, state="running", completed_density_files=completed_density_files,
                current_intervention="density_sealed", current_permutation=None,
            )

        if args.preflight_only:
            preflight = {
                "state": "passed", "completed_utc": utc_now(), "site": args.site,
                "checkpoint_count": len(checkpoint_rows), "density_count": len(completed_density_files),
                "probability_tolerance": tolerance, "scientific_values_displayed": False,
            }
            write_json_atomic(run_dir / "reproduction_preflight.json", preflight)
            update_status(run_dir, state="reproduction_preflight_passed", current_density=None, current_intervention=None)
            print(json.dumps({"state": "reproduction_preflight_passed", "scientific_values_displayed": False}))
            return run_dir

        artifacts = {name: sha256_file(run_dir / name) for name in completed_density_files}
        complete = {
            "state": "complete_pending_score_blind_audit", "completed_utc": utc_now(),
            "runtime_seconds": time.monotonic() - started, "artifacts": artifacts,
            "density_count": len(completed_density_files), "results_embargoed": True,
        }
        write_json_atomic(run_dir / "complete.json", complete)
        update_status(run_dir, state="complete_pending_score_blind_audit", current_density=None, current_intervention=None)
        if args.require_notification:
            publish_sns_notification(
                run_dir, args.notification_topic_arn, f"BuNN E1 COMPLETE: {args.run_id}",
                f"Score-blind E1 smoke {args.run_id} finished. Run the artifact auditor before reading results.",
            )
        print(json.dumps({"state": complete["state"], "artifacts": len(artifacts), "results_displayed": False}))
        return run_dir
    except Exception as error:
        update_status(run_dir, state="failed", error_type=type(error).__name__, error=str(error))
        if args.require_notification:
            publish_sns_notification(
                run_dir, args.notification_topic_arn, f"BuNN E1 FAILED: {args.run_id}",
                f"E1 smoke {args.run_id} failed. Inspect status.json; no result values were displayed.",
            )
        raise


if __name__ == "__main__":
    run(parse_args())
