"""Score-blind structural and numerical audit for an E1 smoke artifact."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from scripts.run_baselines import sha256_file, write_json_atomic
from scripts.run_e1_interventions import INTERVENTIONS, METRICS, RANDOM_INTERVENTIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=Path("configs/extensions/e1_checkpoint_interventions_v1.json"))
    return parser.parse_args()


def audit(args: argparse.Namespace) -> Path:
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    metadata = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    complete = json.loads((args.run_dir / "complete.json").read_text(encoding="utf-8"))
    if complete.get("state") != "complete_pending_score_blind_audit":
        raise ValueError("Run is not complete and awaiting audit")
    if metadata["immutable"]["contract_sha256"] != sha256_file(args.contract):
        raise ValueError("Run contract hash differs from the auditor contract")
    permutations = int(metadata["immutable"]["permutations"])
    if permutations != 100:
        raise ValueError("Promotion smoke does not contain 100 permutations")

    gates = contract["numerical_gates"]
    reproduction_limit = float(gates["maximum_absolute_probability_reproduction_error"])
    equivariance_logit_limit = float(gates["maximum_absolute_logit_equivariance_error"])
    equivariance_diagnostic_limit = float(gates["maximum_diagnostic_equivariance_error"])
    orthogonality_limit = float(gates["maximum_orthogonality_error"])
    seeds = len(contract["cohort"]["final_seeds"])
    audited: dict[str, str] = {}
    subjects: int | None = None

    for density in contract["cohort"]["densities"]:
        name = f"density_{float(density):.2f}.npz"
        path = args.run_dir / name
        if not path.is_file() or complete["artifacts"].get(name) != sha256_file(path):
            raise ValueError(f"Missing or hash-invalid density artifact: {name}")
        with np.load(path, allow_pickle=False) as arrays:
            required = {
                "probabilities", "diagnostics", "orthogonality_error", "reproduction_error",
                "equivariance_logit_error", "equivariance_diagnostic_error",
                "topology_degree_match", "subject_ids", "labels", "interventions", "metrics", "seeds", "density",
            }
            if required - set(arrays.files):
                raise ValueError(f"{name} lacks required arrays")
            local_subjects = len(arrays["subject_ids"])
            subjects = local_subjects if subjects is None else subjects
            if local_subjects != subjects or len(set(arrays["subject_ids"].tolist())) != subjects:
                raise ValueError("Subject coverage is inconsistent or duplicated")
            if set(arrays["labels"].tolist()) != {0, 1}:
                raise ValueError("Smoke labels do not contain both classes")
            if arrays["interventions"].tolist() != list(INTERVENTIONS) or arrays["metrics"].tolist() != list(METRICS):
                raise ValueError("Intervention or diagnostic order differs from the frozen runner")
            probability_shape = (len(INTERVENTIONS), seeds, permutations, subjects)
            diagnostic_shape = (*probability_shape, len(METRICS))
            if arrays["probabilities"].shape != probability_shape or arrays["diagnostics"].shape != diagnostic_shape:
                raise ValueError("Prediction or diagnostic array has an invalid shape")
            for intervention_index, intervention in enumerate(INTERVENTIONS):
                expected = np.ones((seeds, permutations), dtype=bool) if intervention in RANDOM_INTERVENTIONS else np.zeros((seeds, permutations), dtype=bool)
                if intervention not in RANDOM_INTERVENTIONS:
                    expected[:, 0] = True
                probability_finite = np.isfinite(arrays["probabilities"][intervention_index]).all(axis=-1)
                diagnostic_finite = np.isfinite(arrays["diagnostics"][intervention_index]).all(axis=(-1, -2))
                orthogonality_finite = np.isfinite(arrays["orthogonality_error"][intervention_index])
                if not np.array_equal(probability_finite, expected):
                    raise ValueError(f"{name} has missing or unexpected probability cells")
                if not np.array_equal(diagnostic_finite, expected) or not np.array_equal(orthogonality_finite, expected):
                    raise ValueError(f"{name} has missing or unexpected diagnostic cells")
            if np.max(arrays["reproduction_error"]) > reproduction_limit:
                raise ValueError("Archived probability reproduction tolerance failed")
            if np.max(arrays["equivariance_logit_error"]) > equivariance_logit_limit:
                raise ValueError("Encoded-node logit equivariance tolerance failed")
            if np.max(arrays["equivariance_diagnostic_error"]) > equivariance_diagnostic_limit:
                raise ValueError("Encoded-node diagnostic equivariance tolerance failed")
            finite_orthogonality = arrays["orthogonality_error"][np.isfinite(arrays["orthogonality_error"])]
            if not len(finite_orthogonality) or np.max(finite_orthogonality) > orthogonality_limit:
                raise ValueError("Orthogonality tolerance failed")
            if not arrays["topology_degree_match"].all():
                raise ValueError("A topology permutation changed a degree sequence")
        audited[name] = sha256_file(path)

    certificate = {
        "state": "passed", "audited_utc": datetime.now(UTC).isoformat(),
        "score_blind": True, "scientific_values_opened": False,
        "site": metadata["immutable"]["site"], "subjects": subjects,
        "density_count": len(audited), "permutations_per_random_family": permutations,
        "artifact_hashes": audited,
        "gates_passed": [
            "archived probability reproduction", "encoded-node logit equivariance",
            "gauge-aware diagnostic equivariance", "O(2) orthogonality",
            "exact degree preservation", "finite complete intervention coverage",
        ],
    }
    path = args.run_dir / "score_blind_audit.json"
    write_json_atomic(path, certificate)
    print(json.dumps({"state": "passed", "score_blind": True, "scientific_values_displayed": False}))
    return path


if __name__ == "__main__":
    audit(parse_args())
