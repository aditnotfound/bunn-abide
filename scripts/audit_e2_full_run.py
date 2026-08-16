"""Seal every E2 replicate after independently re-auditing all 700 cells score-blind."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_e2_synthetic import FULL_CONDITIONS, OPERATORS
from src.extensions.e2_analysis import sha256_file
from src.extensions.e2_training import audit_e2_cell, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/extensions/e2_synthetic_geometry_v1.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol_hash = sha256_file(args.protocol)
    manager_path = args.run_dir / "manager_complete.json"
    manager = json.loads(manager_path.read_text(encoding="utf-8"))
    if manager.get("state") != "complete_all_cells_score_blind_audited" or manager.get("mode") != "full":
        raise RuntimeError("E2 full manager is not complete")
    if manager.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("E2 full manager protocol mismatch")
    seeds = [int(seed) for seed in protocol["data"]["replicate_seeds"]]
    expected_cells = len(FULL_CONDITIONS) * len(OPERATORS) * len(seeds)
    if manager.get("expected_cells") != expected_cells or len(manager.get("audit_hashes", {})) != expected_cells:
        raise RuntimeError("E2 manager does not contain the frozen 700-cell matrix")
    seal_dir = args.run_dir / "replicate_seals"
    seal_hashes: dict[str, str] = {}
    audited_cells = 0
    for family, noise in FULL_CONDITIONS:
        for seed in seeds:
            replicate_id = f"{family}__noise_{noise:g}__seed_{seed}"
            cell_hashes: dict[str, str] = {}
            for operator in OPERATORS:
                cell_id = f"{replicate_id}__{operator}"
                cell_dir = args.run_dir / "cells" / cell_id
                audit_path = cell_dir / "score_blind_audit.json"
                if manager["audit_hashes"].get(cell_id) != sha256_file(audit_path):
                    raise RuntimeError(f"Manager-to-cell audit mismatch: {cell_id}")
                audit = audit_e2_cell(cell_dir, require_predictions=True)
                if audit["manifest_sha256"] != json.loads(audit_path.read_text(encoding="utf-8"))["manifest_sha256"]:
                    raise RuntimeError(f"Independent cell audit mismatch: {cell_id}")
                manifest = json.loads((cell_dir / "cell_manifest.json").read_text(encoding="utf-8"))
                immutable = manifest["immutable"]
                expected = {
                    "family": family, "transport_noise_degrees": float(noise),
                    "data_seed": seed, "operator": operator, "evaluate_test": True,
                    "protocol_sha256": protocol_hash,
                }
                for key, value in expected.items():
                    if immutable.get(key) != value:
                        raise RuntimeError(f"Frozen cell identity mismatch: {cell_id} field {key}")
                cell_hashes[operator] = sha256_file(audit_path)
                audited_cells += 1
            seal = {
                "state": "replicate_sealed_score_blind",
                "replicate_id": replicate_id,
                "family": family,
                "transport_noise_degrees": float(noise),
                "replicate_seed": seed,
                "operator_audit_hashes": cell_hashes,
            }
            seal_path = seal_dir / f"{replicate_id}.json"
            write_json_atomic(seal_path, seal)
            seal_hashes[replicate_id] = sha256_file(seal_path)
    full_audit = {
        "state": "complete_full_run_reaudit_passed_score_blind",
        "protocol_sha256": protocol_hash,
        "manager_completion_sha256": sha256_file(manager_path),
        "audited_cells": audited_cells,
        "sealed_replicates": len(seal_hashes),
        "replicate_seal_hashes": seal_hashes,
        "scientific_values_opened": False,
    }
    write_json_atomic(args.run_dir / "score_blind_full_audit.json", full_audit)
    print(json.dumps({
        "state": full_audit["state"], "cells": audited_cells,
        "replicates": len(seal_hashes), "scientific_values_opened": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
