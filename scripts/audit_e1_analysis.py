"""Recompute and audit the complete E1 analysis before interpretation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from PIL import Image

from scripts.run_baselines import write_json_atomic
from src.extensions.e1_analysis import load_and_compute, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("configs/extensions/e1_analysis_v1.json"))
    parser.add_argument(
        "--source-root", type=Path,
        default=Path("outputs/extensions/e1_interventions_v1/accepted"),
    )
    parser.add_argument(
        "--analysis-dir", type=Path,
        default=Path("outputs/analysis/e1_interventions_v1"),
    )
    return parser.parse_args()


def audit(args: argparse.Namespace) -> Path:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    metadata_path = args.analysis_dir / "analysis_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("state") != "complete_pending_independent_audit":
        raise ValueError("E1 analysis is not complete and awaiting audit")
    if metadata.get("protocol_sha256") != sha256_file(args.protocol):
        raise ValueError("Analysis metadata references a different protocol")
    for name, expected in metadata["artifact_hashes"].items():
        path = args.analysis_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Missing or hash-invalid analysis artifact: {name}")

    recomputed = load_and_compute(args.source_root, protocol)
    for name, expected_table in recomputed.items():
        observed = pd.read_csv(args.analysis_dir / f"{name}.csv")
        pd.testing.assert_frame_equal(
            observed, expected_table, check_dtype=False, check_exact=False,
            rtol=1e-10, atol=1e-10,
        )

    required_tables = set(protocol["fixed_outputs"]["tables"])
    required_figures = set(protocol["fixed_outputs"]["figures"])
    if required_tables - set(metadata["artifact_hashes"]) or required_figures - set(metadata["artifact_hashes"]):
        raise ValueError("Analysis metadata lacks a fixed table or figure")
    for name in required_figures:
        with Image.open(args.analysis_dir / name) as image:
            image.verify()
        with Image.open(args.analysis_dir / name) as image:
            if image.width < 1000 or image.height < 600:
                raise ValueError(f"Figure resolution is too small: {name}")

    certificate = {
        "state": "passed", "audited_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": sha256_file(args.protocol),
        "analysis_metadata_sha256": sha256_file(metadata_path),
        "recomputed_tables": sorted(f"{name}.csv" for name in recomputed),
        "verified_figures": sorted(required_figures),
        "artifact_hashes": metadata["artifact_hashes"],
        "scientific_values_printed_to_terminal": False,
        "approved_for_interpretation": True,
    }
    path = args.analysis_dir / "analysis_audit.json"
    write_json_atomic(path, certificate)
    print(json.dumps({
        "state": "passed", "approved_for_interpretation": True,
        "scientific_values_displayed": False,
    }))
    return path


if __name__ == "__main__":
    audit(parse_args())
