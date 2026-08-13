"""Independently recompute an E2 analysis package and compare every fixed artifact."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_e2_synthetic import execute
from src.extensions.e2_analysis import sha256_file
from src.extensions.e2_training import write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/extensions/e2_synthetic_geometry_v1.json")
    parser.add_argument("--analysis", type=Path, default=ROOT / "configs/extensions/e2_analysis_v1.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(args.analysis.read_text(encoding="utf-8"))["fixed_outputs"]
    with tempfile.TemporaryDirectory() as temporary:
        recomputed = Path(temporary)
        execute(args.protocol, args.analysis, args.run_dir, recomputed)
        mismatches = []
        for name in expected:
            if sha256_file(recomputed / name) != sha256_file(args.analysis_dir / name):
                mismatches.append(name)
    if mismatches:
        raise RuntimeError(f"E2 independent recomputation mismatch: {mismatches}")
    audit = {
        "state": "independent_recomputation_passed",
        "artifact_count": len(expected),
        "analysis_metadata_sha256": sha256_file(args.analysis_dir / "analysis_metadata.json"),
    }
    write_json_atomic(args.analysis_dir / "independent_audit.json", audit)
    print(json.dumps({"state": audit["state"], "scientific_values_printed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
