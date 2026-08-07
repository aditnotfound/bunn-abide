from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from scripts.audit_neural_full import FullAuditError, validate_parallel_provenance
from scripts.compare_neural_execution_equivalence import EquivalenceError, compare_value
from scripts.run_baselines import sha256_file, write_csv, write_json_atomic
from scripts.run_neural_full import SITE_ARTIFACT_FIELDS, label_for_site
from scripts.run_neural_full_parallel import merge_worker_runs, partition_sites


def test_partition_sites_is_complete_unique_and_deterministic() -> None:
    sites = [f"SITE_{index}" for index in range(8)]
    assert partition_sites(sites, 3) == [
        ["SITE_0", "SITE_3", "SITE_6"],
        ["SITE_1", "SITE_4", "SITE_7"],
        ["SITE_2", "SITE_5"],
    ]
    assert sorted(site for shard in partition_sites(sites, 3) for site in shard) == sorted(sites)


@pytest.mark.parametrize("worker_count", [0, 4])
def test_partition_sites_rejects_invalid_worker_count(worker_count: int) -> None:
    with pytest.raises(ValueError):
        partition_sites(["A", "B", "C"], worker_count)


def test_score_blind_numeric_equivalence_uses_strict_tolerance() -> None:
    assert compare_value("0.5", "0.50000001", 1e-7) == pytest.approx(1e-8)
    with pytest.raises(EquivalenceError, match="exceeds tolerance"):
        compare_value("0.5", "0.500001", 1e-7)


def parent_metadata(sites: list[str], assignments: list[list[str]]) -> dict:
    shared = {
        "run_kind": "smoke", "code_version": "test", "source_hashes": {"x": "y"},
        "frozen_input_hashes": {"a": "b"}, "configurations": [{"operator": "identity", "density": 0.0}],
        "protocol": {"protocol_version": 1}, "operator_contract": {"contract_version": 2},
        "analysis_protocol": {"protocol_version": 1}, "smoke_override": {"candidate_count": 1},
    }
    return {
        "run_id": "parallel-test", "status": "running", "held_out_sites": sites,
        "site_to_outer_fold": {str(index): site for index, site in enumerate(sites)},
        "execution_mode": "site_parallel", "results_embargoed": True, **shared,
        "parallel_execution": {
            "contract_version": 1, "worker_count": len(assignments),
            "assignments": {
                f"worker_{index:02d}": shard for index, shard in enumerate(assignments)
            },
        },
    }


def create_worker(run_dir: Path, metadata: dict, worker_index: int, sites: list[str]) -> None:
    worker_id = f"worker_{worker_index:02d}"
    worker_dir = run_dir / "workers" / worker_id
    (worker_dir / "folds").mkdir(parents=True)
    for site in sites:
        outer_fold = int(next(fold for fold, value in metadata["site_to_outer_fold"].items() if value == site))
        site_dir = worker_dir / "folds" / label_for_site(outer_fold, site)
        site_dir.mkdir()
        for name, fields in SITE_ARTIFACT_FIELDS.items():
            write_csv(site_dir / name, [], fields)
        write_json_atomic(
            site_dir / "complete.json",
            {
                "state": "complete", "outer_fold": outer_fold, "held_out_site": site,
                "row_counts": {name: 0 for name in SITE_ARTIFACT_FIELDS},
                "artifact_hashes": {
                    name: sha256_file(site_dir / name) for name in SITE_ARTIFACT_FIELDS
                },
            },
        )
    for name, fields in SITE_ARTIFACT_FIELDS.items():
        write_csv(worker_dir / name, [], fields)
    write_json_atomic(worker_dir / "summary.json", {"status": "complete"})
    worker_metadata = {
        field: metadata[field]
        for field in (
            "run_kind", "code_version", "source_hashes", "frozen_input_hashes",
            "configurations", "protocol", "operator_contract", "analysis_protocol", "smoke_override",
        )
    }
    worker_metadata.update(
        {
            "run_id": worker_id, "status": "complete", "execution_shard": True,
            "held_out_sites": sites,
            "artifact_hashes": {
                name: sha256_file(worker_dir / name)
                for name in [*SITE_ARTIFACT_FIELDS, "summary.json"]
            },
        }
    )
    write_json_atomic(worker_dir / "metadata.json", worker_metadata)


def test_merge_and_parallel_provenance_detect_copy_corruption() -> None:
    sites = ["A", "B", "C"]
    assignments = [["A"], ["B"], ["C"]]
    metadata = parent_metadata(sites, assignments)
    with TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / metadata["run_id"]
        run_dir.mkdir()
        write_json_atomic(run_dir / "metadata.json", metadata)
        for index, shard in enumerate(assignments):
            create_worker(run_dir, metadata, index, shard)
        merge_worker_runs(run_dir, metadata, assignments)
        final_metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        evidence = validate_parallel_provenance(run_dir, final_metadata, sites)
        assert evidence == {"worker_count": 3, "site_count": 3}

        corrupted = run_dir / "folds" / label_for_site(0, "A") / "predictions.csv"
        corrupted.write_text(corrupted.read_text(encoding="utf-8") + "corrupt\n", encoding="utf-8")
        with pytest.raises(FullAuditError, match="Parallel site copy mismatch"):
            validate_parallel_provenance(run_dir, final_metadata, sites)
