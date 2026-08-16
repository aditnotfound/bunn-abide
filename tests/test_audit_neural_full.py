from __future__ import annotations

import pytest

from scripts.audit_neural_full import FullAuditError, validate_configuration_grid


def contract() -> dict:
    return {
        "operators": {
            "identity": {"densities": [0]},
            "learned_local": {"densities": [0]},
            "gcn": {"densities": [0.01, 0.05, 0.10, 0.20]},
            "trivial_bundle": {"densities": [0.01, 0.05, 0.10, 0.20]},
            "learned_bunn": {"densities": [0.01, 0.05, 0.10, 0.20]},
        }
    }


def metadata_grid() -> list[dict]:
    return [
        {"operator": operator, "density": density}
        for operator, specification in contract()["operators"].items()
        for density in specification["densities"]
    ]


def test_configuration_audit_accepts_json_key_reordering() -> None:
    reordered_contract = {
        "operators": dict(sorted(contract()["operators"].items()))
    }
    validated = validate_configuration_grid(metadata_grid(), reordered_contract, 14)
    assert validated[0] == ("identity", 0.0)
    assert len(validated) == 14


@pytest.mark.parametrize("summary_count", [13, 15, None])
def test_configuration_audit_rejects_wrong_summary_count(summary_count: object) -> None:
    with pytest.raises(FullAuditError, match="Configuration grid mismatch"):
        validate_configuration_grid(metadata_grid(), contract(), summary_count)


def test_configuration_audit_rejects_duplicate_or_missing_cell() -> None:
    grid = metadata_grid()
    grid[-1] = grid[-2]
    with pytest.raises(FullAuditError, match="Configuration grid mismatch"):
        validate_configuration_grid(grid, contract(), 14)
