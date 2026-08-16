from pathlib import Path

from scripts.audit_yhsa_submission import (
    PROJECT_ROOT,
    audit_report,
    extract_bib_keys,
    extract_citations,
)


def test_citation_extractors_handle_multiple_keys() -> None:
    assert extract_citations(r"Text \cite{alpha,beta}.") == {"alpha", "beta"}
    assert extract_bib_keys("@article{alpha,\n}@misc{beta,\n}") == {"alpha", "beta"}


def test_current_submission_source_passes_fail_closed_audit() -> None:
    result = audit_report(
        PROJECT_ROOT / "paper/yhsa-submission/report.tex",
        PROJECT_ROOT / "paper/references.bib",
        PROJECT_ROOT / "reproducibility/final_submission_manifest.json",
    )
    assert result["passed"], result


def test_undefined_citation_is_detected(tmp_path: Path) -> None:
    report = (PROJECT_ROOT / "paper/yhsa-submission/report.tex").read_text(encoding="utf-8")
    report_path = tmp_path / "report.tex"
    report_path.write_text(report + r"\cite{definitely_missing_key}", encoding="utf-8")
    result = audit_report(
        report_path,
        PROJECT_ROOT / "paper/references.bib",
        PROJECT_ROOT / "reproducibility/final_submission_manifest.json",
    )
    assert not result["passed"]
    assert result["details"]["missing_citations"] == ["definitely_missing_key"]
