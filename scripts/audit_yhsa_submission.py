"""Consistency audit for the YHSA-format report."""

import argparse
import hashlib
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_citations(tex: str) -> set[str]:
    keys: set[str] = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", tex):
        keys.update(key.strip() for key in group.split(",") if key.strip())
    return keys


def extract_bib_keys(bib: str) -> set[str]:
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib, flags=re.IGNORECASE))


def audit_report(
    report_path: Path,
    bibliography_path: Path,
    manifest_path: Path,
) -> dict:
    report = report_path.read_text(encoding="utf-8")
    bibliography = bibliography_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    numbers_path = PROJECT_ROOT / "paper/generated/numbers.tex"
    numbers = numbers_path.read_text(encoding="utf-8")
    normalized_report = re.sub(r"\s+", " ", report)
    normalized_evidence_source = re.sub(r"\s+", " ", report + "\n" + numbers)

    citations = extract_citations(report)
    bibliography_keys = extract_bib_keys(bibliography)
    missing_citations = sorted(citations - bibliography_keys)

    required_claim_tokens = {
        "cohort": "754 technically eligible",
        "sites": "18 sites",
        "primary_contrast": "-0.00958",
        "elastic_net_contrast": "-0.05516",
        "rbf_result": "0.6255",
        "e1_random_map_effect": "8.86 percentage points",
        "e2_conditional_effect": "42.75 points",
        "diffusion_operator": r"H=\exp(-tL)",
        "diffusion_time": r"t=1",
        "abide_ii_boundary": "does not report external validation",
        "biological_boundary": "not biological bundle geometry",
    }
    missing_claim_tokens = {}
    for name, token in required_claim_tokens.items():
        source = (
            normalized_evidence_source
            if name in {"primary_contrast", "elastic_net_contrast"}
            else normalized_report
        )
        if token not in source:
            missing_claim_tokens[name] = token

    forbidden_tokens = ["??", "TODO", "TBD", "PLACEHOLDER"]
    found_forbidden = [token for token in forbidden_tokens if token in report]

    required_files = [
        PROJECT_ROOT / "reproducibility/result_snapshot.json",
        PROJECT_ROOT / "reproducibility/final_revision_result.json",
        PROJECT_ROOT / "reproducibility/nonlinear_baseline_result.json",
        PROJECT_ROOT / "paper/yhsa-submission/generated/tables/qc_cascade.tex",
        PROJECT_ROOT / "paper/yhsa-submission/generated/tables/model_weighting_intervals.tex",
        PROJECT_ROOT / "paper/yhsa-submission/generated/tables/weighting_contrasts.tex",
        PROJECT_ROOT / "paper/yhsa-submission/generated/tables/e1_pairwise.tex",
        PROJECT_ROOT / "paper/yhsa-submission/generated/tables/heat_spectrum.tex",
    ]
    missing_files = [
        str(path.relative_to(PROJECT_ROOT)) for path in required_files if not path.is_file()
    ]

    expected_report_hash = manifest["artifacts"]["paper/yhsa-submission/report.tex"]
    report_hash = sha256(report_path)
    manifest_report_hash_matches = report_hash == expected_report_hash

    checks = {
        "all_citations_defined": not missing_citations,
        "required_claim_tokens_present": not missing_claim_tokens,
        "no_unresolved_placeholders": not found_forbidden,
        "required_evidence_files_present": not missing_files,
        "report_source_matches_manifest": manifest_report_hash_matches,
        "accepted_results_marked_unchanged": manifest.get("accepted_results_changed") is False,
        "no_model_retraining_marked": manifest.get("model_retraining_performed") is False,
    }
    return {
        "audit_version": "yhsa_submission_audit_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "details": {
            "citation_count": len(citations),
            "missing_citations": missing_citations,
            "missing_claim_tokens": missing_claim_tokens,
            "found_forbidden_tokens": found_forbidden,
            "missing_files": missing_files,
            "report_sha256": report_hash,
            "manifest_report_sha256": expected_report_hash,
        },
        "boundary": "This audit checks citations, frozen numbers, and evidence files only.",
    }


def write_json_atomic(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "paper/yhsa-submission/report.tex",
    )
    parser.add_argument(
        "--bibliography",
        type=Path,
        default=PROJECT_ROOT / "paper/references.bib",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reproducibility/final_submission_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reproducibility/final_submission_audit.json",
    )
    args = parser.parse_args()
    result = audit_report(args.report, args.bibliography, args.manifest)
    write_json_atomic(result, args.output)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
