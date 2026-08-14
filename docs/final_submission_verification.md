# Final YHSA-format revision verification

Date: 15 August 2026

Branch: `codex/final-yhsa-revision`

Status: technical verification passed; author review remains required

## Remote checkpoint

- GitHub repository: `aditnotfound/bunn-abide`
- The revision branch was pushed and its remote head matched local commit
  `f26614464ceb7677abd131532510762ced3d75a9` before this verification pass.
- GitHub returned the expected 1,700,894-byte report artifact from that branch.
- The previous manuscript and `paper/yhsa-final` were not modified.

## Scientific-source audit

`scripts/audit_yhsa_submission.py` performed a fail-closed consistency check.
It confirmed that:

- all 16 citation keys used by the report exist in `paper/references.bib`;
- the cohort, primary contrast, elastic-net contrast, RBF-SVM result, E1
  random-map effect, E2 conditional effect, diffusion definition, and claim
  boundaries are present in the report or its frozen-number input;
- no unresolved `??`, `TODO`, `TBD`, or `PLACEHOLDER` token appears in the
  report source;
- required frozen result records and generated tables exist;
- the report source matches the hash recorded before this audit;
- the manifest continues to mark accepted results as unchanged, model
  retraining as false, and author review as required.

The machine-readable result is
`reproducibility/final_submission_audit.json`. Three dedicated audit tests and
the complete repository suite passed.

## Rebuild and PDF checks

- Bundled Tectonic rebuilt the report successfully.
- The output has 37 A4 pages and is 1,700,894 bytes.
- The abstract occupies one page.
- Text extraction found no unresolved reference markers or TODO tokens.
- All 37 pages were rendered and inspected through four contact sheets.
- Dense pages containing architecture settings, weighting intervals, E1
  pairwise comparisons, the interpretation table, and references were also
  inspected at full resolution.
- No clipping, overlap, broken glyph, missing figure, or unreadable table was
  found. Remaining TeX warnings are underfull lines in narrow table cells, not
  content loss.
- Final technical suite: **117 passed**.

## Work that only the author can complete

Technical verification does not establish student authorship or satisfy the
AI-disclosure and signature obligations. The student must complete
`paper/yhsa-submission/STUDENT_AUDIT_WORKSHEET.md`, supply the factual fields in
`paper/yhsa-submission/DISCLOSURE_INPUTS.md`, rewrite and own the academic
expression, verify the scientific content personally, and obtain the required
signatures. These items remain open by design and are not inferred by Codex.
