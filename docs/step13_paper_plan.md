# Step 13 Paper and Reproducibility Plan

Status: **Steps 13.0--13.2 complete.** The evidence package, publication
assets, manuscript inputs, main paper, and supplement are now built and
verified. The current complete local unittest suite passes 62 tests.

## Scope

Step 13 converts the completed, audited analyses into a traceable manuscript
and reproducibility package. It does not refit a model, change an estimand,
select a favorable result, or alter any Step 7--12 artifact. The evidence
snapshot is the repository state at commit
`a4def2a1f14f1bbff71356d0689eeeee4f405f4a`.

## Step 13.0 - freeze the evidence boundary

`configs/paper_assets_v1.json` lists every accepted source artifact and its
SHA-256 digest. The paper-assets builder must fail before writing outputs if a
source is missing or its digest differs. It produces:

- `reproducibility/artifact_manifest.json`, the machine-readable evidence
  inventory;
- `reproducibility/result_snapshot.json`, the exact values permitted in the
  manuscript; and
- `docs/step13_claim_ledger.md`, the human-readable boundary between supported
  claims, conditional interpretations, and prohibited claims.

The result snapshot is deliberately deterministic: it has no generation time,
machine path, or other volatile field.

## Step 13.1 - stage paper tables and figures

`scripts/build_paper_assets.py` reads only the hash-bound analysis outputs. It
replots the accepted site-level values with publication labels and layout, and
creates compact paper tables from the accepted CSV rows. Replotting does not
change or add an inferential test. The builder writes a manifest containing the
SHA-256 digest of every generated asset. Re-running it must produce the same
bytes for all outputs.

The staged main assets are:

1. predictive balanced-accuracy curves across density;
2. common-frame representation diagnostics across density;
3. leave-one-site influence;
4. seed-specific stability;
5. classical baseline summary;
6. confirmatory predictive contrasts;
7. representation contrasts; and
8. parameter/runtime efficiency.

The manuscript may move some robustness assets to the supplement, but it may
not silently omit a result because its direction is inconvenient.

## Claim hierarchy

1. The primary result is the equal-site BuNN-minus-GCN normalized density-curve
   contrast frozen in Step 11.
2. The elastic-net comparison and matched-anchor effective-rank contrast are
   pre-specified supporting tests.
3. Site influence, seed sensitivity, alternative weighting, exhaustive
   contrasts, and efficiency are secondary Step 12 analyses.
4. Individual densities and the best observed neural configuration are
   descriptive unless explicitly identified as multiplicity-controlled.

## Quality gates

Step 13.0--13.1 is complete only when:

- all frozen input hashes validate;
- every output in the contract exists and has a recorded digest;
- a second clean generation is byte-identical;
- the automated tests verify the primary values and claim boundaries;
- all four figures pass visual inspection; and
- the execution, decision, and AI-use logs record the work accurately.

All six gates passed on 2026-08-09.

## Step 13.2 - manuscript and supplement

`configs/manuscript_v1.json` binds the claim ledger, result snapshot, evidence
manifest, and paper-asset manifest by SHA-256. The manuscript-input builder
validates those inputs before generating the study-design figure, numeric
LaTeX macros, main-paper tables, inline supplement tables, and a generated-input
manifest. A bounded Windows retry preserves atomic output replacement if
OneDrive briefly locks a generated file.

The manuscript uses the original project's ABIDE connectomes, BuNN operator,
aggregation question, and cross-site objective while explicitly narrowing the
unidentifiable signed-edge interpretation to a controlled computational audit.
Secondary site and seed plots are placed in the supplement so they remain
visible without interrupting the main Results-to-Discussion flow.

Completion checks on 2026-08-09:

- all four bound manuscript inputs and all 11 staged paper assets validated;
- 11 generated manuscript inputs were byte-deterministic across rebuilds;
- all citation keys and claim IDs resolved;
- headline values were absent from narrative source files and supplied through
  generated macros or tables;
- the complete repository suite passed 62 tests;
- the 8-page main PDF and 3-page supplement compiled with BibTeX where needed;
- all 11 final PDF pages were rendered and visually inspected; and
- PDF text and LaTeX logs contained no unresolved citation/reference or fatal
  error markers.

No model was refit and no new inferential analysis was performed in Step 13.2.
The PDFs are a reproducible manuscript scaffold, not a venue-specific final
submission; authorship, target style, and any external-release details still
require the researcher's review.

## Step 13.3 - visual and prose revision

The main-paper prose, supplement, captions, title, and diagram labels were
revised with the project-local Humanizer guidance. The revision removed
template-like phrasing and repetitive disclaimers while preserving every
numeric macro, citation, claim ID, limitation, and result interpretation.

The pastel rounded-box workflow was replaced with a restrained four-part
line-art overview. A second schematic now distinguishes direct GCN averaging,
the learned-local capacity control, and learned bundle transport. Both figures
are generated deterministically by the manuscript builder. No brain
illustration, attention map, or anatomical network view was added because the
analysis does not support a biological localization claim.

The generated-input contract now contains 11 deterministic inputs plus its
manifest. Main-paper figures and tables are anchored to the sections that
introduce them. The final PDFs remain 8 main pages and 3 supplement pages, and
all 11 pages were rendered and inspected after the revision. This step changed
presentation and wording only; it did not alter data, fitting, estimands,
statistics, or conclusions.
