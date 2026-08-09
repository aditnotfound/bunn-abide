# Step 13 Paper and Reproducibility Plan

Status: **Steps 13.0--13.1 complete.** All 28 frozen inputs validated, 12
contract outputs were generated, repeated builds were byte-identical, all four
figures passed visual inspection, and the complete local suite passed 67 tests.

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

## Next paper-writing gate

Only after these checks pass should Step 13.2 create the LaTeX manuscript and
use the generated tables, figures, and result snapshot. Numbers must not be
typed independently into the manuscript when a generated macro or table can
provide them.
