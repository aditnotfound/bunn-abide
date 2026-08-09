# BuNN ABIDE-I Operator Audit

Reproducible evaluation of identity propagation, GCN, and Bundle Neural Networks on ABIDE-I functional connectomes under site-held-out validation.

## Status

Steps 1--12A are complete. The 754-participant, 18-site classical and neural
evaluations passed independent integrity audits before results were opened.
The confirmatory analysis detected no complete transfer of the proposed BuNN
anti-collapse advantage under the frozen pipeline, and the secondary audit
classified the small BuNN--GCN contrast as site- and seed-sensitive. Steps
13.0--13.1 have frozen the paper evidence boundary and generated deterministic
tables, figures, and machine-readable result provenance.

## Layout

- `data/`: local raw and processed data; not committed.
- `docs/`: frozen protocol, decisions, and experiment notes.
- `configs/`: versioned experiment configurations.
- `src/`: reusable implementation modules.
- `scripts/`: data preparation and training entry points.
- `tests/`: unit and leakage-prevention tests.
- `outputs/`: generated runs, figures, and tables; not committed.
- `paper/generated/`: hash-tracked tables and publication-layout figures.
- `reproducibility/`: frozen result snapshot and evidence inventory.

## Current dataset contract

Step 5 is complete locally: the primary ABIDE-I manifest uses the PCP
`cpac/filt_noglobal/rois_aal` derivative, `rater1_and_func2_ok` QC, and a
minimum of one participant from each diagnosis per held-out site. The fully
checked manifest contains 769 participants across 18 evaluable sites. Step 6
derives a 754-participant technical analysis manifest; see
`docs/step6_connectomes.md` for the exact transformation and exclusions.

## Current paper milestone

The next task is Step 13.2: create the LaTeX manuscript and consume only the
frozen values in `reproducibility/result_snapshot.json` and the generated
tables/figures. See `docs/step13_paper_plan.md` and
`docs/step13_claim_ledger.md`. No additional training or AWS compute is needed
for manuscript construction.
