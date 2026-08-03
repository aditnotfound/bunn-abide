# BuNN ABIDE-I Operator Audit

Reproducible evaluation of identity propagation, GCN, and Bundle Neural Networks on ABIDE-I functional connectomes under site-held-out validation.

## Status

Project setup plus ABIDE-I manifest tooling. No models, experiments, or results are included yet.

## Layout

- `data/`: local raw and processed data; not committed.
- `docs/`: frozen protocol, decisions, and experiment notes.
- `configs/`: versioned experiment configurations.
- `src/`: reusable implementation modules.
- `scripts/`: data preparation and training entry points.
- `tests/`: unit and leakage-prevention tests.
- `outputs/`: generated runs, figures, and tables; not committed.

## Current dataset contract

Step 5 is complete locally: the primary ABIDE-I manifest uses the PCP
`cpac/filt_noglobal/rois_aal` derivative, `rater1_and_func2_ok` QC, and a
minimum of one participant from each diagnosis per held-out site. The fully
checked manifest contains 769 participants across 18 evaluable sites. See
`docs/step5_abide_manifest.md` for the exact command and decision rationale.

## Next milestone

Perform the Step 6 ROI time-series download and parsing smoke test.
