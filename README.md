# BuNN ABIDE-I Operator Audit

Reproducible evaluation of identity propagation, GCN, and Bundle Neural Networks on ABIDE-I functional connectomes under site-held-out validation.

## Status

ABIDE-I manifest, technical QC, and Fisher-z connectome construction are complete.
The full 18-site non-graph baseline run is complete, independently audited, and
analysed with pre-specified code. Connectome-only elastic net achieved the
strongest site-weighted baseline result; the paired 95% bootstrap interval for
its advantage over covariates excluded zero. See
`docs/step7_6_baseline_results.md` for the exact result and its limits.

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
checked manifest contains 769 participants across 18 evaluable sites. Step 6
derives a 754-participant technical analysis manifest; see
`docs/step6_connectomes.md` for the exact transformation and exclusions.

## Next milestone

Steps 7.0--7.6 are complete: the protocol, aligned baseline table, frozen
grouped site-held-out splits, leakage-safe baseline runner, engineering smoke
test, checkpoint/resume/status layer, full 18-site execution, audit, and
pre-specified baseline analysis are recorded. Next: implement the shared
neural backbone and its identity, GCN, trivial-bundle, and learned-BuNN
propagation operators before launching any neural experiment.
