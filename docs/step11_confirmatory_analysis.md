# Step 11: Audited Confirmatory Neural Analysis

Status: **execution contract and output templates frozen after the score-blind
integrity audit and before neural results were unsealed.**

## Input gate

The accepted input is `step10_neural_full_parallel_v1`, code version
`c246744`. It completed all 18 sites and passed `scripts/audit_neural_full.py`
without reporting any predictive or representation value. The certificate
checked 52,780 prediction rows, 1,260 independently recomputed hidden metric
rows, 158,340 diagnostic rows, 34,272 inner-site rows, 9,324 runtime rows,
1,008 tuning rows, all three worker roots and all 18 canonical site copies.
It reported zero warning rows.

The sealed archive is retained on AWS and locally at
`outputs/archives/step10_neural_full_parallel_v1.sealed.tar.gz`. Its SHA-256 is
`91ffbd55da98064149d90ff3273e80bcd6c4d5d6195d0ac64344747ecdd5082f`.

## Frozen predictive analysis

Five final seeds are averaged within each site and configuration before
site-level inference. GCN and trivial-bundle curves use identity at 0% as
their anchor; the learned-BuNN curve uses learned-local at 0%. The primary
estimand is the normalized trapezoidal area under each site's balanced-
accuracy curve over 0%, 1%, 5%, 10%, and 20% density. The primary contrast is
learned BuNN minus GCN.

All primary uncertainty intervals use 10,000 paired site bootstrap resamples
with seed `20260808` and equal weight for each of the 18 held-out sites.
Density-specific comparisons are secondary and use exact two-sided paired
sign-flip tests with Holm correction across the four nonzero densities within
each named contrast family. The connectome-only elastic-net model is the
frozen non-graph reference.

## Frozen representation analysis

Participant diagnostics are averaged within site/configuration/seed, then
seeds are averaged within site/configuration. The primary endpoint is common-
frame normalized effective rank at layer 2. Its estimand is the BuNN-minus-GCN
difference in normalized density-curve change relative to the corresponding
zero-density anchors. Dispersion, cosine similarity, invariant edge transport
distance, and encoder-to-layer-2 effective-rank change are secondary. Any
relationship with prediction is co-occurrence, not causation.

## Frozen outputs

The guarded analysis produces site/seed metrics, seed-averaged configuration
metrics, site density curves, five confirmatory predictive contrasts, 20
Holm-adjusted density contrasts, site representation summaries, representation
contrasts, runtime/capacity summaries, selected hyperparameters, warning
summaries, three fixed figures, a machine-readable decision summary, and a
hash-complete analysis manifest. It refuses to overwrite an existing output
directory and requires the exact run ID as an explicit unblinding argument.

The synthetic 18-site suite passed all four new tests, and the complete AWS
suite passed 59 tests using isolated import mode. No accepted neural result was
read during implementation or testing.
