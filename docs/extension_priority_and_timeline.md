# Extension priorities and evidence-based timeline

Status: **planning estimate; full-run timings require measured smoke tests.**

## Timing anchor already measured

The completed Study 1 neural run executed 9,324 fits. Its archived runtime
summary records 81.68 aggregate fit-hours. With three site-parallel workers on
the A10G instance, coordinator timestamps span approximately 28 hours. These
numbers anchor the ABIDE-I graph-model estimates below. They do not establish
the speed of the Wang temporal model or raw ABIDE-II preprocessing.

## Priority order

| Priority | Study | Why it earns inclusion | Intended placement |
| --- | --- | --- | --- |
| 1 | Accepted-checkpoint map/topology interventions | most direct test of whether learned transport and topology affected the completed BuNN predictions | main paper |
| 2 | Synthetic known-geometry suite | establishes whether the implementation can exploit bundle geometry when geometry is necessary, correct, wrong, or unlearnable | main paper |
| 3 | Wang released-code versus leakage-free versus fixed-epoch audit | directly tests whether outer-test-guided selection raises the reconstructed 71.6%-style result | secondary main study or prominent supplement |
| 4 | Targeted five-layer, capacity, orthogonality, and bundle-dimension sensitivity | checks whether the frozen O(2), two-layer design was an unfavorable architectural choice | supplement; main text only if decisive |
| 5 | ABIDE-II external validation | tests transportability of frozen Study 1 choices | main paper only if preprocessing compatibility passes |

## Work and compute estimates

All calendar estimates assume one researcher/Codex workflow, the existing
repository, the same A10G-class GPU, and safe overlap of independent data,
implementation, and analysis tasks. They are predictions, not verified
runtimes. Each full-run range is replaced by a measured estimate after its
one-site smoke.

| Work package | Implementation and testing | Expected full computation | Audit and analysis | Calendar contribution |
| --- | ---: | ---: | ---: | ---: |
| Freeze extension registry and preserve Study 1 | 0.5-1 day | none | 0.5 day | 1 day |
| Checkpoint map/topology interventions | 1.5-2.5 days | 2-8 hours inference | 0.5-1 day | 2-4 days |
| Synthetic known-geometry suite | 2-3 days | 4-16 hours training | 1 day | 3-5 days |
| Targeted five-layer/operator sensitivity | 2-3 days | 36-72 hours after a measured smoke | 1 day | 4-7 days |
| Wang data-reproduction gate | 1-2 days if the released artifact remains accessible | none | 0.5 day | 1.5-3 days |
| Wang PyTorch implementation and numerical tests | 3-5 days | one-site smoke: 4-12 hours reserved | 0.5 day | 4-6 days |
| Wang three-arm full audit | already implemented above | provisional 48-120 wall-hours; smoke must replace this range | 1-2 days | 3-7 days |
| ABIDE-II 10-participant preprocessing gate | 2-4 days | 8-36 hours preprocessing | 0.5-1 day | 3-5 days |
| ABIDE-II full preprocessing, if a compatible derivative exists | 1-2 days integration | 1-3 days | 1 day | 3-5 days |
| ABIDE-II full preprocessing from raw MRI on one four-vCPU instance | 2-4 days integration | roughly 1-3 weeks; must be benchmarked | 1 day | 10-25 days |
| Frozen ABIDE-I-to-II model fitting/evaluation | 1 day | 2-12 hours | 1 day | 2-3 days |
| Manuscript, supplement, figures, claim audit, and PDF QA | 3-5 days | none | included | 3-5 days |

## End-to-end calendar scenarios

### Recommended paper without raw ABIDE-II reprocessing

Includes checkpoint interventions, synthetic geometry, the Wang three-arm
audit, targeted architecture sensitivities, and the finished manuscript.

**Estimated elapsed time: 14-21 calendar days.**

This assumes the Wang CC200 artifact is accessible and its smoke runtime fits
within the provisional range. The mechanism and Wang implementation work can
partly overlap.

### Recommended paper plus compatible ABIDE-II derivative

Adds the feasibility gate and locked external validation without reprocessing
raw ABIDE-I/II scans through a new pipeline.

**Estimated elapsed time: 18-28 calendar days.**

### Full cross-release reprocessing from raw MRI

If no compatible ABIDE-II derivative exists and both releases must be
reprocessed through one frozen pipeline, preprocessing becomes the critical
path.

**Estimated elapsed time: 28-45 calendar days on modest compute.**

Parallel CPU preprocessing could shorten this, but only after a 10-participant
benchmark measures time, memory, storage, and failure rate. Adding unmeasured
instances is not treated as a guaranteed speedup.

## Promotion and stop rules

- A checkpoint intervention enters the main paper only if it distinguishes at
  least two competing explanations.
- Synthetic results enter only if positive and negative controls behave as
  designed; a benchmark that rewards BuNN everywhere is rejected as
  non-diagnostic.
- The Wang audit proceeds to a full run only after the data cohort is traced,
  hand-calculated EdgeConv tests pass, and the one-site smoke produces a real
  timing measurement.
- ABIDE-II enters the paper only if preprocessing comparability and cohort
  independence are documented before predictive output is viewed.
- No favorable post-hoc extension changes the classification of Study 1. It is
  reported as a separately labelled mechanism, architecture, leakage, or
  external-validation study.
