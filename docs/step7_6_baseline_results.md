# Step 7.6: Audited Baseline Results

## Scope and integrity

This is the first unblinded result of the frozen ABIDE-I baseline protocol:
754 technically eligible participants from 18 held-out sites, with all
preprocessing and model selection confined to training sites. The completed
run passed the score-blind integrity audit before unblinding. The analysis
recomputed each site-level metric from participant predictions and found exact
agreement with the runner's sealed per-site metric artifact.

The full analysis output package is ignored by Git but retained locally as
`outputs/analysis/step7_6_full_baselines_v2.tar.gz`. Its AWS/local SHA-256 is
`c60d4f4918a1acfeab0bd2d8fdb1a4b2f171b1fd951fcad12c7b38b2e163267a`.
The generated `analysis_manifest.json` records hashes for every result input.

## Pre-specified primary result

The primary endpoint is the unweighted mean held-out-site balanced accuracy;
each of the 18 sites contributes once regardless of its number of participants.

| Model | Mean site balanced accuracy | Pooled balanced accuracy | Pooled AUROC |
| --- | ---: | ---: | ---: |
| Covariates-only L2 logistic regression | 0.565 | 0.563 | 0.597 |
| Connectome elastic-net logistic regression | **0.640** | **0.650** | **0.679** |
| Connectome-plus-covariate elastic net | 0.634 | 0.643 | 0.677 |

The 10,000-resample paired site bootstrap produced the following contrasts.
Positive differences below are expressed in the direction stated in the first
column, rather than choosing a direction after seeing the results.

| Contrast | Mean site BA difference | 95% paired bootstrap interval | Interpretation |
| --- | ---: | ---: | --- |
| Connectome minus covariates | +0.075 | +0.021 to +0.123 | Connectome advantage detected under this pipeline. |
| Combined minus covariates | +0.068 | +0.019 to +0.114 | Combined-model advantage detected under this pipeline. |
| Connectome minus combined | +0.007 | -0.020 to +0.006 | No advantage detected between the two connectome-containing baselines. |

The displayed interval for the last row is the sign-reversed version of the
stored `combined minus connectome` contrast, whose interval was -0.006 to
+0.020. This makes the table direction consistent with the first column; it
does not change the bootstrap calculation.

There were no recorded fit-warning rows. All site-level values, selected
hyperparameters, and fixed figures are retained in the analysis package.

## What this does and does not establish

Within this exact AAL-116, no-global-signal-regression, ABIDE-I, held-out-site
pipeline, regularized connectome features provided more cross-site predictive
information than the restricted covariate set. Adding the same covariates to
connectomes did not show a detectable advantage over connectomes alone.

This does **not** establish a diagnostic tool, a causal ASD mechanism, or a
biological explanation of functional-connectivity edges. It also does not make
the two connectome-containing baselines equivalent: the 95% interval for their
difference includes both a small advantage for either one. Several held-out
sites are very small (including two sites with three participants), so the
primary site-weighted summary and its interval should be read alongside the
full per-site figure rather than as evidence that every site behaves alike.

## Consequence for the neural study

Phase 8 will retain all three baselines. The connectome-only elastic net is the
empirically strongest non-graph reference that identity propagation, GCN,
trivial bundle transport, and learned BuNN must beat under the same held-out
site protocol. A neural model that only exceeds covariates-only would not meet
the project's main predictive criterion.
