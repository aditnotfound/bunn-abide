# E1 analysis plan (frozen before unblinding)

## Purpose

E1 asks what the already-trained learned-BuNN checkpoints actually depend on. It does not retrain models and cannot rescue or replace the frozen Study 1 result. The analysis compares each checkpoint with itself after a pre-specified inference intervention.

## Primary analysis

The reference is unaltered learned-BuNN inference. Four primary interventions are tested: identity maps, node-shuffled learned maps, random orthogonal maps, and degree-preserving graph rewiring. Effects are always reported as intervened minus unaltered balanced accuracy in percentage points; a negative value therefore means that the intact component supported held-out classification.

For randomized interventions, balanced accuracy is computed separately for every stored permutation and model seed. The 100 permutation estimates are averaged within model seed and site-density, followed by the five model seeds. The four density-specific paired effects are averaged within each held-out site. The primary estimate is then the unweighted mean of the 18 site effects.

Uncertainty uses 10,000 paired site bootstrap samples with seed 2026081401. Two-sided exact paired sign-flip tests enumerate all 2^18 site sign assignments. The four primary p-values receive Holm adjustment. Density-specific estimates are descriptive secondary results and cannot replace the overall primary contrast.

## Secondary analysis

Secondary endpoints are absolute probability change, classification-flip fraction, AUROC change, sensitivity change, specificity change, and changes in the four final-layer gauge-aware diagnostics. They receive paired site-bootstrap intervals without confirmatory p-values. A descriptive Spearman association across the 72 site-density cells asks whether representation and balanced-accuracy changes co-occur. It is not a mediation analysis.

The encoded-node permutation condition is an engineering equivariance control only. Its numerical tolerances were already checked by the score-blind site auditors and it is excluded from scientific contrasts.

## Fixed presentation

The main output is a four-row forest plot of overall primary effects. A second figure shows density-specific effects for all four interventions. A third figure summarizes descriptive representation--prediction associations. Site-density effects remain available as a complete table; no site, seed, density, or intervention may be omitted after values are opened.

## Claim boundaries

Sensitivity to learned maps shows computational use of those maps, not biological bundle geometry. Sensitivity to rewired topology shows dependence on the constructed positive-edge graph, not anatomical connectivity or causal neural flow. Representation changes and prediction changes may be described as co-occurring only. The result is conditional on the accepted ABIDE-I cohort, C-PAC no-GSR AAL pipeline, trained checkpoints, densities, and perturbations.

## Execution gate

The analyzer must first pass synthetic known-answer tests covering aggregation order, equal-site weighting, contrast direction, paired bootstrapping, exact sign-flip testing, Holm correction, missing-cell rejection, and gauge-aware metric handling. Real values may be opened once, using the exact acknowledgement stored in `configs/extensions/e1_analysis_v1.json`. The analyzer writes no scientific values to the terminal. An independent auditor must reproduce all tables before interpretation.
