# Final revision analysis plan (v1)

Frozen: 15 August 2026, before the revision-only calculations were run.

This plan adds evidence that can be computed from the accepted data and outputs. It does not retrain a model, alter the frozen Study 1/E1/E2/RBF results, or turn an exploratory result into a confirmatory one.

## 1. Cohort accounting

Reconstruct the participant cascade from all 1,112 ABIDE-I phenotype rows through diagnosis validity, file availability, the two manual QC gates used by the project, eligibility of the 18-site analysis set, and the final zero-variance-ROI exclusion. Report totals and ASD/control counts at every gate. Report per-site attrition and observed mean framewise displacement for retained and excluded groups when available. The motion summary is descriptive; it is not evidence that motion caused exclusion or bias.

## 2. Predictive-result sensitivity

Keep the accepted equal-site normalized trapezoidal density curve as primary. Add:

- participant-weighted site-cluster bootstrap intervals;
- a uniform-density-grid sensitivity analysis;
- equal-site bootstrap intervals for each model-level result;
- a transparent statement that the normalized trapezoid gives 62.5% of its weight to the 10% and 20% density points.

These calculations clarify weighting dependence. They do not replace the frozen estimand.

## 3. E1 intervention comparisons

Compute all six pairwise differences among identity maps, learned-map shuffling, random orthogonal maps, and degree-preserving topology rewiring. Average each site's effect across the four nonzero densities, use paired site-level intervals and exact sign-flip tests, then apply Holm correction across the six exploratory comparisons. These comparisons are post-hoc and will be labelled accordingly.

## 4. Heat-operator regime

For every participant graph at 1%, 5%, 10%, and 20% density, construct the same binary graph used in Study 1. Compute the spectrum of the random-walk Laplacian through its symmetric-similar form. Summarize the eigenvalues and the exact response at the implemented diffusion time, `exp(-lambda)` for `t=1`. This describes how strongly the fixed heat operator contracts graph modes in the actual graphs; it does not by itself explain prediction errors.

## 5. Report repairs

The standalone submission copy will:

- replace significance-to-equivalence language with `no advantage detected` where intervals cross zero;
- describe `pre-specified`, not `pre-registered`, analyses;
- document the complete operator and transport-map parameterization;
- distinguish common-frame absolute rank from matched-anchor density-change diagnostics;
- label all RBF comparisons and the new analyses as post-hoc;
- give the exact QC cascade rather than calling all excluded records manual-QC failures;
- preserve the accepted data, results, and claim boundary.

An optional transport-angle analysis is outside this frozen revision contract. It may be added only under a separate dated protocol if the mandatory report repairs, compilation, and compliance work are already complete.
