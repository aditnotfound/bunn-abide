# Step 6: Connectome Construction and Technical QC

Step 6 converts the frozen PCP AAL ROI time series into model-independent,
Fisher-z functional-connectome artifacts. It does not train, tune, or compare
any predictive model.

## Input and technical eligibility

The parent manifest contains 769 participants across 18 sites. Every raw file
was parsed as a `T x 116` AAL matrix after its `#` ROI-label header. A row is
technically eligible only when it has a valid, consistent 116-label header,
finite numeric values, non-zero variance at every ROI, and a finite
Pearson/Fisher-z connectome.

Fifteen participants failed that non-negotiable calculation requirement because
at least one ROI had zero variance. They are recorded, with subject, site,
class, and reason, in `data/processed/abide_i_technical_exclusions.csv`; the
parent manifest remains unchanged. The resulting analysis manifest has 754
participants across the same 18 sites: 371 ASD and 383 controls. All 18 sites
still contain both classes, so held-out-site balanced accuracy remains defined.

## Transformation

For every retained subject:

1. Compute the 116-by-116 Pearson correlation matrix from the ROI time series.
2. Apply Fisher-z to off-diagonal entries.
3. Set the diagonal to zero.
4. Store the full matrix as graph node features and the strict lower triangle
   as the 6,670-edge elastic-net feature vector.

This is a computational representation of functional association. It does not
identify structural connections, causal flow, excitation, or inhibition.

## Completed artifacts

- `data/processed/abide_i_analysis_manifest.csv` — 754 retained participants.
- `data/processed/abide_i_technical_exclusions.csv` — 15 explicit technical
  exclusions.
- `data/processed/abide_i_connectomes_fisher_z.npz` — arrays of shape
  `(754, 116, 116)` and edge features of shape `(754, 6670)`.
- `data/processed/abide_i_connectome_qc.csv` — per-subject time points,
  value ranges, Fisher-z ranges, and clipping count.
- `data/processed/abide_i_connectome_metadata.json` — hashes and build
  provenance.
- `outputs/figures/step6_connectome_qc.png` — one visual QC artifact, not a
  scientific result figure.

All generated data are intentionally ignored by Git. Their hashes and the
reproduction contract are versioned in `configs/abide_i_analysis_manifest.json`.

## Validation completed

- Full preflight: 754 retained, 15 documented exclusions.
- Site-diverse smoke build: 18 connectomes, one from each site, no failures.
- Full build: 754 finite symmetric connectomes, zero diagonals, and no perfect
  off-diagonal correlations requiring clipping.
- Stored artifact check: a retained subject was recomputed from raw time series
  and matched the stored connectome exactly at float32 precision.

## Next step

Step 7 creates frozen outer leave-one-site-out splits and the non-graph
baselines. No scaling, feature selection, tuning, or calibration may use a
held-out site.
