# Step 12A: Frozen Influence and Robustness Analysis

Status: **contract frozen after the Step 11 result and before executing any
robustness calculation.** This is a secondary analysis and cannot replace or
upgrade the confirmatory Step 11 result.

## Inputs

- The immutable Step 11 analysis package for
  `step10_neural_full_parallel_v1`.
- The audited 18-site connectome elastic-net baseline table.
- No raw-data change, new model fit, new seed, new density, or new endpoint.

Both input manifests are bound by SHA-256 in
`configs/neural_robustness_analysis_v1.json`.

## Analyses

1. Recalculate the BuNN-minus-GCN curve, BuNN-minus-elastic-net curve, and
   primary matched-anchor effective-rank contrasts after excluding each site
   in turn. Each 17-site estimate receives a 10,000-resample paired bootstrap
   interval. The full 18-site estimate remains primary.
2. Construct complete anchored curves separately for each of the five final
   seeds. Report every seed-specific BuNN-minus-GCN and BuNN-minus-baseline
   effect with paired site intervals.
3. Report equal-site mean, participant-weighted mean, and median-site
   summaries for the three core contrasts. Alternative weighting is
   descriptive only.
4. Produce the complete pre-listed family of nine operator/anchor/baseline
   curve contrasts rather than selecting favorable pairs.
5. Aggregate parameter count, fit count, runtime, peak GPU memory, and curve
   performance by operator to document computational efficiency.

## Classification

- `site_sensitive` means leave-one-site point estimates for the primary
  BuNN-minus-GCN effect occur on both sides of zero.
- `seed_sensitive` means the five seed-specific point estimates occur on both
  sides of zero.
- Both may be true. Neither permits a positive override.
- The report must distinguish point-estimate sensitivity from an interval
  that establishes a positive effect.

## Fixed outputs

- `leave_one_site_out.csv` and `site_influence.png`
- `seed_specific_curves.csv`, `seed_configuration_rankings.csv`, and
  `seed_stability.png`
- `alternative_summaries.csv`
- `exhaustive_curve_contrasts.csv`
- `operator_efficiency.csv`
- `robustness_decision.json`
- `analysis_manifest.json`
