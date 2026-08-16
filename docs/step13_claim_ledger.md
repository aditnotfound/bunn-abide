# Step 13 Claim Ledger

This ledger defines what the paper may say from the frozen ABIDE-I evidence at
commit `a4def2a1f14f1bbff71356d0689eeeee4f405f4a`. Exact machine-readable values
are in `reproducibility/result_snapshot.json`. All conclusions are conditional
on the specified C-PAC `filt_noglobal`, AAL-116, positive-edge, ABIDE-I
pipeline and held-out-site evaluation.

## Primary and supporting claims

| ID | Evidence class | Supported paper statement | Evidence | Boundary |
| --- | --- | --- | --- | --- |
| C01 | Cohort/QC | The analysis retained 754 participants (371 ASD, 383 controls) from 18 sites after 15 technical exclusions from 769 downloaded time-series files. | `configs/abide_i_analysis_manifest.json`; snapshot `cohort` | This is technical eligibility, not population representativeness. |
| C02 | Classical baseline | The connectome elastic net was the strongest classical baseline by equal-site balanced accuracy: 0.6401, versus 0.6336 for connectome plus covariates and 0.5652 for covariates only. | `outputs/analysis/step7_6_full_baselines_v2/model_summary.csv`; snapshot `classical_baselines` | Do not call 0.6401 clinical diagnostic accuracy. |
| C03 | Primary confirmatory | The BuNN-minus-GCN normalized performance-curve contrast was -0.00958 (95% site-bootstrap CI -0.03665 to 0.01146; exact sign-flip p=0.524). No BuNN advantage was detected. | `confirmatory_predictive_contrasts.csv`; snapshot `confirmatory_predictive_contrasts.learned_bunn_curve_minus_gcn_curve` | The interval includes zero. Do not claim BuNN is universally worse than GCN. |
| C04 | Practical-margin interpretation | The upper confidence bound for C03 was below the pre-specified +0.03 balanced-accuracy margin, excluding a gain of that size under this estimand and pipeline. | Same as C03 | This is not universal model equivalence and does not exclude smaller advantages. |
| C05 | Supporting confirmatory | The BuNN curve was 0.05516 below the site-matched connectome elastic net (95% CI -0.08297 to -0.02752; exact sign-flip p=0.00176). | `confirmatory_predictive_contrasts.csv`; snapshot `confirmatory_predictive_contrasts.learned_bunn_curve_minus_connectome_elastic_net` | This comparison is conditional on the frozen tuning budgets, inputs, and evaluation. |
| C06 | Primary representation | The matched-anchor layer-2 normalized effective-rank contrast was -0.00719 (95% CI -0.01310 to -0.00105), opposite the proposed representation-preservation advantage. | `representation_contrasts.csv`; snapshot `primary_representation_contrast` | Say the diagnostic did not support the proposed transfer. Do not claim a biological mechanism or causal mediation. |
| C07 | Robustness | For the primary BuNN-GCN contrast, 1 of 18 leave-one-site estimates and 2 of 5 seed estimates were positive; no favorable interval excluded zero, while one unfavorable seed interval did. The small contrast was site- and seed-sensitive. | `leave_one_site_out.csv`; `seed_specific_curves.csv`; snapshot `robustness` | Do not select a favorable site exclusion, seed, or participant weighting as the headline result. |
| C08 | Alternative summaries | The BuNN-GCN contrast was -0.00958 by equal-site mean, +0.00297 by participant-weighted mean, and -0.00439 by median site difference. | `alternative_summaries.csv`; snapshot `robustness.alternative_summaries` | Equal-site weighting remains primary; the sign change supports caution, not replacement of the estimand. |
| C09 | Efficiency | BuNN used 8,529 parameters and 31.14 fit-hours versus GCN's 5,889 parameters and 15.09 fit-hours; its curve balanced accuracy was lower (0.5849 versus 0.5945). | `operator_efficiency.csv`; snapshot `operator_efficiency` | Runtime is implementation- and hardware-dependent. Describe this execution, not universal computational complexity. |
| C10 | Decision rule | None of the pre-specified predictive, representation, or baseline conditions for a complete BuNN advantage was satisfied. | `decision_summary.json`; snapshot `step11_decision` | Report a conditional negative/null result, not proof that bundle transport can never help. |

## Permitted synthesis

The paper may conclude:

> Under the specified ABIDE-I pipeline, learned bundle transport did not
> produce a more favorable held-out-site performance-density curve than GCN,
> did not outperform the strongest regularized connectome baseline, and did
> not show the proposed matched-anchor representation-preservation advantage.
> The small BuNN-GCN predictive contrast was sensitive to site weighting and
> training seed.

The paper may describe the work as a controlled operator audit and a rigorous
negative/null result. It may say that the evidence did not support transfer of
the proposed anti-collapse advantage to this setting.

## Prohibited or unsupported claims

The manuscript must not state or imply that:

- BuNN is universally inferior, useless, or disproven;
- the study proves model equivalence beyond the pre-specified practical margin;
- functional-correlation signs represent excitation, inhibition, causal flow,
  anatomical connections, or graph-learning heterophily;
- learned transport discovers biological bundle geometry;
- representation changes caused predictive changes;
- increasing density proves over-smoothing without the qualified diagnostic
  language used in the Methods;
- the models constitute a clinical diagnostic system;
- the ABIDE-I result automatically generalizes to ABIDE-II, another atlas,
  another preprocessing pipeline, or another condition; or
- a favorable seed, excluded site, density, or weighting may replace the
  confirmatory equal-site density-curve result.

## Required limitations

The Discussion must state that the result is limited to one dataset, atlas,
preprocessing choice without global-signal regression, positive-edge graph
construction, connectivity-profile node features, architecture/backbone,
tuning budget, density grid, and multi-site sample. The paper must disclose the
additional BuNN parameter count, the learned-local capacity control, and the
absence of locked external validation.
