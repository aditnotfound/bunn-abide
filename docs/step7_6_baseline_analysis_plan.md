# Step 7.6: Baseline Analysis Contract

Status: **implemented and synthetic-data tested; results remain embargoed.**

This contract fixes the calculations, uncertainty summaries, tables, and
figures for the completed `step7_5_full_baselines_v2` archive before any
predictive result is read. The analysis code recomputes site metrics directly
from participant-level held-out predictions and refuses to continue if they do
not match the runner's saved per-site metric artifact.

## 1. Fixed estimands

- **Primary:** unweighted mean of the 18 held-out-site balanced accuracies,
  separately for each of the three frozen baseline models.
- **Secondary pooled summaries:** balanced accuracy, AUROC, sensitivity, and
  specificity over all held-out participants. These are descriptive only: a
  large site must not dominate the primary cross-site conclusion.
- **Paired contrasts:** all three unordered pairs of the frozen model order
  (`covariates_l2_logistic`, `connectome_elastic_net_logistic`,
  `combined_elastic_net_logistic`). A positive value always means *left model
  minus right model* in unweighted mean site balanced accuracy.
- **Tuning/convergence descriptors:** one selected candidate per model/site
  and a count grouped by model, fit phase, and warning category.

## 2. Fixed uncertainty procedure

- Draw 10,000 bootstrap samples of the 18 held-out sites with replacement.
- Apply each sampled list of site positions to both models in a contrast; this
  preserves the site-level pairing.
- Use the 2.5th and 97.5th percentiles as the two-sided 95% percentile
  interval.
- Use seed `20260803`, inherited from the frozen protocol.
- Do not select a winning model from pooled accuracy, a single site, or a
  confidence interval alone. State only whether a pre-specified contrast shows
  an advantage detected or no advantage detected under this pipeline.

## 3. Frozen analysis outputs

`scripts/analyze_baselines.py` writes a new, non-overwriting output directory:

| File | Purpose |
| --- | --- |
| `per_site_metrics.csv` | All 18 site-level sample/class counts and predictive summaries by model. |
| `model_summary.csv` | Primary site-weighted summary plus descriptive pooled summaries. |
| `paired_balanced_accuracy_differences.csv` | Observed paired differences and 10,000-resample intervals. |
| `selected_hyperparameters.csv` | Selected inner-CV candidates for every model/site. |
| `fit_warning_summary.csv` | Grouped convergence/retry warning record. |
| `per_site_balanced_accuracy.png` | All sites and models, with no pooled visual weighting. |
| `paired_balanced_accuracy_differences.png` | All fixed model contrasts with 95% intervals. |
| `analysis_manifest.json` | Analysis version, frozen estimands, bootstrap settings, and input hashes. |

## 4. Integrity and unblinding guard

The module validates prediction schema, unique participant/model rows, common
site coverage, binary labels, finite probabilities in `[0, 1]`, consistency
with the frozen threshold, complete tuning selection, and exact agreement with
the saved per-site metrics before writing outputs. It has an explicit command
line acknowledgement: `--confirm-unblind-run-id` must match `metadata.json`.
This prevents accidental reading of a completed result archive.

It was tested only against a synthetic three-site dataset and deliberately
corrupted saved metrics. The AWS suite passed 16 tests on 2026-08-07. The
real result files must be read only after the analysis implementation is
committed and its source hashes recorded in the decision log.

## 5. Deferred unblinding command

Run this only after the implementation commit is recorded:

```bash
cd ~/bunn-abide
.venv/bin/python scripts/analyze_baselines.py \
  --run-dir outputs/runs/baselines/step7_5_full_baselines_v2 \
  --protocol configs/baseline_protocol.json \
  --output-dir outputs/analysis/step7_6_full_baselines_v2 \
  --confirm-unblind-run-id step7_5_full_baselines_v2
```

The command is intentionally absent from any automatic runner. It should be
started only as the documented unblinding action, and the generated analysis
manifest must be retained with the output tables and figures.
