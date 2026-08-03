# Step 7: Site-Held-Out Baseline Evaluation

This step provides the non-graph reference models that every later identity,
GCN, and BuNN model must beat under the same evaluation protocol.

## Frozen protocol (Step 7.0)

The protocol in `configs/baseline_protocol.json` was frozen before fitting a
baseline. It specifies 18 outer leave-one-site-out folds, four inner
`StratifiedGroupKFold` folds, seed `20260803`, balanced accuracy as the primary
endpoint, and a 0.5 decision threshold. Imputation, scaling, tuning, and model
fitting must use only the outer-training sites.

The three later baseline models are:

1. L2 logistic regression using age, categorical sex, mean framewise
   displacement, and scan length.
2. Elastic-net logistic regression using 6,670 connectome edges.
3. Elastic-net logistic regression using both sets of features.

Site ID, FIQ, diagnosis-derived variables, PCA, external feature selection,
ComBat, and harmonization are excluded from this phase.

## Baseline table (Step 7.1)

`data/processed/abide_i_baseline_table.csv` contains 754 rows aligned exactly
to the Step 6 connectome artifact. Its `connectome_row` field gives the one-to-
one index into the 6,670-edge feature matrix. The table has complete age, sex,
and scan-length fields; mean framewise displacement is missing for one
participant. That value will be imputed using the median of the relevant
training partition only.

## Frozen splits (Step 7.2)

The generated files are:

- `data/processed/splits/outer_loso_assignments.csv`
- `data/processed/splits/inner_grouped_assignments.csv`
- `data/processed/splits/split_summary.json`

The outer file assigns all 754 participants to exactly one held-out site. The
inner file contains 12,818 assignments: for each outer fold, every outer-
training participant appears once as an inner-validation participant. The
generator independently verifies that test subjects never appear in the
corresponding inner assignments, no validation site appears in that inner
training partition, and every inner validation/test partition contains both
classes.

An independent audit rechecked these properties from the saved CSVs. A second
generation with the same table and seed produced byte-identical assignment
hashes. The small three-participant CALTECH and CMU test sites remain in the
protocol and will be reported separately rather than hidden in a pooled score.

## Not yet performed

No classifier has been fit, no hyperparameter has been selected, and no
predictive metric has been observed. Step 7.3 will implement train-only
preprocessing and the three baseline pipelines, beginning with a two-site
smoke run.
