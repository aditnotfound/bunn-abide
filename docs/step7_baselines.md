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

## Step 7.3: implemented baseline runner

`scripts/run_baselines.py` consumes the frozen table and split files rather
than generating any new assignments. Before fitting it verifies the frozen
SHA-256 hashes and the table/connectome row alignment. For every outer site it
then:

1. builds only the allowed covariate, connectome, or combined feature set;
2. fits imputation, one-hot encoding, scaling, tuning, and classification only
   on the relevant training partition;
3. evaluates each candidate using the mean of per-site balanced accuracies in
   the pre-generated grouped inner folds;
4. uses the pre-specified lower-`C`, then lower-`l1_ratio` tie-break;
5. refits the selected candidate on all outer-training sites; and
6. writes predictions, test metrics, tuning records, per-inner-site scores,
   convergence warnings, metadata, and artifact hashes.

The runner has tests covering training-only imputation, equal site weighting,
tie-breaking (including the covariate model's absent `l1_ratio`), the explicit
fast-smoke override, and the pre-existing split/connectome tests. The complete
AWS test suite passed before the smoke test.

## Step 7.4: two-site engineering smoke test

The completed smoke run is `step7_4_fast_smoke_nyu_caltech` on the AWS
instance. It held out NYU (155 participants) and CALTECH (3 participants), ran
all three pipelines, preserved the real four grouped inner folds, and wrote 474
out-of-sample predictions (158 participants times three models). All six saved
site/model metric rows were finite and all expected artifacts were present.

This is deliberately **not** a scientific result. To keep the smoke test an
engineering check instead of an expensive partial experiment, the invocation
used `--fast-smoke`. That recorded override used only the lowest candidate from
each frozen grid and capped elastic-net fitting at 100 iterations with no retry.
It produced 20 recorded `ConvergenceWarning`s (16 inner fits and 4 final fits),
which are expected under that cap. Its metrics, selected settings, and warnings
must not be pooled with or compared to the later full evaluation.

Two earlier smoke attempts are retained in the AWS run history: the first
stopped before predictions because covariate tie-breaking mishandled an empty
stored `l1_ratio`; a regression test and fix were added. The second used the
full grid and was deliberately interrupted after showing that a complete
72-fit elastic-net grid is too expensive for an interactive smoke check on the
four-vCPU machine. Neither attempt produced test predictions.

## Next: full baseline evaluation

Step 7.5 must use `--run-kind full` without `--fast-smoke`, include all 18
frozen held-out sites, and retain the unmodified grids and retry rules in
`configs/baseline_protocol.json`. The smoke runtime should be used for cloud
scheduling only; it does not justify changing the frozen scientific protocol.
The tested checkpoint/resume, status, managed-launch, and notification design
documented in `docs/step7_run_control.md` must be used before Step 7.5. The
complete prelaunch, monitoring, recovery, and integrity contract is recorded in
`docs/step7_5_full_baseline_plan.md`.
