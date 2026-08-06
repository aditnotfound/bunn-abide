# Step 7.5: Full Baseline Execution Contract

Status: **running** (`step7_5_full_baselines_v2`, started 2026-08-05T08:01:22Z)

This document defines the complete baseline run before any final held-out-site
result is inspected. Step 7.5 establishes the non-graph reference that the
later identity, GCN, trivial-bundle, and learned-BuNN models must be compared
against. It is not a reduced run and it does not reuse Step 7.4 smoke metrics.

## 1. Frozen scientific contract

The run must use the already frozen artifacts without changing them:

- 754 technically eligible ABIDE-I participants from 18 sites;
- 371 ASD and 383 control participants;
- Fisher-z AAL-116 connectomes with 6,670 lower-triangle features;
- 18 outer leave-one-site-out folds;
- four site-grouped inner folds for model selection;
- seed `20260803` and decision threshold `0.5`;
- training-partition-only imputation, encoding, scaling, tuning, and fitting;
- unweighted mean held-out-site balanced accuracy as the primary endpoint;
- per-site AUROC, pooled balanced accuracy, sensitivity, and specificity as
  secondary endpoints; and
- 10,000 paired site-level bootstrap resamples in the later analysis step.

The models and complete frozen hyperparameter grids are:

| Model | Candidates | Inner fits per outer site | Final fits per outer site |
| --- | ---: | ---: | ---: |
| Covariates-only L2 logistic regression | 4 | 16 | 1 |
| Connectome elastic-net logistic regression | 9 | 36 | 1 |
| Combined connectome-plus-covariate elastic net | 9 | 36 | 1 |
| **Total** | **22** | **88** | **3** |

This gives a minimum of 91 fits per outer site and 1,638 initial fits over all
18 sites. A non-converged elastic-net fit is retried at the frozen higher
iteration limit. In the extreme case where every elastic-net fit needs one
retry, the run can reach 2,970 total fits. Runtime is therefore expected to be
hours or potentially longer; neither the two-site fast smoke nor its capped
iteration limit is a valid estimate of full scientific runtime.

## 2. Expected outputs

A successful run must produce 18 atomically sealed site directories. Before
any metric is interpreted, the aggregate artifacts must have exactly:

| Artifact | Expected rows |
| --- | ---: |
| Out-of-sample predictions | 2,262 (754 participants x 3 models) |
| Held-out-site metric rows | 54 (18 sites x 3 models) |
| Candidate tuning rows | 396 (18 sites x 22 candidates) |
| Inner validation site-score rows | 6,732 |
| Sealed `complete.json` manifests | 18 |

Warning rows are variable and must be reported rather than forced to an
expected count. Each sealed site must contain predictions, test metrics,
tuning scores, inner-site scores, fit warnings, and a hash manifest.

## 3. Required prelaunch gate

The full run may start only after all of the following pass on AWS:

1. The instance is the intended machine and has enough free disk space.
2. The 769 downloaded ROI time-series files, 754-subject analysis manifest,
   connectome artifact, baseline table, frozen splits, and protocol exist.
3. Frozen input and split hashes verify successfully.
4. The complete test suite passes in the deployed environment.
5. The deployed runner, manager, status reader, and configuration hashes match
   the reviewed local files.
6. No baseline runner is active and no launch lock is stale.
7. The proposed run directory does not already exist.
8. The SNS topic configuration exists with owner-only permissions, the EC2
   role can publish to that topic, and the researcher has confirmed an email.
9. The exact run ID and code identifier below are recorded before launch.

The GPU health check remains useful for the later neural models, but Step 7.5
itself is dominated by CPU-based scikit-learn elastic-net fitting. Low GPU use
during this step is expected and is not evidence of a broken run.

## 4. Planned identity and launch command

- Run ID: `step7_5_full_baselines_v2`
- Run kind: `full`
- Execution code identifier: `7ad7246`
- Models: all three frozen baselines
- Held-out sites: all 18 frozen sites
- Fast-smoke override: forbidden

Planned command, to be executed only after explicit authorization:

```bash
cd ~/bunn-abide
scripts/launch_managed_baseline.sh step7_5_full_baselines_v2 \
  --run-kind full \
  --models covariates_l2_logistic connectome_elastic_net_logistic combined_elastic_net_logistic \
  --code-version 7ad7246
```

No `--held-out-sites` or `--fast-smoke` argument is permitted. The managed
launcher must detach the process from SSH, record the PID and log, require SNS
notifications, and prevent a duplicate launch.

## 5. Monitoring without interfering with training

The run is observed through three independent layers:

1. **Process layer:** PID liveness and the detached runner log show whether the
   process is still executing.
2. **Progress layer:** atomic `status.json` and sealed per-site checkpoints show
   the current site, model, stage, heartbeat, and completed-site count.
3. **Alert layer:** SNS sends STARTED and terminal COMPLETE, FAILED, or
   INTERRUPTED notifications.

Email is an alert, not proof of completion. A machine crash or force kill can
prevent a terminal email. Conversely, delayed email does not invalidate sealed
artifacts.

The current heartbeat is updated between candidates, not inside every
individual fit. A valid slow candidate can therefore exceed the default
45-minute stale threshold. During this full run the status reader should use a
180-minute threshold. A stale label is investigated by checking PID liveness,
log movement, CPU activity, and the last sealed checkpoint before any process
is stopped. It is never treated as automatic permission to kill the run.

Suggested check:

```bash
cd ~/bunn-abide
python scripts/check_baseline_run.py \
  --run-id step7_5_full_baselines_v2 \
  --stale-minutes 180
```

Check shortly after STARTED, after the first sealed site, approximately every
2-4 hours while awake, and after any alert. The first sealed site supplies a
runtime estimate for cloud planning only. Its prediction or metric values must
not be opened.

## 6. Failure and recovery rules

- **Transient interruption with unchanged code and inputs:** relaunch the exact
  command with `--resume`. The runner verifies immutable metadata, skips only
  hash-valid sealed sites, and reruns an incomplete site from the beginning.
- **Instance stop or reboot:** restart the instance, repeat environment and
  storage checks, and resume with the identical contract.
- **Out-of-disk or infrastructure failure:** preserve the run directory, repair
  infrastructure, verify existing checkpoints, then resume.
- **Data, split, protocol, or code change:** do not resume this run. Preserve it
  as invalid/incomplete, review the change, assign a new code identifier and
  run ID, rerun tests, and start a new execution.
- **Persistent convergence warnings:** do not silently increase an iteration
  limit beyond the frozen retry rule. Finish the audit, quantify the warnings,
  and decide whether a separately versioned protocol is scientifically needed.

Planned resume command for a transient interruption:

```bash
cd ~/bunn-abide
scripts/launch_managed_baseline.sh step7_5_full_baselines_v2 \
  --run-kind full \
  --models covariates_l2_logistic connectome_elastic_net_logistic combined_elastic_net_logistic \
  --code-version 7ad7246 \
  --resume
```

## 7. Completion audit and results embargo

The run is not scientifically complete merely because a COMPLETE email arrives.
Before looking at model performance, verify:

1. final status is `complete` at 18 of 18 sites;
2. all 18 site manifests and every file hash validate;
3. root aggregate files are present and their hashes match metadata;
4. each of the 754 participants has exactly one prediction per model;
5. no participant or site is missing or duplicated;
6. all 2,262 probabilities are finite and lie in `[0, 1]`;
7. all 54 metric rows are finite and correspond to the frozen sites/models;
8. exactly one candidate is selected for each site/model;
9. tuning and inner-site score counts match the expected contract;
10. convergence/retry records are summarized and retained; and
11. the complete run directory is copied to the local ignored output area
    before scientific analysis.

The integrity audit should operate on row counts, identifiers, schemas,
finiteness, ranges, and hashes without printing performance values. Only after
it passes should Step 7.6 analysis code be run.

## 8. Permitted conclusions

Step 7.5 alone can estimate how the three pre-specified non-graph baselines
generalize to held-out ABIDE-I sites under this pipeline. It cannot establish
a BuNN advantage, biological bundle geometry, clinical diagnostic value,
causal effects, or generalization beyond this cohort and preprocessing path.

## 9. Pre-fit execution incident

The original planned run, `step7_5_full_baselines_v1`, launched successfully
but stopped before its first fit because Boto3 could not infer an AWS region
when publishing the required STARTED notification. It produced zero completed
sites, zero predictions, and zero test metrics. The notification implementation
was corrected to derive the SNS region from the topic ARN and was covered by a
new unit test. The corrected code passed the full intended 13-test suite on
AWS. The failed `v1` directory remains as an audit trail and will not be
resumed; `v2` is a new immutable run with the corrected code identifier.

The `v2` STARTED notification was successfully published before model fitting;
the runner then began the full CALTECH outer fold. This confirms only launch
integrity, not model performance.
