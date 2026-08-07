# Complete Research Execution Plan

Status: **active**. Steps 1--7.5 are complete. The completed Step 7.5 archive
passed its score-blind integrity audit; Step 7.6 analysis remains blinded until
its code, estimands, and output templates are frozen.

## Research question

On the pre-specified ABIDE-I pipeline, does learned bundle transport produce a
more favorable held-out-site performance-versus-density curve than identity
propagation and ordinary GCN aggregation, and does any advantage exceed the
strongest regularized connectome-plus-covariate baseline?

The study treats BuNN solely as a computational propagation operator. It does
not interpret functional-correlation signs as excitation, inhibition, causal
flow, anatomical connectivity, or node-label heterophily.

## Phase map

| Phase | Work | Gate to advance |
| --- | --- | --- |
| 1-6 | Repository, ABIDE manifest/download, technical QC, and connectomes | Completed and hash-checked |
| 7.0-7.4 | Frozen baseline protocol, aligned table/splits, tested runner, smoke and recovery tests | Completed; smoke is engineering-only |
| 7.5 | Full 18-site non-graph baseline execution | Completed; independent integrity audit passed before metrics were viewed |
| 7.6 | Pre-specified baseline analysis | Completed; audited results analysed with frozen code and archived |
| 8 | Shared neural architecture and operator implementation | Core data/operators pass synthetic and real-data GPU checks; fold-aware runner remains Step 9 |
| 9 | Training-only engineering pilot and configuration freeze | Completed; pilot audited and full training/analysis contracts frozen before accepted neural results |
| 10 | Full identity/learned-local/GCN/trivial-bundle/BuNN evaluation | Runner/auditor implementation must pass score-blind recovery smoke, then all folds/seeds are sealed and audited |
| 11 | Confirmatory statistics and mechanism analysis | Claims follow pre-specified decision rules |
| 12 | Optional robustness/external evaluation | Core result is complete first |
| 13 | Paper and reproducibility package | Every claim traces to an audited artifact |

## Step 7.5 - full baselines

The corrected `v2` execution completed 1,638 initial fits across 18 held-out
sites with resumable site checkpoints and SNS alerts. It passed its independent
completion audit before any final metric was interpreted. The pre-fit `v1`
notification-region failure remains retained as an engineering incident.

## Step 7.6 - baseline analysis

### Build before reading performance

1. Create a read-only integrity auditor that checks hashes, schemas, exact row
   counts, participant uniqueness, finite probabilities, and full site/model
   coverage without printing metric values. **Completed for Step 7.5.**
2. Create and test an analysis module on synthetic predictions.
3. Freeze the estimands and table/figure templates.
4. Record the analysis-code commit and input run hash in the decision log.
5. Only then unblind the full baseline result files.

**Completed.** The frozen implementation passed 17 AWS tests and recomputed
all participant-level metrics consistently with the sealed runner artifacts.
The analysis result and limits are recorded in
`docs/step7_6_baseline_results.md`. Phase 8 must compare neural models against
all three baselines; the connectome-only model is the empirically strongest
reference in this run, not a reason to drop the other controls.

### Required baseline analyses

- Calculate the unweighted mean of the 18 site-level balanced accuracies for
  each model as the primary summary.
- Use 10,000 paired site-level bootstrap resamples. Resample sites once per
  replicate and apply that same draw to every compared model.
- Report paired differences and confidence intervals, not only separate model
  intervals.
- Report every site's sample size, class balance, balanced accuracy, AUROC,
  sensitivity, and specificity. Flag very small sites rather than hiding them.
- Report pooled balanced accuracy only as secondary because a large site must
  not dominate the primary multi-site conclusion.
- Summarize selected hyperparameters, retry rates, and convergence warnings.
- Produce a baseline result table, per-site forest/point plot, and paired
  model-difference plot.

Interpretation must use language such as "advantage detected" or "no advantage
detected." Absence of a practically meaningful gain may be claimed only when
the relevant upper confidence bound excludes the pre-specified three-
percentage-point balanced-accuracy margin.

## Step 8 - neural operator implementation

**Core implementation complete.** The graph representation, four matched
operators, common-frame diagnostics, synthetic tests, real-input validation,
and GPU forward/backward smoke are documented in
`docs/step8_neural_implementation.md`. No neural predictive metric exists.
The fold-aware runner, checkpoint/resume, and training policy deliberately
remain the next engineering gate.

### 8.1 Shared data representation

- Each participant is one graph with 116 AAL regions as nodes.
- Each node receives its full functional-connectivity row as a globally
  informed feature vector.
- Symmetric propagation graphs retain the strongest positive off-diagonal
  edges at densities 0%, 1%, 5%, 10%, and 20%.
- Density selection is deterministic and its tie behavior is tested.
- The 0% condition uses identity/self propagation and acts as the no-neighbor-
  aggregation reference.

### 8.2 Shared neural backbone

Implement one common encoder, hidden width, depth, nonlinearity, normalization,
dropout, pooling, classifier, optimizer, early-stopping rule, epoch budget, and
tuning budget. Change only the propagation component where possible:

1. identity/no-neighbor propagation;
2. normalized GCN aggregation;
3. trivial-bundle diffusion with fixed identity transports; and
4. learned orthogonal BuNN transport.

Report exact parameter counts and compute cost. BuNN transport may add capacity
that cannot be removed by ordinary width matching, so the trivial-bundle
control is required to distinguish diffusion from learned transport. Add a
separate transport-capacity control only if it can be defined without changing
the primary experiment after results are seen.

### 8.3 Gauge-aware representation measurements

Raw BuNN embeddings from node-specific coordinate frames must not be stacked
and compared directly. Before effective rank, dispersion, or cosine similarity
is calculated, embeddings must be transported into a clearly defined common
reference frame. Also report a justified invariant edge quantity such as

`||h_u - T_(v->u) h_v||_2`.

GCN, identity, and trivial-bundle controls use the corresponding identity
transport convention. The paper must state exactly at which layer and over
which nodes/subjects each diagnostic is computed.

### 8.4 Required tests

- correct graph batching and participant isolation;
- exact feature/label/site alignment with the frozen cohort;
- no preprocessing or normalization fitted on held-out data;
- expected edge counts and symmetry at each density;
- stable 0% identity behavior;
- trivial-bundle/identity-transport equivalence where mathematically expected;
- orthogonality of learned transport maps within numerical tolerance;
- correct forward and inverse transport orientation;
- gauge transformation/invariance tests for diagnostic quantities;
- finite forward pass, backward gradients, loss, and probabilities;
- deterministic seed behavior within documented tolerances; and
- checkpoint/resume equivalence.

## Step 9 - engineering pilot and configuration freeze

The pilot is for correctness and runtime measurement, not scientific model
selection.

### Frozen pilot execution

The first pilot uses the existing outer split with `CALTECH` held out and
excluded from all model tensors, and grouped inner-validation fold 0. It runs
the four already specified operators over every already specified density for
three epochs each, with the shared 116-to-32 two-layer backbone, batch size 8,
AdamW (`1e-3` learning rate, `1e-4` weight decay), and seed `20260803`.
These are engineering settings, not a result-driven tuning decision. The pilot
records only fitting/validation BCE loss, runtime, GPU peak allocation,
checkpoint integrity, and failure state. It does not compute probabilities,
balanced accuracy, AUROC, thresholds, or any held-out-site prediction.

`scripts/run_neural_pilot.py` and
`scripts/launch_managed_neural_pilot.sh` implement the pilot. They require
frozen input hashes, validate an immutable resume contract, save one atomic
checkpoint after each epoch, maintain a live `status.json`, and require SNS
start/terminal notifications in managed execution. A completed checkpoint must
contain exactly the planned epochs before a cell is counted as complete.
Before the pilot is accepted, one managed invocation intentionally stops after
its first durable epoch checkpoint; the same named run must then resume under
the unchanged contract and finish. This tests recovery on the actual GPU
workflow rather than assuming the unit test alone is enough.

1. Use only outer-training data and its grouped inner partitions.
2. Exercise all four operators, all densities, batching, checkpointing,
   representation logging, and recovery on a deliberately labelled pilot.
3. Confirm the GPU is used, measure memory/time per epoch, and choose batch
   size for reliability rather than maximum utilization.
4. Diagnose exploding/vanishing gradients, NaNs, early stopping, and transport
   orthogonality.
5. Fix implementation defects and rerun tests.
6. Freeze the neural tuning grid, maximum epochs, patience, seeds, primary
   checkpoint rule, and failure/retry policy.
7. Record the final code/config hashes before any outer held-out metric is
   inspected.

If an outer test fold is used for an engineering end-to-end check, its metrics
must remain automatically hidden and the run must never enter the scientific
results. Development must not repeatedly query favorable test sites.

## Step 10 - full neural evaluation

Run identity, GCN, trivial-bundle, and learned BuNN using the same 18 outer
sites and four grouped inner folds as the baselines. Use the frozen density
grid and an equal tuning budget. Hyperparameters are selected using training
sites only; the held-out site is evaluated once per final seed.

Use five pre-specified final seeds for each neural configuration. Every
site/operator/density/seed unit needs:

- immutable run metadata and code/config/data hashes;
- checkpoint and resume support;
- probability-level out-of-sample predictions;
- selected hyperparameters and training curves;
- parameter count, runtime, and resource use;
- gauge-aware representation diagnostics; and
- warnings/failure states rather than silent omission.

Before launch, estimate the full matrix from measured pilot runtime and decide
whether one A10G is practical. More compute may shorten wall time, but it must
not change folds, search budgets, seeds, or models. Reduced seeds, densities,
sites, or controls are not acceptable substitutes for a full primary run.

## Step 11 - confirmatory analysis

### Predictive comparison

- Primary neural summary: unweighted mean held-out-site balanced accuracy
  across the complete density curve.
- Report paired site-level differences with 10,000 paired bootstrap resamples.
- Model the operator-by-density pattern using a pre-specified analysis that
  respects repeated observations from sites and seeds.
- Report individual-site and individual-seed results, uncertainty intervals,
  and effect sizes.
- Compare every neural model with identity, GCN, trivial bundle, and the
  strongest Step 7 baseline.

### Representation comparison

- Plot gauge-aware effective rank, normalized dispersion, similarity, and
  invariant transport-distance measures by layer and density.
- Test whether BuNN's representation curve changes less adversely with density
  than GCN's.
- Treat association between representation preservation and prediction as
  co-occurrence. The experiment does not prove that one caused the other.

### Decision rule

A positive architectural conclusion requires all three:

1. learned BuNN transport preserves gauge-aware representation diversity more
   strongly than GCN as density increases;
2. BuNN improves held-out-site prediction over GCN and identity propagation;
   and
3. BuNN exceeds the strongest regularized non-graph baseline.

If only the representation condition holds, conclude that bundle transport
changed propagation behavior but no predictive advantage was detected. If
neither holds, conclude that no transfer of the proposed anti-collapse benefit
was detected under this ABIDE-I pipeline. A smaller or mixed result must be
reported as such rather than promoted to broad BuNN superiority.

## Step 12 - robustness and external evaluation

These are secondary and must not delay or redefine the core study:

### High-priority robustness, if resources permit

- sensitivity to global-signal-regression versus no-GSR preprocessing;
- negative-edge handling as a computational choice, without biological sign
  claims;
- alternative density/tie sensitivity close to the frozen grid; and
- capacity/runtime-normalized comparison.

### External evaluation

ABIDE-II can be a locked external ASD evaluation only after preprocessing and
atlas compatibility are established and the complete ABIDE-I analysis is
frozen. ADHD-200 or another condition would test transportability to another
task; it is not a replication of the ASD result. These datasets are future
extensions, not permission to search for a favorable diagnosis.

## Step 13 - paper and reproducibility package

### Paper structure

1. **Introduction:** motivate aggregation failure on globally informed
   connectome features and introduce bundle transport as an operator-level
   hypothesis.
2. **Compact refinement statement:** explain in one paragraph that FC-edge
   sign does not identify biological excitation, inhibition, causal flow, or
   graph-learning heterophily; therefore the original BuNN motivation was
   retained but narrowed to a controlled computational test.
3. **Methods:** cohort/QC, connectome construction, folds, baselines, shared
   backbone, operators, densities, gauge-aware diagnostics, endpoints,
   statistics, and run integrity.
4. **Results:** cohort/QC first, then baselines, predictive density curves,
   representation curves, paired effects, site heterogeneity, and robustness.
5. **Discussion:** what survived the original hypothesis, what did not,
   alternative explanations, capacity/measurement limits, multi-site
   instability, and scope of generalization.
6. **Conclusion:** answer only the pre-specified computational question.

### Reproducibility release

Retain code, environment lock, manifests, frozen hashes, split assignments,
run configurations, seeds, prediction tables, warnings, analysis scripts,
figure-generation code, and a machine-readable artifact inventory. Do not
publish restricted/private credentials or unnecessary participant-level
phenotypic information. The AI-use log remains a provenance record even though
this project is no longer being prepared for S.T. Yau.

## Recommended order and realistic schedule

There is no artificial competition deadline, so quality gates control the
schedule. A realistic sequence is:

| Work block | Approximate duration after authorization |
| --- | --- |
| Step 7.5 launch, run, recovery if needed, and integrity audit | Compute-dependent; likely hours to days |
| Step 7.6 analysis implementation and baseline report | 2-3 focused days |
| Step 8 neural implementation and mathematical tests | 4-7 focused days |
| Step 9 pilot, debugging, and configuration freeze | 2-4 focused days |
| Step 10 full neural matrix | Compute-dependent; calibrate from pilot |
| Step 11 statistics and figures | 3-5 focused days |
| Step 12 optional robustness | Only after the core; scope separately |
| Step 13 paper and reproducibility audit | 5-10 focused days |

These are planning ranges, not promises. Failed tests or a scientifically
important defect must extend the schedule rather than be bypassed.

## Scope-control rule

Required work is the full ABIDE-I baseline, controlled four-operator neural
comparison, gauge-aware diagnostics, paired site-aware statistics, and complete
paper/reproducibility package. New diagnoses, atlases, architectures, or broad
biological interpretations are optional future work. They must never replace
an unfinished primary control or be added after seeing a favorable result.
