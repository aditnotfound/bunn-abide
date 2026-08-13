# Post-hoc mechanism and external-validation extension plan

Status: **implementation plan only; no extension result has been produced.**

The completed ABIDE-I operator audit remains Study 1. Its data, folds, models,
analysis, and conclusion are immutable. The extension studies below will have
new configuration files, run identifiers, audit certificates, and manuscript
labels. They may explain or qualify Study 1, but they may not overwrite it.

## 1. Evidence language

Every statement in the extension will be assigned one of four labels:

- **Verified fact:** directly supported by a versioned local artifact, official
  dataset documentation, a paper, or released source code.
- **Pre-specified prediction:** an outcome expected under a named hypothesis;
  it is not presented as something that has happened.
- **Observed result:** calculated only after the run passes its score-blind
  integrity audit and the analysis code is frozen.
- **Interpretation:** the narrowest explanation consistent with the result and
  its controls. Alternative explanations and unresolved uncertainty remain
  explicit.

The words *causes*, *proves*, *biological geometry*, *excitation*,
*inhibition*, and *clinical diagnosis* are prohibited unless a future design
actually identifies those claims. Functional-connectivity edges remain
statistical associations.

## 2. Questions the extension must answer

The extension has four distinct questions. They must not be collapsed into one
leaderboard.

1. **Mechanism:** Can bundle transport recover information when node features
   genuinely occupy different known coordinate frames?
2. **ABIDE-I failure analysis:** Why was no BuNN advantage detected in the
   completed globally informed connectome setting?
3. **Architecture sensitivity:** Does the conclusion change when depth,
   residual/skip structure, temporal input, or neighbourhood construction is
   changed under leakage-free evaluation?
4. **Transportability:** Do frozen ABIDE-I conclusions survive a genuinely
   separate ABIDE-II evaluation after compatible preprocessing?

## 3. Phase E0 - preserve Study 1 and freeze the extension contract

Before implementation:

1. Record the current Study 1 commit, evidence-package hashes, paper hashes,
   cohort manifest, and result classification.
2. Create a separate extension namespace under `configs/extensions/`,
   `outputs/extensions/`, and `docs/extensions/`.
3. Write one machine-readable registry listing every planned experiment,
   primary endpoint, secondary endpoints, seeds, comparison family, and stop
   rule.
4. Preserve the existing 754-participant AAL-116 cohort, 18 outer sites, four
   inner site-grouped folds, density grid, and archived Study 1 outputs.
5. Mark all ABIDE-I follow-ups as **post-hoc mechanistic/architecture studies**.
6. Mark ABIDE-II as **prospective external evaluation** only after its data and
   preprocessing contract pass the feasibility gate.

Gate E0 passes only when a clean rebuild reproduces Study 1's aggregate
artifacts and the extension registry contains no reference to a held-out
extension result.

## 4. Phase E1 - audit competing explanations of the completed result

We will not begin with a preferred story. We will test the following competing
explanations.

| Hypothesis | Evidence that would support it | Evidence that would weaken it |
| --- | --- | --- |
| H1: globally informed connectivity-row features already contain most usable graph information | identity/elastic-net remain strong; topology perturbations change little; local-feature synthetic conditions show a larger transport benefit than global-feature conditions | correct topology and transport consistently improve held-out prediction over identity and elastic net |
| H2: aggregation removes task-relevant information | accuracy and train-only linear-probe information decrease with propagation/density; decline co-occurs with gauge-aware contraction | propagated representations retain or improve label information without predictive degradation |
| H3: BuNN learned no stable/useful transport | maps vary strongly across seeds/folds, remain close to identity, or identity-reset/shuffled-map interventions barely change predictions | maps are stable and replacing/shuffling them reliably harms prediction |
| H4: extra BuNN capacity is unhelpful or overfits | learned-local and learned-BuNN have larger train/validation gaps without held-out gain | added capacity improves inner and outer sites consistently |
| H5: optimization limited the neural models | selected candidates sit on search boundaries, training is under-converged/unstable, or a pre-specified broader search improves inner-site and outer-site performance | convergence, gradient, seed, and search-boundary checks are healthy and added search does not help |
| H6: the conclusion is site/data-regime dependent | effect signs vary with held-out site, motion/demographic shift, or training-set size | effects remain stable across sites and learning-curve points |
| H7: two propagation layers were an unfavorable architecture choice | a matched deeper model improves held-out performance and changes the operator ordering reproducibly | depth/skip variants preserve the original ordering or worsen all operators |

### E1.1 Existing-artifact audit

First inspect only already accepted Study 1 artifacts:

- training and validation loss curves;
- selected learning-rate/weight-decay candidates and whether selections lie on
  the tuning-grid boundary;
- selected epoch distributions;
- train-versus-inner-validation gaps;
- seed variance and site influence;
- density-response curves;
- identity, learned-local, GCN, trivial-bundle, and learned-BuNN contrasts;
- gauge-aware effective rank, dispersion, cosine similarity, and edge
  transport distance;
- parameter counts, runtime, and failure/warning records.

This phase can diagnose inconsistencies and generate testable alternatives. It
cannot retroactively prove a cause.

### E1.2 Held-out intervention audit

First inventory the accepted archive and verify that the required trained
checkpoints and map-producing state were preserved. If they were not, replay
the exact frozen fits and require equality with the archived held-out
predictions within a pre-declared numerical tolerance before accepting any
intervention. Then run inference-only interventions on each held-out site:

1. replace learned node maps with identity maps;
2. shuffle learned maps across nodes with a fixed recorded permutation;
3. replace maps with fixed random orthogonal maps;
4. shuffle graph topology while preserving node degree as closely as the
   implementation permits;
5. apply one common node permutation to feature rows/columns, adjacency,
   node-index/positional inputs, maps, and diagnostic ordering as an
   equivariance sanity check;
6. repeat the common-frame and invariant diagnostics after each intervention.

Interventions use the same checkpoint and held-out participants. They measure
prediction sensitivity to a component; they do not show that the component is
biological. Any random intervention uses at least 100 stored permutations per
checkpoint, or a smaller number only if a precision analysis conducted before
unblinding justifies it.

### E1.3 Information and learning-curve diagnostics

- Fit train-only linear probes to encoder and post-layer common-frame
  representations. All probe tuning remains inside the active training sites.
- Measure whether label information and site information change after
  propagation. Site decodability is a warning diagnostic, not proof of
  confounding.
- Run pre-specified training-fraction curves for the strongest linear model,
  GCN, and BuNN using site-grouped subsamples. Use identical fractions and
  resampling seeds. Do not read the outer site during selection.

E1 ends with an explanation table stating which hypotheses were supported,
weakened, or left unresolved. More than one explanation may remain viable.

## 5. Phase E2 - synthetic ground-truth geometry

This is the only study in which the true coordinate frames and transport maps
are known. It tests a computational mechanism, not brain biology.

### E2.1 Data-generating families

Use the same graph sizes and class balance across all families.

- **S0, no geometry needed:** every node uses the common identity frame.
- **S1, fixed recoverable geometry:** class-relevant latent signals are placed
  in a canonical frame and rotated by fixed node-specific orthogonal maps.
- **S2, incorrect topology:** use S1 features with a rewired graph.
- **S3, shuffled geometry:** use S1 but assign correct maps to the wrong nodes.
- **S4, transport-noise sweep:** gradually perturb the true maps from correct
  to random.
- **S5, unlearnable subject-specific frames:** rotate every participant with
  new maps that cannot be inferred from fixed node identity alone.
- **S6, local versus global features:** expose either local node signals or a
  global connectivity-profile analogue.

Positive control: a model given the true maps must recover the S1 mechanism.
Negative controls: S0, S2, S3, and S5 must prevent a synthetic benchmark from
rewarding BuNN merely because it has more parameters.

### E2.2 Models and endpoints

Compare identity, GCN, trivial-bundle diffusion, fixed-random transport,
learned-local, learned BuNN, and oracle true-map bundle diffusion. Use matched
encoders/readouts and report exact parameter counts.

Primary mechanistic endpoint: held-out balanced-accuracy improvement of learned
BuNN over GCN in S1 minus the same improvement in S0.

Required supporting endpoints:

- recovery of the oracle performance gap;
- prediction degradation under map/topology corruption;
- gauge-invariant map or transport error against ground truth;
- performance across transport-noise, density, sample-size, and signal-to-noise
  sweeps;
- calibration and seed uncertainty.

The useful result is conditional: bundle transport helps when alignment is
necessary and learnable. If it also wins in S0 or survives wrong-map controls,
the benchmark is probably measuring capacity or optimization rather than the
intended mechanism.

## 6. Phase E3 - matched ABIDE-I depth and capacity sensitivity

This phase keeps the accepted AAL-116 participants, features, outer/inner
splits, metrics, and graph construction. It changes architecture only.

### E3.1 Primary depth matrix

- depths: 2 and 5 propagation blocks;
- structures: plain and residual/skip-connected;
- operators: identity, GCN, trivial bundle, learned-local, learned BuNN;
- constant hidden width 32 for the primary depth comparison;
- identical encoder, activation, dropout policy, pooling, classifier, loss,
  optimizer family, tuning budget, outer folds, seeds, and stopping rule;
- exact parameter counts, FLOPs/step, peak memory, and runtime reported.

The two-layer Study 1 cells are referenced from the frozen archive when the
implementation is identical. Any code-semantic change requires a fresh run and
an equivalence test before comparison.

### E3.2 Capacity controls

Depth and parameter count are different questions. Therefore:

1. compare constant-width depth 2 versus depth 5;
2. compare parameter-budget-matched shallow and deep variants;
3. run the Wang width schedule `[8, 16, 32, 64, 128]` only as a separately
   labelled capacity sensitivity;
4. retain learned-local at each BuNN depth so transport is not credited for its
   map-generator capacity;
5. include skip/no-skip pairs at the same depth.

Primary endpoint: change in equal-site held-out balanced accuracy from depth 2
to depth 5 for each operator, followed by the operator-by-depth interaction.
The extension does not declare BuNN successful merely because one deeper BuNN
cell beats the old two-layer GCN.

### E3.3 Original-outline operator ablations

The first outline explicitly proposed ablating orthogonality, bundle/stalk
dimension, sign handling, and learned maps. The map interventions in E1 cover
the last item. If E1 and E2 leave an architectural explanation unresolved, run
the remaining items as a separately labelled post-hoc sensitivity family:

1. hold hidden width fixed while comparing bundle dimensions 1, 2, and 4,
   adjusting the number of bundles/channels transparently;
2. compare orthogonal transport with a parameter-matched unconstrained linear
   transport control, reporting map norms and condition numbers so scaling or
   numerical instability is not mistaken for useful geometry;
3. compare the frozen positive-edge graph with explicitly computational
   negative-association variants, such as separate positive/negative channels
   or signed weights derived from the observed correlation matrix;
4. evaluate global-signal-regression sensitivity only if a provenance-matched
   derivative can be obtained and frozen before predictive output is read.

Negative correlations are never called node-label heterophily, excitation,
inhibition, or antagonistic neural connections. These ablations can show that
the operator is sensitive to an edge-sign representation; they cannot identify
a biological signed geometry.

## 7. Phase E4 - leakage-free Wang cGCN study

The published/released Wang model is a different experiment, not simply a
five-layer GCN. The verified recipe uses CC200 raw ROI time series, a
training-derived directed `k=5` connectivity-neighbour graph, EdgeConv features
`[x_i, x_j-x_i]`, max-neighbour aggregation, five convolution widths
`[8,16,32,64,128]`, skip fusion into the fifth layer, frame-level prediction,
and temporal averaging. The released leave-one-site-out code uses 315-frame
zero-padded/truncated tensors, Adam at `1e-4`, batch size 4, and 50 epochs.

The released runner also uses the held-out site as Keras validation data for
learning-rate scheduling, early stopping, and checkpoint selection. We will
not copy that leakage into the accepted benchmark.

### E4.1 Data-reproduction gate

1. obtain the public CC200 derivative named by the authors;
2. hash it and reconstruct the reported 1,057-participant/17-site eligibility
   as far as the public artifact permits;
3. verify frame counts, zero-padding, labels, site names, graph construction,
   and class totals;
4. document every mismatch as verified, unresolved, or corrected;
5. stop if the released data cannot be traced sufficiently for a meaningful
   reproduction.

### E4.2 Model ladder

Run, in order:

1. time-series model with no neighbours;
2. random `k=5` graph;
3. training-derived connectivity `k=5` graph;
4. two-layer EdgeConv;
5. five-layer EdgeConv without skip fusion;
6. five-layer EdgeConv with Wang skip fusion;
7. zero-padded temporal mean exactly matching the release;
8. mask-aware temporal mean as a separately labelled correction;
9. only after the cGCN ladder works, a matched spatiotemporal BuNN variant.

All graph construction, scaling, tuning, stopping, and checkpoint selection
use outer-training sites and grouped inner validation only. Each held-out site
is evaluated once after the model is frozen.

Report both ordinary accuracy for historical comparison with 71.6% and the
equal-site balanced accuracy/AUROC family used in Study 1. The historical value
is not treated as a clean target because its released selection procedure uses
outer-test labels.

### E4.3 Direct outer-test-leakage audit

After the data and architecture reproduction gates pass, run three matched
arms for every outer site and fixed initialization seed:

1. **Released-code arm:** reproduce the public runner's use of the outer site
   for learning-rate scheduling, early stopping, and best-checkpoint selection.
   This arm is deliberately contaminated and is never reported as an unbiased
   generalization estimate.
2. **Leakage-free arm:** choose learning-rate behavior, stopping epoch, and
   checkpoint using grouped inner training sites, refit on all outer-training
   participants for the selected epoch, and evaluate the outer site once.
3. **Fixed-epoch arm:** train on all outer-training participants for a
   pre-registered epoch count with no validation feedback from the outer site,
   then evaluate it once.

Graph construction, preprocessing, architecture, batch size, optimizer,
initialization seeds, and outer folds are identical across arms. The primary
audit estimand is the paired equal-site ordinary-accuracy difference between
the released-code and leakage-free arms. Balanced accuracy, AUROC, selected
epoch, learning-rate trajectory, and per-site differences are secondary.

The difference is called **apparent inflation under the reconstructed released
procedure**, not the universal amount by which the published 71.6% is inflated.
That stronger claim would require exact equality of the authors' data artifact,
software behavior, randomness, and all undocumented choices.

## 8. Phase E5 - ABIDE-II prospective external evaluation

Official ABIDE-II documentation lists 1,114 datasets from 19 sites, including
521 ASD and 593 control datasets. It exposes phenotypic files, quality metrics,
and site-level imaging downloads. A ready-made derivative matching Study 1's
C-PAC filtered-no-global-signal-regression AAL-116 time series has not yet been
verified.

### E5.1 Feasibility gate

Before downloading or preprocessing the full cohort:

1. create a phenotypic/site/scan manifest from official files;
2. identify longitudinal scans, repeated collections, returning sites, and any
   plausible ABIDE-I overlap;
3. select 10 participants across at least three sites without reading outcomes;
4. run a versioned container that reproduces the Study 1 preprocessing choices
   as closely as technically possible;
5. extract AAL-116 ROI time series and apply the same zero-variance, finite,
   motion/QC, metadata, and class/site eligibility checks;
6. compare output dimensions, scan-length handling, correlation distributions,
   and QC definitions with Study 1;
7. have an independent script verify the manifest and preprocessing provenance.

If the old preprocessing cannot be reproduced defensibly, choose one of two
transparent paths before labels are modeled:

- stop ABIDE-II and report preprocessing incompatibility; or
- reprocess both ABIDE-I and ABIDE-II through one new frozen pipeline and call
  it a separate cross-release study, not a direct validation of Study 1.

### E5.2 Locked external test

After feasibility passes, freeze the ABIDE-II cohort without predictive model
output. Train selected models on all eligible ABIDE-I participants using only
ABIDE-I decisions and hyperparameters. At minimum test:

- connectome elastic net;
- the already identified Study 1 GCN at 1% density, locked before any ABIDE-II
  predictive output;
- learned BuNN at the same 1% density so topology is held fixed;
- identity/learned-local controls needed to interpret propagation.

Evaluate ABIDE-II once. Report pooled and equal-site balanced accuracy, AUROC,
sensitivity, specificity, calibration, recurring-site and new-site summaries,
and uncertainty clustered by ABIDE-II site. Recurring versus new sites are
descriptive strata, not independently randomized groups.

No ABIDE-II result may choose preprocessing, thresholds, hyperparameters,
epochs, models, or manuscript headline wording.

## 9. Analysis rules for explaining model differences

For every comparison, the final analysis must answer five questions:

1. **Did the difference occur?** Give the paired site-level estimate and
   interval, not only a rank or p-value.
2. **Was the comparison fair?** Report cohort, atlas, input, topology, depth,
   capacity, tuning, seeds, selection procedure, and metric.
3. **Which component changed?** Use ablations that change one component at a
   time whenever possible.
4. **Does a mechanism diagnostic move with prediction?** Say *co-occurs with*,
   not *causes*, unless the controlled synthetic intervention identifies it.
5. **Which alternatives remain?** List explanations not ruled out by the
   experiment.

Primary uncertainty remains equal-site paired inference because sites, not
participants pooled across sites, define generalization. Secondary comparison
families use a pre-declared multiplicity correction. Effect sizes and intervals
remain visible even when adjusted tests do not reject.

## 10. Result-to-explanation decision table

| Observed pattern | Permitted interpretation | Not permitted |
| --- | --- | --- |
| BuNN helps only in recoverable synthetic geometry and not S0/S5 | transport can help when alignment is necessary and learnable | ABIDE contains the same geometry |
| Identity-reset or node-shuffle barely affects ABIDE predictions | the learned maps were not materially used by those checkpoints | bundle geometry does not exist in brains |
| Map perturbation hurts but BuNN still trails GCN | transport affects computation but did not yield a held-out predictive advantage | transport caused the classification error |
| Five-layer models improve all operators similarly | depth/capacity/skip structure mattered more than operator choice in this study | Wang's result was replicated |
| Five-layer BuNN improves more than matched controls | BuNN is architecture-sensitive under this AAL protocol | BuNN is generally superior |
| Wang cGCN improves only with the connectivity graph | time-series neighbourhood structure contributed under the leakage-free cGCN protocol | the 71.6% value is directly confirmed unless protocol and metric also match |
| ABIDE-II performance drops for every model | cross-release/preprocessing/site shift limits transportability | all models are intrinsically poor |
| Elastic net remains strongest on ABIDE-II | the simple connectome model transported better under the frozen external test | GNNs never help neuroimaging |

## 11. Engineering and audit gates

Each phase follows the established pattern:

1. implementation;
2. hand-calculated unit tests where possible;
3. synthetic corruption tests that must fail;
4. local score-blind smoke test;
5. one-site AWS runtime/memory measurement;
6. immutable configuration and code hashes;
7. resumable managed launch with start/terminal alerts;
8. score-blind completion audit;
9. freeze analysis code on synthetic fixtures;
10. one acknowledged unblinding;
11. hash-bound tables, figures, and manuscript inputs.

AWS runtime is not estimated from intuition. It is measured after the one-site
smoke and multiplied by the exact fit count, worker count, and observed
utilization. Reduced smokes are engineering tests only and never replace the
scientific grid.

## 12. Recommended execution order

1. E0 evidence freeze and extension registry.
2. E1 existing-artifact audit and checkpoint perturbation implementation.
3. E2 synthetic geometry suite.
4. E3 matched depth/capacity study.
5. E4 Wang data-reproduction and model-ladder gate.
6. E5 ABIDE-II 10-participant preprocessing feasibility gate.
7. Review E1-E5 engineering evidence and freeze only the defensible full runs.
8. Launch accepted full runs, audit them, and analyze once.
9. Rewrite the paper as Study 1 plus explicitly labelled extension studies.

The paper is stronger even if BuNN still loses, provided the controls show
*where* its inductive bias helps, *where* it is irrelevant, and which competing
explanations remain unresolved. A favorable score without those controls is
less informative than a well-explained null result.

## 13. Paper-inclusion rule

The final manuscript must remain one coherent paper rather than a catalogue of
every completed run.

- **Main paper:** Study 1, the strongest E1 map/topology interventions, the E2
  synthetic mechanism result, and E5 external validation if preprocessing is
  genuinely compatible.
- **Supplement:** complete E1 diagnostics, learning curves, optimization
  checks, E3 depth/capacity/operator sensitivities, and all per-site/seed
  results.
- **Separate benchmark or companion paper:** the faithful CC200 temporal Wang
  cGCN study, unless it can be summarized briefly without implying that it is
  an ablation of the AAL connectome-row experiment.
- **Methods/reproducibility only:** E0 registry details, managed-run machinery,
  hashes, alerts, and score-blind audit internals.

An experiment is included as an explanation only when it distinguishes at
least two competing hypotheses. A result that merely adds another model score
is reported as architecture benchmarking, not as an explanation of Study 1.
