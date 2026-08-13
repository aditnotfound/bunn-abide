# Verified feasibility audit for one instance and 72 hours

Date checked: 2026-08-13

Status: **quality-first replacement for the provisional 72-hour sprint. No
extension training or inference was launched.**

## Conclusion

One rigorous extension is supportable in 72 hours: the existing-artifact and
accepted-checkpoint intervention audit (E1). A complete Wang leakage audit,
full synthetic mechanism suite, five-layer sensitivity matrix, or ABIDE-II
validation is not currently verified as feasible within the same window.

The paper should gain one complete explanatory study rather than several
under-tested experiments.

## Evidence checked

### Repository readiness

- No intervention, synthetic, Wang/cGCN, or extension runner currently exists
  under `src/`, `scripts/`, `tests/`, or `configs/`.
- The existing neural model exposes learned maps, common-frame conversion,
  relative transports, gauge-aware diagnostics, configurable depth, durable
  checkpoints, and audited site-parallel infrastructure. These are useful
  foundations, but they are not completed extension implementations.
- The sealed Study 1 archive is 1,048,461,070 bytes and contains 9,324 fit
  checkpoints.
- Exactly 360 accepted final learned-BuNN checkpoints are required for the
  intervention study. Together they are only 45,034,440 bytes, approximately
  125 kB each, so selective extraction is not a storage or transfer bottleneck.
- The archive inventory completed locally in approximately seven seconds.

### Measured Study 1 runtime

- The 360 learned-BuNN outer-final fits consumed 3,890.37 aggregate seconds,
  or 1.08 fit-hours, including training rather than inference alone.
- Their recorded peak GPU memory was 290,552,832 bytes. Learned-BuNN inner
  tuning peaked at 405,795,840 bytes.
- The A10G therefore has ample memory for checkpoint inference. The unresolved
  cost is the new perturbation loop, repeated permutations, implementation,
  testing, audit, and interpretation—not VRAM.
- These measurements support an intervention smoke but do not prove the full
  perturbation runtime. A one-site timing run remains mandatory before launch.

### Wang reproduction blockers

- No CC200 Wang data artifact or cGCN implementation is present locally.
- The released repository specifies Keras 2.1.5, TensorFlow 1.4.1, h5py 2.8.0,
  Nilearn 0.5.0, and NumPy 1.15.4. A modern implementation therefore requires
  a numerical port or a carefully isolated legacy environment.
- The released leave-one-site-out script expects
  `ABIDE_I_leave_one_site_out.h5` with 1,057 participants and 315-by-200 padded
  time-series inputs.
- The public Google Drive file ID named in that script returned HTTP 404 during
  this audit. The repository README also displays its leave-one-site-out data
  link as struck through.
- Reconstructing the artifact from public CC200 derivatives would require
  tracing cohort eligibility, time-point handling, preprocessing, site labels,
  and subject order before model work begins. Filling gaps by assumption would
  invalidate a claimed replication.
- The released runner uses the held-out site's labels for validation loss,
  learning-rate scheduling, early stopping, and checkpoint selection. The
  leakage hypothesis is real and testable, but the experiment is not ready to
  time or run.

### Current AWS access

- TCP port 22 accepted a connection at the time of the audit, but the instance
  reset the SSH session before key exchange. Live CUDA, RAM, disk, and
  environment checks could therefore not be refreshed.
- SSH access must be restored before the 72-hour execution clock begins.

## Feasibility decision by study

| Study | Data ready | Code ready | Runtime measured | 72-hour decision |
| --- | --- | --- | --- | --- |
| E1.1 existing-artifact diagnostics | yes | mostly existing analysis infrastructure | no new training required | include |
| E1.2 checkpoint map/topology interventions | yes; 360 accepted checkpoints | not yet | one-site smoke still required | include, gated |
| E2 synthetic known geometry | generated data must be built | no | no | defer |
| E3 five-layer/capacity matrix | yes for ABIDE-I inputs | partial depth primitive only | prior estimate 36-72 compute hours is not a smoke measurement | defer |
| E4 Wang leakage audit | released LOSO artifact currently unavailable | no modern validated port | no | defer |
| E5 ABIDE-II | no compatible frozen derivative established | no | no | defer |

## Revised 72-hour execution plan

### Hours 0-6: restore and freeze

1. Restore SSH and verify CUDA, RAM, disk, SNS, package lock, code commit, and
   archive hashes.
2. Freeze the E1 estimands, intervention definitions, seeds, permutation count,
   numerical tolerances, and promotion rules.
3. Selectively extract the 360 final learned-BuNN checkpoints.
4. Reproduce archived predictions from representative checkpoints before
   modifying any maps or topology.

### Hours 6-18: implement the intervention runner

1. Add identity-map reset, fixed node-map shuffle, fixed random orthogonal
   maps, degree-aware topology shuffle, and common-permutation equivariance.
2. Ensure one checkpoint is loaded once and reused across its intervention
   permutations to avoid needless I/O.
3. Add hand-calculated operator tests, fixed-seed determinism tests, corrupted
   artifact tests, resume tests, and result-embargo/audit checks.
4. Implement atomic per-site artifacts, heartbeat, SNS alerts, and independent
   recomputation.

### Hours 18-24: measured one-site gate

1. Run one complete held-out site with every intervention and the planned
   permutation count while keeping predictive results sealed.
2. Measure wall time, checkpoint-load time, inference time, CPU/RAM, peak VRAM,
   and output size.
3. Project the 18-site runtime with a 30% contingency.
4. Authorize the full run only if compute, audit, analysis, and at least 12
   hours of manuscript QA fit within the remaining 48 hours.

The permutation count is not reduced merely to meet the deadline. It may be
changed only through a pre-result precision calculation documented before
unsealing.

### Hours 24-50: full E1 execution and existing-artifact audit

- Run held-out sites with durable post-site seals and resume support. Test one
  versus two local workers only if the smoke shows the job is throughput-bound;
  accept two workers only after numerical equivalence and resource-safety
  checks.
- In parallel off-instance, analyze already unsealed Study 1 training curves,
  tuning choices, warnings, train/inner-validation gaps, runtime, parameter
  counts, density behavior, site sensitivity, and seed sensitivity.
- Prepare fixed figure/table builders without opening E1 predictive results.

### Hours 50-60: audit and analysis

1. Require all sites, interventions, seeds, permutations, hashes, and metric
   recomputations to pass the independent audit.
2. Unseal once through the frozen analysis script.
3. Report prediction sensitivity, gauge-aware representation changes, per-site
   variation, and uncertainty for every intervention family.
4. Update the competing-explanation table. Sensitivity to a component does not
   establish biological geometry; insensitivity does not prove geometry absent.

### Hours 60-72: paper integration and QA

1. Add one concise mechanism section and one restrained figure to the main
   paper; put complete interventions, sites, seeds, and permutations in the
   supplement.
2. Update the abstract and conclusion only if the audited result supports a new
   statement.
3. Rebuild manuscript inputs, tables, claim ledger, artifact manifest,
   supplement, and PDFs.
4. Run tests, citation checks, hash checks, and page-by-page visual inspection.

## Output promised after 72 hours

- one complete, auditable real-data mechanism study tied directly to the
  original BuNN question;
- an evidence table separating supported, weakened, and unresolved explanations
  of the frozen result;
- revised manuscript, supplement, figures, reproducibility records, and PDFs;
- no Wang, synthetic, depth, or ABIDE-II result presented as completed.

## Work after this sprint

- Synthetic known-geometry experiments should receive their own approximately
  three-to-five-day implementation, control-validation, compute, and analysis
  window.
- Wang requires recovery or reconstruction of the CC200 LOSO artifact, a
  validated modern port, a one-site three-arm smoke, and only then a full
  runtime estimate. On one instance, this is more plausibly a separate
  week-plus study than a 72-hour add-on.
- Five-layer sensitivity proceeds only if E1 leaves architecture restriction
  as a live explanation.
