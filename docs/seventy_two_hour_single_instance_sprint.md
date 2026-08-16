# Seventy-two-hour single-instance extension sprint

Status: **superseded by `docs/seventy_two_hour_feasibility_audit.md` after the
implementation, artifact, released-code, data-link, and access audit. No
extension compute was launched.**

## Fixed resource and objective

The sprint uses the existing AWS `g5.xlarge`: four vCPUs, 16 GiB system RAM,
and one NVIDIA A10G GPU. The goal is to add the strongest audited evidence that
can be completed in 72 hours without weakening the frozen Study 1 protocol or
placing incomplete results in the paper.

The instance is used as a relay, not as several independent machines. We may
run multiple processes on its single GPU only after a measured one-versus-two
worker benchmark demonstrates a worthwhile speedup without out-of-memory
errors, CPU starvation, numerical drift, or excessive checkpoint contention.

## Ranking under the 72-hour constraint

| Rank | Study | Decision |
| --- | --- | --- |
| 1 | Accepted-checkpoint map/topology interventions | mandatory; cheapest direct explanation of whether learned BuNN transport affected predictions |
| 2 | Wang three-arm leakage audit | priority gate because the user specifically wants the apparent-inflation claim tested; full run is conditional on measured smoke runtime |
| 3 | Compact synthetic known-geometry suite | guaranteed fallback if the full Wang audit cannot finish; stronger mechanism evidence than an incomplete Wang run |
| 4 | Five-layer/capacity sensitivity | exclude from this sprint; requires roughly 36-72 compute hours after implementation |
| 5 | ABIDE-II validation | exclude; preprocessing compatibility cannot be responsibly established in this window |

Scientific value and execution order are not identical. Synthetic geometry is
more directly relevant to BuNN mechanism than Wang, but Wang is benchmarked
first because its runtime uncertainty determines whether the requested leakage
audit is possible within the hard deadline.

## Non-negotiable controls

1. Preserve the sealed Study 1 archive and its hash.
2. Use one frozen extension commit and environment lock.
3. Extract only required accepted final checkpoints from the 1 GB archive.
4. Reproduce archived predictions before accepting checkpoint interventions.
5. Keep the Wang released-code, grouped-inner leakage-free, and fixed-epoch
   arms identical except for validation/stopping/checkpoint selection.
6. Keep all three Wang arms for a site in the same worker assignment.
7. Store atomic per-site checkpoints, heartbeats, and SNS completion/failure
   alerts.
8. Audit and hash artifacts before viewing results.
9. Never convert an engineering smoke or incomplete site subset into a
   scientific result.

## Seventy-two-hour schedule

### Hours 0-4: freeze and preflight

- Freeze hypotheses, endpoints, interventions, seeds, Wang arm definitions,
  stopping rules, and promotion criteria.
- Verify AWS health, CUDA, free disk, package-lock hash, SNS, S3/archive access,
  and the frozen Study 1 archive hash.
- Create the selective checkpoint extraction list: accepted learned-BuNN final
  checkpoints only, not all 9,324 training checkpoints.
- Prepare manuscript table/figure placeholders locally while the instance is
  being readied.

**Gate:** no scientific run starts if code, data, environment, or archive hashes
do not match the recorded contract.

### Hours 4-14: implement and smoke checkpoint interventions

- Implement identity-map reset, fixed node-map shuffle, fixed random
  orthogonal maps, degree-aware topology shuffle, and common-permutation
  equivariance.
- Add hand-calculated operator tests and deliberately corrupted tests.
- Load one accepted checkpoint, reproduce its archived predictions within a
  declared tolerance, and run a one-site intervention smoke.
- While AWS performs inference, prepare the Wang CC200 manifest and released
  recipe locally without competing for the instance GPU.

**Gate:** failure to reproduce the archived prediction means the intervention
runner is repaired; it is never allowed to continue on approximately matching
weights.

### Hours 14-22: full intervention audit

- Run all held-out sites and predeclared interventions.
- Use the planned permutation count unless a pre-result precision calculation
  justifies a smaller count.
- Seal, hash, independently audit, analyze, and generate the mechanism figure
  and supplement table.

**Expected outcome by hour 22:** one complete new real-data study suitable for
the main paper.

### Hours 22-36: Wang data and implementation gate

- Obtain and hash the public CC200 time-series derivative.
- Reproduce the paper/code cohort, site membership, padding/time-point logic,
  training-only connectivity-neighbour graph, and `k=5` construction.
- Implement the five-layer temporal cGCN and three stopping/selection arms.
- Verify EdgeConv and temporal aggregation against tiny hand-calculated
  examples and, where possible, the released implementation.
- Build the score-blind per-site runner, resume logic, and auditor before any
  full execution.

**Gate:** if the cohort or released recipe cannot be traced, stop the Wang
study rather than filling undocumented choices with assumptions. Move
immediately to the synthetic fallback.

### Hours 36-44: Wang timing and concurrency smoke

- Run one fixed held-out site through all three arms using one worker.
- Record data time, seconds per epoch, epochs selected, wall time, CPU/RAM,
  GPU utilization, peak VRAM, and checkpoint size.
- Repeat a small score-blind timing segment with two workers on distinct whole
  sites.
- Accept two-worker execution only if it provides at least a 1.3x measured
  throughput improvement, preserves results within the declared tolerance,
  and stays below safe RAM/VRAM limits. Do not assume the earlier Study 1
  three-worker speedup transfers to the Wang model.

### Hour 44 decision gate

Estimate the remaining 16-site runtime from measured site workloads, including
a 20% contingency and eight hours reserved for merge, analysis, manuscript
rebuild, and PDF QA.

- **Route W: full Wang audit.** Take this route only if the conservative
  forecast completes compute by hour 64.
- **Route S: synthetic mechanism fallback.** Take this route if the Wang
  forecast exceeds hour 64, the data/implementation gate fails, or concurrency
  is unstable. Preserve the Wang smoke only as engineering evidence outside
  the Results section.

The decision is made from timing and integrity information before opening
predictive values.

## Route W: full Wang audit, hours 44-72

### Hours 44-64: remaining held-out sites

- Bin-pack whole sites across one or two validated local workers by measured
  runtime and participant count.
- Run all three arms and fixed seeds for each owned site.
- Keep post-site seals, heartbeats, resumeable checkpoints, and SNS alerts.
- Do not start the five-layer ABIDE-I sensitivity or synthetic suite while the
  Wang audit is consuming the instance.

### Hours 64-72: merge, analyze, and rebuild

- Require all 17 sites, hashes, arm coverage, and metric recomputation to pass
  the independent audit.
- Estimate the paired equal-site ordinary-accuracy difference between the
  reconstructed released-code and leakage-free arms. Report balanced accuracy,
  AUROC, per-site differences, selected epochs, and fixed-epoch behavior.
- Call the result apparent inflation under the reconstructed released
  procedure, not the universal inflation of the published 71.6%.
- Rebuild the manuscript and supplement, update the claim ledger and artifact
  manifest, run tests, compile both PDFs, and visually inspect every changed
  page.

## Route S: compact synthetic mechanism study, hours 44-72

### Hours 44-62: validated compact suite

- Run S0 identity geometry, S1 recoverable geometry, S2 wrong topology, S3
  shuffled geometry, and S5 unlearnable participant-specific geometry.
- Compare identity, GCN, trivial bundle, learned-local, learned BuNN, and an
  oracle true-map model using matched backbones and reported parameter counts.
- Use a small predeclared noise and sample-size grid with fixed seeds; do not
  delete negative controls to save time.
- Require the oracle to recover S1 and wrong-map/topology controls to degrade
  before interpreting learned-BuNN performance.

### Hours 62-72: audit, analyze, and rebuild

- Audit every simulation cell and generate one mechanism figure plus complete
  supplementary tables.
- State conditional computational boundary conditions only; do not infer brain
  biology from synthetic success.
- Rebuild and inspect the paper using the same final QA sequence as Route W.

## Expected deliverable

### Reliable 72-hour deliverable

- complete checkpoint intervention study;
- either a complete Wang three-arm audit or a complete compact synthetic suite;
- revised manuscript, supplement, figures, claim ledger, hashes, and PDFs;
- Wang smoke and timing evidence even if Route S is selected.

### Not promised within 72 hours

- both a full Wang audit and a full synthetic suite;
- five-layer/capacity/orthogonality sensitivity training;
- ABIDE-II or raw-MRI preprocessing.

Trying to promise all of those on one A10G would make auditing and writing the
critical path and would encourage incomplete or selectively reported results.
