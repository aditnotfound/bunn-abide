# Four-day parallel extension sprint

Status: **execution plan only; no extension result has been produced.**

## Objective

Use a maximum of four AWS GPU instances over 96 hours to complete the most
informative extensions that can strengthen the existing frozen ABIDE-I paper
without changing or relabelling Study 1. Parallel execution changes scheduling
only. Every run retains fixed seeds, immutable contracts, atomic per-site
artifacts, independent audits, and result unsealing only after integrity checks.

## Scientific priority

| Rank | Study | Four-day decision |
| --- | --- | --- |
| 1 | Accepted-checkpoint map and topology interventions | Must finish; most direct real-data test of whether learned BuNN transport affected predictions |
| 2 | Compact synthetic known-geometry suite | Must finish; positive and negative controls test whether the implementation can exploit bundle geometry under identifiable conditions |
| 3 | Wang three-arm leakage audit | Must attempt and receives all freed instances after its smoke; directly measures apparent inflation under the reconstructed released selection procedure |
| 4 | Five-layer/capacity/orthogonality sensitivity | Stretch study; launch only when the Wang completion forecast is safe |
| 5 | ABIDE-II external validation | Excluded from the four-day sprint; preprocessing comparability cannot be responsibly established on this deadline |

## Recommended AWS layout

Use the same validated Ubuntu/CUDA environment and one frozen code commit on
every machine. Do not develop separate versions of the code on separate
instances.

| Node | Suggested type | Initial role | Relay role |
| --- | --- | --- | --- |
| A | existing `g5.xlarge` | checkpoint extraction and intervention audit | Wang site shard A |
| B | new `g5.xlarge` | synthetic suite | Wang site shard B |
| C | new `g5.xlarge` | synthetic replication shard or idle until its contract is ready | Wang site shard C |
| D | new `g5.2xlarge` | Wang data gate, model tests, and one-site three-arm smoke | Wang coordinator and site shard D |

The `g5.2xlarge` has the same single A10G GPU as `g5.xlarge`; its reason here is
the extra CPU and RAM for CC200 time-series loading and graph construction, not
additional VRAM. If the on-demand G-instance vCPU quota is below 20, use three
`g5.xlarge` nodes instead and split Wang sites 6/6/5. Do not substitute one
larger single-GPU G5 instance expecting extra GPU parallelism.

Use on-demand capacity for the Wang audit and final merge. Synthetic jobs may
use Spot only after checkpoint/resume has passed an intentional-interruption
test. Set the instance-initiated shutdown behavior to stop, preserve SNS
terminal alerts, and stop each node after its artifacts are uploaded and
hash-verified.

## Shared deployment contract

Before scientific compute:

1. create one extension branch and record the frozen Study 1 archive hash;
2. build and test code locally, then deploy the same commit to every node;
3. store the 1 GB sealed Study 1 archive and public Wang inputs in private S3;
4. give every run a unique run ID and node/site ownership manifest;
5. verify CUDA, package-lock hash, data hashes, clock, disk, SNS, and S3 access;
6. never allow two nodes to write the same site directory;
7. merge only sealed site artifacts with matching hashes;
8. keep all three Wang arms for a held-out site on the same node;
9. compare a small sequential/parallel control to a declared numerical
   tolerance before accepting distributed results.

## 96-hour schedule

### Hours 0-6: freeze and provision

- Freeze the extension registry, endpoints, seeds, intervention definitions,
  Wang arms, and promotion rules.
- Confirm the G-instance vCPU quota before launching new nodes.
- Create a reusable image or bootstrap script from the validated environment.
- Upload hash-bound input bundles and create node ownership manifests.
- Start node A immediately; start other nodes only when their code/data contract
  is ready to avoid paid idle time.

**Gate:** all nodes report the same code, environment, and immutable input
hashes. Failure blocks scientific execution.

### Hours 6-18: intervention implementation and Wang data gate

- Extract only accepted final learned-BuNN checkpoints from the sealed archive,
  rather than all 9,324 fit checkpoints.
- Implement identity-map reset, fixed node-map shuffle, fixed random orthogonal
  maps, degree-aware topology shuffle, and common-permutation equivariance.
- Add hand-calculated and corruption tests, then run a one-site intervention
  smoke on node A.
- In parallel, reproduce the Wang CC200 cohort/time-series artifact on node D,
  record subject/site counts and hashes, and compare them with the paper/code.
- Begin the synthetic generator and oracle-map tests locally for deployment to
  node B.

**Gate:** accepted checkpoints reproduce archived predictions before any
intervention; the Wang study cannot train until its data artifact is traced.

### Hours 18-30: first full jobs

- Run all intervention permutations on node A. Store every permutation seed;
  use the predeclared precision check before reducing the planned 100
  permutations.
- Run synthetic S0/S1 positive controls and S2/S3/S5 negative controls on node
  B. Node C may take deterministic simulation cells only after sequential and
  distributed equivalence passes.
- Implement the Wang five-layer temporal cGCN and the released-code stopping
  behavior without yet running outer results.

**Gate:** an oracle-map synthetic model must recover S1 while wrong-map and
wrong-topology controls fail appropriately. A benchmark rewarding BuNN in all
families is rejected and repaired, not reported.

### Hours 30-48: audit early studies and smoke Wang

- Seal, merge, audit, and analyze the intervention study.
- Complete the compact synthetic noise/sample-size sweep and audit it.
- Run numerical EdgeConv tests and a one-site, fixed-seed Wang smoke containing
  released-code, grouped-inner leakage-free, and fixed-epoch arms on node D.
- Measure data time, epoch time, peak RAM/VRAM, selected epochs, and projected
  site runtime. Replace every provisional Wang estimate with this measurement.

**Gate at hour 48:** launch the full Wang audit only if the smoke is numerically
valid, all three arms complete, and the measured forecast can finish by hour
84 with available nodes. Otherwise finish the paper with Studies 1, E1, and E2
and label Wang as unfinished engineering work outside the results.

### Hours 48-84: distributed Wang audit

- Bin-pack the 17 held-out sites across four nodes by measured workload, not by
  site count. A three-node fallback uses 6/6/5 sites adjusted for participant
  count.
- Each node runs all three arms and fixed seeds for each site it owns.
- Use separate node directories, heartbeats, post-site seals, resumeable
  checkpoints, and terminal SNS alerts.
- When a node finishes early, it may take only an unstarted, reassigned whole
  site recorded by a new ownership manifest. Never split the three arms of an
  already-started site.
- Keep five-layer ABIDE-I sensitivity queued. Launch it only if the measured
  Wang forecast leaves one node unnecessary and the full sensitivity can seal
  by hour 84.

**Gate:** the coordinator refuses to merge if any site is missing, duplicated,
unsealed, or hash-invalid. No result values are read before the independent
audit passes.

### Hours 84-96: analysis and paper rebuild

- Compute the paired equal-site ordinary-accuracy difference between the
  reconstructed released-code and leakage-free Wang arms, with per-site
  uncertainty and secondary balanced accuracy/AUROC.
- Call the contrast apparent inflation under the reconstructed procedure, not
  the universal inflation of the published 71.6%.
- Add one real-data mechanism figure, one synthetic boundary-condition figure,
  and a compact Wang protocol-sensitivity table. Put full site/seed tables in
  the supplement.
- Rebuild manuscript inputs, claim ledger, tables, supplement, and both PDFs.
- Run repository tests, artifact-hash checks, citation checks, and page-by-page
  visual QA.

## Four-day completion definitions

### Minimum defensible completion

- checkpoint intervention audit complete;
- compact synthetic suite complete with valid controls;
- Wang three-arm implementation and measured smoke complete;
- existing manuscript revised only with completed, audited evidence.

### Target completion

- minimum completion plus the full 17-site Wang three-arm audit;
- updated main paper, supplement, reproducibility manifest, and final PDFs.

### Stretch completion

- target completion plus a restricted five-layer/depth sensitivity matrix.

ABIDE-II and raw-MRI reprocessing are not stretch items for this sprint. They
require a separate compatibility gate and would displace more direct evidence.

## Expected elapsed time

- E1 interventions: approximately 12-24 hours including implementation,
  inference, and audit.
- compact E2 synthetic suite: approximately 24-42 hours including generator
  validation and analysis.
- Wang data/model/smoke gate: approximately 30-48 hours from sprint start.
- full Wang audit after a successful smoke: provisionally 12-36 additional
  wall-hours with four site-sharded nodes; this is a scaling prediction, not a
  measured promise.
- manuscript rebuild and QA: 8-12 hours because a complete manuscript builder
  already exists.

The target can fit within 96 hours only if the Wang data gate succeeds on day
one, the hour-48 smoke passes, and distributed runtime scales adequately. The
minimum completion is the reliable four-day deliverable; the full Wang result
is conditional until its smoke supplies a real runtime.
