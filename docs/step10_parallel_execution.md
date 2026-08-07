# Step 10: Site-Parallel Execution Extension

Status: **implemented and validated in an isolated AWS development workspace.
Unit, real-data, recovery, merge, score-blind audit, exact equivalence, and
speed gates passed. A break-even/restart decision remains; the active
single-worker run has not been stopped or modified.**

## Scope

This extension changes scheduling only. It does not change the ABIDE-I cohort,
outer or inner splits, operator matrix, hyperparameter candidates, seeds,
training rules, diagnostic definitions, endpoints, or analysis contract.

Three subprocesses share one CUDA device. Each receives a deterministic,
non-overlapping set of complete outer sites and one CPU thread. Every worker
uses the existing nested neural runner, fit checkpoints, and atomic site seal
inside an isolated directory. Workers cannot send run-level alerts or write
the canonical aggregate.

## Coordination and recovery

The coordinator owns the immutable parallel metadata, root status, STARTED and
terminal SNS alerts, worker lifecycle, and final merge. A failed worker does
not write a canonical result and does not prevent the other workers from
finishing their assigned sites. Explicit resume restarts each worker against
its immutable contract; verified completed sites are skipped and unsealed
sites resume from their own epoch checkpoints.

## Merge and audit

Canonical merging begins only after all workers exit successfully. Every
source site must be complete, hash-valid, uniquely owned, and copied into the
frozen outer-fold order. The parallel manifest records worker ownership and
source hashes. The extended score-blind auditor verifies worker metadata,
root hashes, exact site coverage, source-to-canonical byte identity, and all
existing prediction/metric/tuning/diagnostic contracts without reporting any
held-out value.

## Acceptance gates

1. Complete AWS unit suite passes.
2. Fresh three-worker, three-site real-data smoke completes all configurations.
3. One worker is intentionally interrupted; the other workers finish; explicit
   resume completes without missing or duplicated work.
4. The extended score-blind audit passes.
5. A separately run sequential smoke with identical sites and seeds matches
   the parallel artifacts within the frozen `1e-7` numerical tolerance.
6. Measured wall-clock speedup is at least `1.5x`.
7. A break-even check shows that restarting would finish meaningfully earlier
   than allowing the active single-worker run to complete.

Failure of any gate keeps `step10_neural_full_v1` as the primary execution.
Its artifacts will never be mixed with a parallel run.

## Validation evidence

The clean three-worker smoke `step10_parallel_clean_smoke_v1` evaluated
CALTECH, CMU, and KKI across all 14 configurations. The extended score-blind
audit checked 714 prediction rows, 42 hidden metric rows, 2,142 diagnostic
rows, 84 runtime rows, three worker roots, and three canonical site copies
with zero warnings.

In `step10_parallel_recovery_smoke_v1`, worker 0 was intentionally interrupted
after its first durable checkpoint. Workers 1 and 2 continued and sealed their
sites; the coordinator correctly refused to merge. Explicit resume retained
those sites, resumed worker 0, completed the merge, and passed the same audit.

The sequential control `step10_sequential_equivalence_smoke_v1` used identical
sites, configurations, inputs, and seeds. The score-blind comparison found
exact equality (`0.0` maximum absolute difference) for predictions, metrics,
diagnostics, tuning, inner-site scores, and training curves. Runtime-only
resource fields were excluded as intended. Parallel wall time was about 131
seconds versus 213 seconds sequentially, a measured speedup of about `1.63x`.
