# E2 synthetic known-geometry results

## Integrity status

The corrected `e2_synthetic_full_v2` run completed all 700 frozen cells: ten
conditions, seven matched operators, and ten paired replicate seeds. Every cell
passed its score-blind audit, all 100 condition-replicate groups were sealed by
an independent full-run audit, and the analysis was unblinded once under the
frozen E2 analysis protocol. Independent recomputation matched all nine fixed
analysis artifacts.

The earlier v1 run was stopped before unblinding because the S0 constructor
produced reflections in half its nominal identity maps. It is quarantined and
none of its checkpoints or predictions entered v2.

## Pre-specified primary result

The conditional learned-BuNN advantage was **42.75 percentage points** in test
balanced accuracy (95% paired bootstrap CI **38.42 to 47.67**; exact two-sided
sign-flip **p = 0.001953**). This estimand subtracts the learned-BuNN-minus-GCN
difference in the no-geometry condition (S0) from the same difference under
fixed recoverable geometry (S1).

All ordered gates passed:

| Frozen gate | Mean difference (pp) | 95% CI (pp) | Exact p | Decision |
|---|---:|---:|---:|---|
| Oracle minus GCN in S1 | 43.50 | 39.42 to 48.00 | 0.001953 | Passed |
| Conditional BuNN advantage | 42.75 | 38.42 to 47.67 | 0.001953 | Passed |
| BuNN minus learned-local in S1 | 48.67 | 45.83 to 51.50 | 0.001953 | Passed |
| BuNN minus GCN in S0 | -0.25 | -0.58 to 0.00 | 0.500000 | No capacity warning |

The frozen classification is therefore **supported**: bundle transport helped
when alignment was necessary, fixed across samples, recoverable from a
label-independent marker, and used on the correct topology.

## Boundary-condition results

| Condition | GCN BA | Learned-local BA | Learned BuNN BA | Oracle BA |
|---|---:|---:|---:|---:|
| S0: no geometry | 1.000 | 0.508 | 0.998 | 1.000 |
| S1: recoverable geometry | 0.565 | 0.503 | 0.990 | 1.000 |
| S2: incorrect topology | 0.507 | 0.503 | 0.493 | 0.517 |
| S3: shuffled frame marker | 0.550 | 0.496 | 0.503 | 1.000 |
| S5: unlearnable subject frames | 0.522 | 0.490 | 0.498 | 1.000 |
| S6: global feature analogue | 1.000 | 1.000 | 1.000 | 1.000 |

Relative to S1, learned-BuNN balanced accuracy fell by 49.67 pp under incorrect
topology, 48.67 pp under shuffled marker geometry, and 49.17 pp under
unlearnable subject-specific frames. All three paired exact p-values were
0.001953.

The marker-noise sweep supplied a graded check. Learned-BuNN mean balanced
accuracy was 0.990 at 0 degrees, 0.992 at 15 degrees, 0.990 at 30 degrees,
0.891 at 60 degrees, and 0.511 at 120 degrees. The 60-degree result was variable
across seeds (SD 0.202), so the sweep supports a corruption boundary rather than
a precise universal tolerance threshold.

## What this explains about ABIDE-I

E2 shows that the bundle operator can be useful when three conditions hold:
the task depends on neighbour relations, the correct topology is supplied, and
the coordinate alignment is stable and recoverable. E1 showed that learned maps
in the ABIDE-I models were computationally active, but topology perturbation had
only a modest predictive effect. E2's S6 condition adds a controlled analogue:
when each node already receives a replicated global statistic, every operator,
including identity and learned-local, reached 1.000 balanced accuracy. This is
consistent with the proposed explanation that full connectivity-row node
features can make additional message passing redundant.

This is not proof that ABIDE-I lacks brain geometry. It shows that the frozen
ABIDE-I representation did not supply the combination of topology dependence
and recoverable transport structure that produced a BuNN advantage in E2.

## Important limitation

Learned-BuNN prediction in S1 was nearly perfect, and mean relative-transport
error improved from the identity-frame value of 30.62 to 22.07, but it did not
approach the oracle error of zero. The model therefore found a useful
feature-conditioned transport solution without exactly recovering the planted
maps. Claims must concern task-relevant alignment, not exact ground-truth map
identification.

## Runtime

The 700 cells used 2,155.6 aggregate training seconds (mean 3.08 seconds per
cell; mean selected epoch 71.17) on the A10G runner. This is aggregate cell time,
not a claim about hardware-independent efficiency.
