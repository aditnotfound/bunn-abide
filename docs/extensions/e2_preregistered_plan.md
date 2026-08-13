# E2 preregistered synthetic-geometry plan

This document records the E2 design before any E2 test probability or metric is
opened. The machine-readable contracts are
`configs/extensions/e2_synthetic_geometry_v1.json` and
`configs/extensions/e2_analysis_v1.json`.

## Question

Does learned bundle transport improve classification specifically when a
node-wise coordinate alignment is necessary, fixed across samples, and
recoverable from label-independent features?

The primary comparison is a paired difference in differences: the learned-BuNN
minus GCN balanced-accuracy difference under recoverable fixed geometry (S1),
minus the same operator difference when no geometry is required (S0).

## Protection against a benchmark designed to reward BuNN

Class 0 and class 1 use the same node-value marginals. They differ in graph
smoothness: class 1 is smooth on the true ring graph and class 0 is a
participant-specific node permutation. S0 checks whether ordinary aggregation
can solve the task without alignment. An oracle supplied with the true maps is
the positive control. Learned-local has the same learned-map capacity as BuNN
without cross-node diffusion. Wrong-topology, shuffled-marker, transport-noise,
and subject-specific-frame conditions test whether any learned advantage follows
the intended boundary conditions.

## Evaluation discipline

Each of ten replicate seeds generates a balanced 600-sample dataset with fixed
360/120/120 train/validation/test partitions. Selection uses validation loss;
the test partition is evaluated once. All cell predictions remain embargoed
until every expected cell passes a score-blind structural and hash audit. The
analysis is then executed once under the frozen analysis contract.

## Interpretation

The mechanism is classified as supported only if the oracle positive control,
the primary conditional advantage, and the matched learned-local comparison all
pass their pre-specified gates without an S0 capacity warning. Synthetic success
would explain a computational boundary condition, not brain biology, and would
not revise the completed ABIDE-I result.
