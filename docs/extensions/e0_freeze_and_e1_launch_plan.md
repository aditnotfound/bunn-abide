# E0 freeze and E1 execution plan

Date frozen: 2026-08-13

## Decision

The immediate extension is E1 only. It uses accepted Study 1 artifacts and
accepted final learned-BuNN checkpoints to test competing explanations of the
completed null result. It does not retrain or replace Study 1.

Synthetic geometry, five-layer/capacity training, the Wang cGCN leakage audit,
and ABIDE-II are separate deferred studies. Their present data, implementation,
or runtime gates are not strong enough to combine with E1 without lowering the
quality of the completed paper.

## Execution order

1. Validate the sealed archive and integrity-certificate hashes.
2. Selectively extract 360 learned-BuNN outer-final checkpoints plus the
   canonical archived predictions and run metadata.
3. Reproduce every CALTECH learned-BuNN probability for four densities and
   five final seeds before applying an intervention.
4. Implement and test identity-map reset, fixed node-map shuffle, fixed random
   orthogonal maps, exact-degree topology rewiring, and encoded-node
   permutation equivariance.
5. Run the complete CALTECH smoke with 100 stored permutations for each random
   family. Keep scores sealed; record only runtime, resource, completeness, and
   tolerance-gate status.
6. Independently audit shapes, hashes, reproduction, structural invariants,
   random-seed coverage, and finite outputs.
7. Project the 18-site runtime with 30 percent contingency. Launch the full E1
   run only if at least 12 hours remain for analysis and paper QA.
8. Freeze the E1 analysis on synthetic fixtures, unseal once, and update the
   explanation table without changing the Study 1 conclusion.

## Scientific boundary

An inference-time intervention can establish whether a trained checkpoint is
sensitive to a computational component. It cannot establish that functional
brain activity follows a learned bundle geometry. The primary result remains
equal-site held-out performance, with participant-weighted summaries secondary.

The machine-readable contracts are:

- `configs/extensions/registry_v1.json`
- `configs/extensions/e1_checkpoint_interventions_v1.json`
