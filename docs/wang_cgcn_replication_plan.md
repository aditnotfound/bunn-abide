# Wang cGCN replication benchmark

Status: **planned secondary study; not part of the frozen BuNN result.**

## Why this exists

Wang, Li, and Hu reported a mean leave-one-site-out ABIDE accuracy of 71.6%
with a connectivity-neighbour graph convolutional network (cGCN).  This is a
meaningfully different model family from the completed AAL-connectome operator
study, so it must be tested as a separate, explicitly labelled replication
benchmark rather than substituted into the finished experiment.

## What the published model does

- Input: 315 padded raw resting-state fMRI time points per subject in the
  released leave-one-site-out script, rather than a finished correlation
  matrix.
- Nodes: CC200 regions.
- Graph: one shared k-nearest-neighbour graph, built from functional
  connectivity in the training data; the paper's strongest leave-one-site-out
  result used `k = 5`.
- Network: five graph-convolution layers followed by temporal aggregation and
  an ASD/control classifier.
- Released script: Adam (`1e-4`), batch size 4, 50 epochs, graph-convolution
  widths `[8, 16, 32, 64, 128]`, and `k = 5`.

The published 71.6% is **ordinary accuracy** on 1,057 people from 17 sites
under leave-one-site-out evaluation.  It is not numerically interchangeable
with the completed study's 0.640 equal-site balanced accuracy on 754 people
from 18 sites.

## Critical released-code audit

The official leave-one-site-out script loads the left-out site into `x_val`
and `y_val`, passes those arrays to Keras as `validation_data` during every
training epoch, and uses `val_loss` for early stopping and `val_acc` for
checkpoint selection.  Thus the outer-test labels choose the training epoch
and final model weights.  This is outer-test leakage, even though the shared
FC graph itself is constructed from training sites only.

The 71.6% value is therefore a useful historical target for a **paper-fidelity
engineering reproduction**, but it is not a clean estimate of performance on
an unseen site.  The strict replication below deliberately removes this
leakage; its result must not be expected to equal 71.6%.

## Replication rules

1. Obtain the public ABIDE-I CC200 time-series derivative and record a new
   immutable manifest.  Do not alter the original AAL cohort or its results.
2. Recreate the authors' eligibility, preprocessing, CC200 representation,
   time-frame selection, and architecture as closely as public code and paper
   specify.
3. In every outer fold, construct the shared k-NN graph from **outer-training
   participants only**.  The held-out site may never influence its topology,
   normalization, stopping decision, or hyperparameter choice.
4. Use grouped inner validation among outer-training sites to choose
   regularization, epoch/checkpoint, and any unresolved implementation choice.
   The held-out site's labels must be unread until that selection is final.
5. Pre-register two summaries before any result is viewed:
   - paper-compatible equal-site ordinary accuracy, for a faithful comparison
     with the reported 71.6%;
   - equal-site balanced accuracy, AUROC, sensitivity, and specificity, for
     comparison with the completed study.
6. Run random-graph and no-neighbour controls alongside `k = 5`.  A cGCN
   improvement is informative only if the connectivity-neighbour graph beats
   these controls under the same data split.
7. Keep all results, code hashes, warnings, checkpoints, and per-site
   predictions behind the same audit-before-unblinding process used in the
   completed study.

## Required comparison

The result table must distinguish these questions:

| Comparison | What it tests |
| --- | --- |
| Wang-style cGCN vs random graph | whether the connectivity-neighbour graph helps |
| Wang-style cGCN vs time-series no-neighbour model | whether message passing helps beyond temporal features |
| Wang-style cGCN vs frozen elastic net | whether raw-time-series graph learning beats regularized connectomes |
| Wang-style cGCN vs completed GCN/BuNN | whether changing input representation and graph construction changes the outcome |

## Claim boundary

Matching 71.6% would reproduce a published result under an aligned protocol;
it would not validate a clinical diagnostic tool.  Missing 71.6% would not by
itself show an error in either study, because cohort, preprocessing, atlas,
time-frame handling, hyperparameter selection, and metric can each differ.
