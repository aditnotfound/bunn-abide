# Post-hoc nonlinear connectome baseline

## Why this is being added

Study 1 compared the graph models with regularized logistic regression. That
baseline is strong, but it is linear in the 6,670 connectome features. A
nonlinear model on the same vectorized features can test a narrower question:
did the neural models lose because graph propagation was unhelpful, or because
the task required nonlinear decision boundaries that a linear baseline could
not represent?

This extension does not repair, replace, or reopen Study 1. It was specified
after the Study 1, E1, and E2 results were known. The report must call it a
post-hoc practical comparator.

## Frozen experiment

The model is an RBF-kernel support vector machine trained on the same 6,670
Fisher-z connectome features used by the elastic-net baseline. It uses the same
754 participants, 18 outer held-out-site folds, and four inner grouped folds.
The outer test site remains untouched until the selected model is fitted on all
outer-training sites.

Within each inner fit, a `StandardScaler` is fitted on the fitting partition
and then applied to its validation partition. The model uses balanced class
weights and no probability calibration. Predictions use the fixed SVM decision
threshold of zero, while AUROC uses the continuous decision score.

The grid contains nine candidates:

- `C`: 0.1, 1, or 10;
- `gamma`: 0.25/D, 1/D, or 4/D, where D is 6,670.

Candidates are ranked by the unweighted mean balanced accuracy across the
inner validation sites. A tie selects the lower `C`, followed by the lower
gamma multiplier. These rules were fixed before any RBF-SVM fit.

## Run and disclosure rules

The runner seals one complete outer site at a time and can resume only after
its immutable inputs have been checked. A score-blind audit must confirm all
666 expected fits, all 754 held-out predictions, all 18 test sites, artifact
hashes, and recomputed metrics before results are opened.

The managed full run requires SNS alerts at launch and after the final site is
sealed. The terminal message means that compute finished, not that the result
passed audit or was opened.

The first remote job is a timing smoke, not evidence. The full run is promoted
only if measured time projects to at most 16 hours on the existing machine.
The full run should use three CPU workers at most, leaving one vCPU for the
operating system and the run monitor. The GPU is not used by scikit-learn's
SVC implementation.

The first reported comparison is RBF-SVM versus connectome elastic net.
Comparisons with GCN and learned BuNN follow. Every result is retained,
including a null or worse result. The extension cannot establish external
validation, clinical utility, biological geometry, or a cause of Study 1's
outcome.

## Stop conditions

The run stops rather than changing the protocol if an input hash differs, a
split leaks a site, a held-out site lacks both classes, an artifact cannot be
sealed, or the audit cannot reproduce the recorded metrics. A runtime failure
may resume from a verified site seal. Scientific settings do not change on
resume.
