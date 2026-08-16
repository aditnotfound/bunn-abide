# Step 8: Neural Operator Implementation

Status: **core graph data and operator layer implemented and tested.** No
neural classifier has been trained, tuned, or evaluated on a held-out site.

## Frozen representation contract

The same 754 participants and 18 sites used in the audited baseline study are
loaded from the Fisher-z connectome artifact after row-by-row alignment with the
baseline table. Each participant is a separate dense graph with 116 AAL nodes.
Every node receives its complete 116-value connectivity row as a feature.

Propagation graphs are binary and undirected. At each density, the code keeps
the strongest positive off-diagonal associations from all 6,670 possible
undirected edges, not merely from the positive subset. The deterministic counts
are therefore 0, 67, 334, 667, and 1,334 edges per participant at 0%, 1%, 5%,
10%, and 20%. Stable strict-upper-triangle ordering breaks exact score ties.

Feature standardization is implemented as a separate object that can only be
fitted on supplied graphs. The eventual runner must fit it separately inside
each relevant training partition; it never receives site IDs or labels.

## Matched operators

The shared pilot backbone is a 116-to-32 encoder, two propagation layers,
GELU, dropout 0.2, mean node pooling, and one graph-level logit head. The
32-dimensional hidden state is organised as eight bundles, two vector-field
channels, and two dimensions per bundle. Optimizer and training choices remain
unfrozen until the engineering pilot.

| Operator | Propagation rule | Parameters in the two-layer pilot backbone |
| --- | --- | ---: |
| Identity | Pointwise update only; no inter-node exchange | 5,889 |
| GCN | Standard normalized adjacency with one self-loop | 5,889 |
| Trivial bundle | Exact heat diffusion with identity transports | 5,889 |
| Learned BuNN | Exact heat diffusion with learned feature-conditioned O(2) transports | 8,529 |

The extra 2,640 learned-BuNN parameters come entirely from its transport-map
generators. They are reported as a capacity difference; the trivial-bundle
control isolates diffusion without learned transport. A later capacity control
may be added only as a separately labelled secondary analysis, never silently
substituted for this primary comparison.

## BuNN implementation and diagnostics

For every bundle at node \(v\), the learned orthogonal map \(O_v\) maps the
local field to a common frame. The layer applies the shared update in that
common frame, diffuses it with \(\exp(-tL)\), then maps it back with
\(O_v^T\). This follows the flat-bundle heat-diffusion ordering described by
Bamberger et al. The pilot uses eight direct two-dimensional maps: four
rotations and four reflections, generated from the current node features.

Representation diagnostics first map embeddings into that common frame. They
report normalized effective rank, normalized dispersion, mean pairwise cosine
similarity, and the invariant edge quantity
\(\lVert x_u - O_u^T O_v x_v\rVert^2\). Raw local BuNN coordinates are never
stacked and compared directly.

## Completed evidence

- 13 Phase 8 synthetic unit tests passed: exact density/symmetry/tie behavior,
  baseline-table alignment, train-only scaling, 0%-density behavior,
  identity-transport heat equivalence, orthogonality, transport orientation,
  gauge invariance, subject-batch isolation, finite gradients, and deterministic
  initialization.
- The real 754-subject input check passed at all five densities.
- A GPU-only smoke ran every operator/density combination on eight real graphs.
  It used no labels and reported no predictive metric. All 20 cells had finite
  forwards, losses, gradients, and diagnostics; peak allocated memory was below
 127 MB in this small batch. Learned transport orthogonality error was at most
  `1.2e-7`. The complete AWS repository suite passed 34 tests.

## Remaining gate: Step 9

The next work is a fold-aware neural runner, not the scientific run. It must
consume the existing frozen outer and inner assignments, fit feature scaling
only on the relevant training partition, implement atomic checkpoints/resume,
hide held-out-site metrics during engineering checks, and record runtime and
memory. Only after that pilot is complete can the neural training grid, seeds,
epochs, patience, and retry policy be frozen for the full evaluation.
