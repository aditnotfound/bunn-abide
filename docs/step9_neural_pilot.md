# Step 9: Score-Blind Neural Engineering Pilot

Status: **completed and integrity-audited as an engineering check.** This is
not a neural prediction result and does not select a preferred operator.

## Purpose and fixed scope

The pilot validates the real GPU training/recovery path before a full
nested, held-out-site neural evaluation is implemented. It uses the frozen
754-participant ABIDE-I artifact, but the `CALTECH` outer-test site is excluded
from all model tensors. The run contains only outer-training and one grouped
inner-validation partition.

The frozen pilot grid is four operators by five densities: identity, GCN,
trivial-bundle heat diffusion, and learned orthogonal BuNN at 0%, 1%, 5%, 10%,
and 20% density. Each cell runs for three epochs with the shared 116-to-32,
two-layer backbone, batch size 8, AdamW (`1e-3` learning rate, `1e-4` weight
decay), and seed `20260803`.

Only fitting/validation BCE loss, runtime, GPU allocation, checkpoint state,
and failure state are written. There is no held-out-site tensor, probability,
predicted label, balanced accuracy, AUROC, threshold, or operator conclusion.

## Recovery and deployment evidence

Two engineering defects were found before the accepted pilot and corrected:

1. The initial AWS deployment lacked the frozen neural-operator contract. The
   managed launcher now checks every required configuration, data, and split
   artifact before creating a process. This attempt created no run metadata,
   tensor, checkpoint, or alert.
2. The first recovery attempt treated every unstarted cell as an error when a
   global `--resume` flag was present. The corrected runner validates and
   resumes existing checkpoints while starting untouched cells fresh. A
   regression test covers this behaviour. Because the executable code changed,
   the partial `step9_neural_pilot_v1` artifact is retired and was not reused.

The accepted `step9_neural_pilot_v2` run intentionally stopped after the first
durable identity/0%-density epoch checkpoint. It resumed under the unchanged
code and input contract, completed all remaining work, and published its
terminal SNS notification.

## Integrity audit

`scripts/audit_neural_pilot.py` was tested on valid and mutated synthetic
artifacts, then run against `step9_neural_pilot_v2` with recovery required. It
verified:

- 20 of 20 operator-density cells are complete;
- 20 checkpoints have exactly three sequential epochs each and match their
  operator/density contracts;
- the loss-only history contains the expected 60 records with finite
  numerical-stability fields;
- the recorded hashes for the history and summary match;
- run metadata and live status explicitly disable outer-test evaluation;
- no top-level predictive-output artifact exists; and
- the intentional interruption is recorded as successfully resumed.

The audit certificate passed. The maximum observed GPU allocation was about
103 MiB, so batch size 8 is operationally safe on the A10G; this does not
imply that a larger batch is scientifically preferable.

The complete accepted run directory, including the checkpoints and integrity
certificate, was archived from AWS and copied to the ignored local output
store. The remote and local archive SHA-256 values match:
`04a8b70efb575383db1f8b1d9269cc81044d54267debd1fcad46a9e09ed24c84`.

## What this does and does not permit

The pilot permits the next pre-result task: freezing the complete neural
evaluation protocol (training schedule, inner-validation rule, seeds,
candidate budget, final refit rule, diagnostics, and full-run audit contract)
using the tested recovery machinery.

It does **not** permit a claim that any GNN, BuNN, density, or representation
metric is better. All predictive comparisons remain uncomputed.
