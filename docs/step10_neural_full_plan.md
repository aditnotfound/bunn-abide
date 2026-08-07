# Step 10: Frozen Full Neural Evaluation

Status: **protocol and analysis contract frozen before any accepted neural
held-out result. Runner and auditor require score-blind smoke validation.**

## Primary configuration matrix

The full study uses 14 unique configurations per outer site:

- identity at 0% as the no-exchange anchor;
- learned-local at 0% as the same-capacity BuNN anchor;
- GCN at 1%, 5%, 10%, and 20%;
- trivial-bundle heat diffusion at 1%, 5%, 10%, and 20%; and
- learned BuNN at 1%, 5%, 10%, and 20%.

Learned-local contains the same feature-conditioned O(2) map generators,
pointwise updates, and parameter count as learned BuNN, but applies no
inter-node diffusion. It prevents an apparent BuNN advantage from being
attributed automatically to transport when it may instead reflect additional
node-wise capacity.

## Nested training contract

For every outer held-out site and configuration, four frozen grouped inner
folds compare the same four AdamW candidates: learning rates `3e-4` and `1e-3`
crossed with weight decay `1e-5` and `1e-4`. Each candidate is evaluated with
tuning seeds `20260803` and `20260804`.

Every inner fit uses batch size 8, class-balanced BCE, gradient-norm clipping
at 5, a maximum of 150 epochs, a minimum of 10 epochs, and early stopping on
unweighted mean validation-site BCE with patience 20 and minimum delta
`1e-4`. Candidate selection uses unweighted mean per-site balanced accuracy
across all validation sites and both tuning seeds. Exact ties prefer higher
weight decay and then lower learning rate.

The final epoch count is the ceiling of the median one-based best epoch across
the eight selected-candidate inner fits. The selected model is refit on all
outer-training sites and evaluated once on the held-out site using each of five
fixed final seeds (`20260803` through `20260807`).

## Workload and artifacts

The full contract contains 8,064 inner fits, 1,260 final fits, 52,780 held-out
prediction rows, and 1,260 per-site/seed metric rows. Each site is sealed only
after predictions, metrics, tuning scores, inner-site scores, training curves,
gauge-aware diagnostics, runtimes, warnings, hashes, and its completion marker
exist.

The runner stores encoder, layer-1, and layer-2 common-frame diagnostics for
every held-out participant and final seed. Normalized effective rank is the
primary representation endpoint; dispersion, cosine similarity, and invariant
edge transport distance are secondary.

## Prelaunch gates

1. The complete AWS unit suite must pass.
2. Every operator/configuration must pass a real-data GPU forward/backward
   smoke.
3. A reduced one-site real-data smoke must run all 14 configurations, create
   the complete artifact structure, intentionally interrupt after a durable
   epoch checkpoint, and resume under an unchanged contract.
4. `scripts/audit_neural_full.py` must recompute and validate every hidden
   smoke metric without reporting values.
5. Any executable-code correction retires that smoke attempt and requires a
   new run ID under the corrected commit.

## Results embargo

The full runner may calculate held-out probabilities internally, but neither
the runner summary nor the score-blind auditor prints them or any predictive or
representation value. Full results may be analysed only after all 18 sites are
sealed and the independent audit passes.
