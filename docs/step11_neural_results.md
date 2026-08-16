# Step 11 Confirmatory Neural Results

## Audit and analysis provenance

The accepted run was `step10_neural_full_parallel_v1` under training code
version `c246744`. All 18 sites completed, and the score-blind integrity audit
passed before any neural value was viewed. The Step 11 implementation was then
frozen in commit `6e1716e`, passed 4/4 synthetic analysis tests and the complete
59-test AWS suite, and required the exact run ID for unblinding.

The guarded analysis completed once without an error, compatibility change,
or rerun. Its 15 generated artifacts passed manifest-hash verification. The
AWS/local analysis archive SHA-256 is
`d03235fd99d08c6eff639070731c822d0fc9bf57d8382a2a697f2678b2ee5405`.

## Primary predictive result

Five seeds were averaged within site/configuration before equal-site
inference. The primary estimand was normalized area under the held-out-site
balanced-accuracy density curve. Identity anchored the GCN and trivial-bundle
curves at 0%; learned-local anchored learned BuNN.

| Contrast | Mean difference | Paired 95% bootstrap interval | Exact sign-flip p |
| --- | ---: | ---: | ---: |
| Learned BuNN minus GCN curve | -0.0096 | [-0.0366, 0.0115] | 0.524 |
| Learned BuNN minus trivial-bundle curve | -0.0062 | [-0.0293, 0.0108] | 0.681 |
| Learned BuNN nonzero densities minus learned-local | -0.0167 | [-0.0347, 0.0009] | 0.096 |
| Learned BuNN nonzero densities minus identity | -0.0146 | [-0.0320, 0.0018] | 0.116 |
| Learned BuNN curve minus connectome elastic net | -0.0552 | [-0.0830, -0.0275] | 0.0018 |

No BuNN predictive advantage was detected. The upper paired confidence bound
for the primary BuNN-minus-GCN contrast was 0.0115, below the pre-specified
0.03 practical margin. Under this pipeline, the analysis therefore excludes a
three-percentage-point BuNN curve advantage at the stated interval level; it
does not establish exact equivalence.

The equal-site curve means were 0.5945 for GCN, 0.5911 for trivial bundle, and
0.5849 for learned BuNN. Identity at 0% averaged 0.6033, learned-local averaged
0.6053, and the previously audited connectome elastic net averaged 0.6401.
The highest descriptive neural configuration was GCN at 1% density (0.6166),
followed by learned BuNN at 1% (0.6114). These single-density observations are
secondary and do not replace the curve-level primary estimand.

At 20% density, learned BuNN was worse than learned-local by 0.0282 (Holm-
adjusted p = 0.0357) and worse than identity by 0.0261 (Holm-adjusted p =
0.0316). BuNN was below the elastic-net reference at all four nonzero
densities; the Holm-adjusted comparisons were below 0.05 at 5%, 10%, and 20%
density, but not at 1%.

## Representation result

The primary representation endpoint was common-frame normalized effective
rank at layer 2. The estimand compared each operator's density-curve change
with its matched 0% anchor before taking learned BuNN minus GCN.

| Endpoint | Mean matched change difference | Paired 95% bootstrap interval |
| --- | ---: | ---: |
| Normalized effective rank (primary) | -0.0072 | [-0.0131, -0.0010] |
| Normalized dispersion | -0.0417 | [-0.0523, -0.0306] |
| Mean pairwise cosine similarity | 0.0415 | [0.0319, 0.0507] |
| Invariant edge transport distance | -0.6436 | [-0.8205, -0.4720] |
| Encoder-to-layer-2 effective-rank change | -0.0056 | [-0.0144, 0.0048] |

Learned BuNN had higher *absolute* effective rank than GCN across the plotted
density range, but its learned-local 0% control also began substantially
higher than identity. After comparing density-dependent change relative to
these matched anchors, the primary difference favored GCN. The dispersion and
cosine endpoints pointed in the same anti-preservation direction. Thus, the
raw rank separation cannot be attributed specifically to inter-node bundle
transport; it is consistent with additional learned pointwise capacity.

These representation and predictive patterns co-occur. They do not establish
that representation collapse caused predictive performance.

## Site and seed heterogeneity

The BuNN-minus-GCN curve difference was positive at 8 sites and negative at 10.
CALTECH was an influential negative site (-0.1800), while the largest positive
site difference was UM_2 (+0.0561). The pre-specified equal-site bootstrap
retains both; influence and leave-one-site sensitivity belong in explicitly
labelled robustness analysis rather than replacing the confirmatory estimate.

Seed variation was material. Across the 14 configurations, the range of the
five seed-specific equal-site balanced accuracies was 0.0265 to 0.1058. This
supports averaging the five pre-specified seeds and argues against reporting a
single favorable seed.

## Decision-rule conclusion

The three required conditions were all false:

1. BuNN did not preserve the primary gauge-aware representation endpoint more
   favorably than GCN.
2. BuNN did not improve held-out-site prediction over GCN, identity, and
   learned-local.
3. BuNN did not exceed the connectome elastic-net baseline.

The defensible conclusion is:

> No complete transfer of BuNN's proposed anti-collapse advantage was detected
> under this frozen ABIDE-I pipeline. Increasing graph density degraded the
> predictive and gauge-aware representation behavior of learned bundle
> transport, while a regularized connectome baseline remained stronger.

This is a conditional computational result. It does not establish biological
bundle geometry, excitation or inhibition, causal neural flow, diagnostic
utility, exact model equivalence, or general BuNN inferiority.
