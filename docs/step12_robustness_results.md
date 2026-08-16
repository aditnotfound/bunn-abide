# Step 12A Robustness Results

## Provenance

The secondary robustness contract was frozen in commit `c797122` after Step 11
was complete and before any leave-one-site, individual-seed, alternative-
weighting, or exhaustive-curve calculation was executed. The analyzer consumed
only the hash-bound Step 11 tables and the audited elastic-net site table; it
did not retrain a model or change data, folds, seeds, densities, operators, or
endpoints.

The guarded analysis completed once. All nine generated artifacts passed their
manifest hashes, and both figures passed visual inspection. The archived
package SHA-256 is
`c29ea30fa65fa4acd04bb80889c6cd30aad0caa4be79c4e86a6322574811be0c`.

## Leave-one-site influence

The full 18-site BuNN-minus-GCN curve estimate was -0.00958. Seventeen of 18
leave-one-site estimates remained negative, ranging down to -0.01344. Excluding
CALTECH produced the sole positive point estimate, +0.00045, with a paired 95%
bootstrap interval of [-0.01474, 0.01548]. No exclusion produced an interval
wholly above or below zero for the BuNN-minus-GCN comparison.

This establishes point-estimate site sensitivity, not evidence that BuNN
outperforms GCN without CALTECH. The sign flip is tiny and imprecise, while the
full pre-specified 18-site estimate remains the confirmatory result.

Two other findings were much more stable:

- BuNN remained below the connectome elastic net after every site exclusion.
  All 18 intervals were wholly negative; leave-one-site point estimates ranged
  from -0.06267 to -0.04870.
- The matched-anchor effective-rank change remained below GCN after every site
  exclusion. All 18 intervals were wholly negative; point estimates ranged
  from -0.00853 to -0.00583.

## Seed stability

The five seed-specific BuNN-minus-GCN curve estimates were:

| Seed | Mean difference | Paired 95% bootstrap interval |
| --- | ---: | ---: |
| 20260803 | +0.0223 | [-0.0012, 0.0487] |
| 20260804 | +0.0059 | [-0.0351, 0.0418] |
| 20260805 | -0.0270 | [-0.0672, 0.0069] |
| 20260806 | -0.0159 | [-0.0785, 0.0325] |
| 20260807 | -0.0333 | [-0.0636, -0.0052] |

Two seeds favored BuNN by point estimate and three favored GCN. No favorable
seed had an interval wholly above zero; one unfavorable seed had an interval
wholly below zero. Selecting seed `20260803` would therefore create a much more
favorable story than the complete five-seed evidence supports.

BuNN was below elastic net for all five seed-specific point estimates. Four
seed intervals were wholly negative; seed `20260804` crossed zero.

## Alternative summaries

| Contrast | Equal-site mean | Participant-weighted mean | Median site difference |
| --- | ---: | ---: | ---: |
| BuNN curve minus GCN curve | -0.0096 | +0.0030 | -0.0044 |
| BuNN curve minus elastic net | -0.0552 | -0.0445 | -0.0464 |
| BuNN minus GCN matched-anchor effective-rank change | -0.0072 | -0.0109 | -0.0092 |

The small participant-weighted BuNN-minus-GCN sign change confirms sensitivity
to whether large sites dominate. It is descriptive and cannot replace the
pre-specified equal-site estimand. Both alternative summaries preserve the
elastic-net and representation conclusions.

## Exhaustive curve comparisons

All three propagation curves were below the connectome elastic-net reference:

| Curve minus elastic net | Mean difference | Paired 95% bootstrap interval |
| --- | ---: | ---: |
| GCN | -0.0456 | [-0.0713, -0.0194] |
| Trivial bundle | -0.0490 | [-0.0739, -0.0226] |
| Learned BuNN | -0.0552 | [-0.0823, -0.0265] |

GCN-minus-identity and trivial-bundle-minus-identity intervals crossed zero.
BuNN-minus-learned-local was negative but narrowly included zero. The
BuNN-minus-identity percentile interval was narrowly negative while its exact
sign-flip p-value was 0.071; this secondary discrepancy is reported rather than
resolved by selecting the favorable inferential method.

## Computational efficiency

| Operator | Parameters | Fit-hours | Mean seconds/fit | Peak GPU GiB | Curve BA |
| --- | ---: | ---: | ---: | ---: | ---: |
| GCN | 5,889 | 15.09 | 20.39 | 0.373 | 0.5945 |
| Trivial bundle | 5,889 | 25.99 | 35.12 | 0.377 | 0.5911 |
| Learned BuNN | 8,529 | 31.14 | 42.08 | 0.378 | 0.5849 |

Learned BuNN used 44.8% more parameters and approximately 2.06 times the total
fit time of GCN, with essentially the same small GPU-memory footprint, while
its curve performance was lower. This is an engineering efficiency result,
not a claim that BuNN is generally inefficient outside this implementation.

## Robustness classification

The frozen rule classifies the result as a **mixed site- and seed-sensitive
null**:

- Site-sensitive because the CALTECH exclusion changes the primary point-
  estimate sign, although not the interval-level conclusion.
- Seed-sensitive because two seed point estimates favor BuNN and three favor
  GCN, with no favorable seed interval excluding zero.
- Still negative against elastic net and for matched-anchor representation
  behavior under every leave-one-site exclusion.

Step 11 remains unchanged: no complete transfer of the proposed BuNN anti-
collapse advantage was detected under this ABIDE-I pipeline. Step 12 adds that
the small BuNN-versus-GCN predictive difference is unstable in direction,
whereas the baseline and matched-anchor representation findings are robust.
