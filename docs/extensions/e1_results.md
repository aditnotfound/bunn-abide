# E1 accepted-checkpoint intervention results

## Audit status

The full E1 run completed all 18 held-out sites. Every site artifact passed its score-blind audit, the sealed local archive matched AWS at SHA-256 `19ed2729d30b4e379b8e526b8146d1f8d1d11ff09e233c6c7142fc8a2762eb4d`, and the frozen analysis was independently recomputed before interpretation. E1 is a post-hoc mechanistic audit of accepted Study 1 checkpoints; it does not replace the frozen Study 1 result.

## Primary result

Replacing the learned maps with random orthogonal maps produced the clearest change: held-out balanced accuracy fell by 8.86 percentage points (95% paired site-bootstrap interval -12.36 to -4.78; exact sign-flip p=0.00095; Holm-adjusted p=0.0038). The effect was negative at every tested density and in 15 of 18 sites after density averaging. This is evidence that the trained predictions depended on the learned map system rather than merely requiring arbitrary orthogonal rotations.

Replacing the maps with identities reduced balanced accuracy by 4.72 points (95% interval -8.56 to -0.29). Its unadjusted sign-flip p-value was 0.0468, but the Holm-adjusted value was 0.1404, so it did not pass the four-contrast multiplicity-controlled test. Shuffling the learned maps between nodes caused a smaller 1.94-point reduction (95% interval -3.77 to 0.05; adjusted p=0.1404). Degree-preserving topology rewiring caused a 1.23-point reduction (95% interval -2.49 to -0.05; adjusted p=0.1404). Only the random-map intervention passed the pre-specified multiplicity-controlled primary test.

## What the interventions reveal

The random-map intervention changed nearly half of the binary decisions (49.46%) and reduced AUROC by 14.59 points. Identity maps changed 34.41% of decisions and reduced AUROC by 9.50 points. Shuffling learned maps changed 15.12% of decisions, whereas degree-preserving rewiring changed 6.96%. The hierarchy is therefore clear: the trained coordinate transformations mattered considerably more than the exact positive-edge topology.

The relatively small node-shuffle effect is important. Keeping the learned map collection but assigning maps to different nodes preserved much more performance than replacing the collection with new random maps. The audit therefore supports dependence on properties learned by the map generator, while providing weaker evidence that each learned map's exact node assignment was essential.

Topology rewiring substantially changed several representation diagnostics but had a small predictive effect. Because every node feature already contained its complete 116-region connectivity profile, the classifier retained global connectome information even when the propagation graph was altered. This offers a direct explanation for the original null comparison: bundle-map computations were active inside BuNN, but graph topology and message passing added relatively little predictive information beyond globally informed inputs.

## Representation and prediction

All interventions changed at least some final-layer gauge-aware diagnostics. Random maps increased normalized effective rank by 0.414 and pairwise cosine similarity by 0.138 while reducing dispersion and edge-transport distance. Rewiring produced a large dispersion reduction and cosine increase despite its modest balanced-accuracy effect. Across the 72 site-density cells, descriptive Spearman associations between diagnostic changes and balanced-accuracy changes were weak (absolute rho at most 0.21).

Representation change therefore did not track predictive change closely. The intervention audit does not support a simple account in which preserving one measured diversity statistic automatically improves held-out classification. These are co-occurring changes, not evidence of causal mediation.

## Scientific conclusion

BuNN did not fail because its learned maps were unused. Randomizing them caused a large, reproducible loss of held-out performance. Instead, the results suggest a different limitation: the model learned computationally important coordinate transformations, but those transformations did not yield an advantage over GCN or elastic-net models in the original cross-site comparison. Exact node-map placement and constructed graph topology contributed less, consistent with the full connectivity row already supplying each node with global information.

This conclusion is conditional on the accepted ABIDE-I cohort, C-PAC no-GSR AAL pipeline, positive-edge graphs, density grid, trained checkpoints, and selected interventions. It does not establish biological bundle geometry, anatomical transport, causal neural flow, or clinical utility.
