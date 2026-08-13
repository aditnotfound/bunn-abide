# Final claim and literature audit

Status: completed for the separate `paper/yhsa-final/report.tex` report.

## Evidence boundary

The report contains three studies with different evidential roles:

1. **Study 1:** the pre-specified ABIDE-I operator comparison. This is the
   confirmatory result for the project.
2. **E1:** a post-hoc intervention audit of the 360 accepted BuNN checkpoints.
   It tests whether learned maps and the selected topology affected the frozen
   model's own predictions.
3. **E2:** a post-hoc synthetic known-geometry study. It tests whether the same
   transport design can help when topology dependence and recoverable coordinate
   frames are deliberately present.

The earlier E2 report contained one contradictory Methods sentence saying that
the mechanism audits were not included. The final report now states that E1 and
E2 are included as separately frozen, post-hoc evidence. Five-layer models,
Wang-style cGCN training, and ABIDE-II remain outside the completed findings.

## Numerical audit

The report's Study 1 quantities were checked against
`reproducibility/result_snapshot.json` and the generated LaTeX macros:

| Claim family | Checked values | Status |
| --- | --- | --- |
| Cohort | 769 downloaded, 15 excluded, 754 eligible, 371 ASD, 383 control, 18 sites | matched |
| Classical baselines | connectome elastic net 0.6401; combined 0.6336; covariates 0.5652 equal-site BA | matched |
| Primary prediction | BuNN minus GCN -0.00958, CI [-0.03665, 0.01146], p=0.524 | matched |
| Baseline contrast | BuNN minus elastic net -0.05516, CI [-0.08297, -0.02752], p=0.0018 | matched |
| Representation | matched-anchor effective-rank contrast -0.00719, CI [-0.01310, -0.00105] | matched |
| Efficiency | GCN 5,889 parameters and 15.09 fit-hours; BuNN 8,529 and 31.14 | matched |

E1 and E2 values remain tied to their own result summaries, claim ledgers, and
sealed analysis archives. No accepted prediction or metric archive was changed
while preparing the final report.

## Literature comparison rule

Published ABIDE percentages are shown as protocol descriptions, not as a
leaderboard. They differ in atlas, cohort, node definition, input type, split,
metric, and model-selection access. Ordinary accuracy is not interchangeable
with equal-site balanced accuracy.

Wang et al.'s 71.6% is especially important because it used a leave-one-site-out
design. The paper used 1,057 participants, CCS preprocessing, CC200 ROI time
series, a training-derived k-nearest-neighbour graph, five EdgeConv-style graph
layers, and temporal averaging. The released LOSO script also supplies the
left-out site's labels to Keras validation and uses validation performance for
stopping and checkpoint choice. The number is therefore retained as a
historical paper result, while the code audit is disclosed as a reason it is not
a clean unseen-site comparator.

Parisot et al.'s 70.4% is a 10-fold transductive population-graph result on 871
participants. Its nodes are participants, not brain regions, and its graph uses
phenotypic information including acquisition site and sex. It answers a
different question from unseen-site brain-graph classification.

Han et al. provide the closest conceptual comparison rather than a directly
matched score. Their controlled benchmark used 1,009 ABIDE participants,
CC200 connectivity matrices, and AUROC, and reported that simple models often
matched or exceeded graph models while aggregation performance declined with
density. That pattern is consistent with Study 1, but the absolute values are
not directly compared.

## Permitted final synthesis

The report may say that Study 1 found no detected BuNN advantage under the
frozen ABIDE-I pipeline; E1 showed that the learned map system influenced the
accepted model's predictions; and E2 showed that bundle transport can be useful
when its computational assumptions are planted and recoverable.

The report must not claim that E1 or E2 independently confirms an ABIDE-I brain
mechanism, that ABIDE-I contains biological bundle geometry, that published
accuracy differences rank model quality, or that BuNN is generally inferior.

