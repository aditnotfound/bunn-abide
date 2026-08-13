# Original-outline alignment and paper-scope audit

Status: **document and design audit; no new model result.**

## Source documents checked

- Original: `C:\Users\ASUS\Downloads\Comp-183 Research Outline.pdf`, three
  A4 pages, SHA-256
  `f2b25f1c2926c4361403f9294ef47b73cd3a6d74d678ff02eb806974402e4305`.
- Updated: `C:\Users\ASUS\Downloads\Updated Comp-183 Research Outline.pdf`,
  two A4 pages, SHA-256
  `7b8b802fab7eb6fbd7bc3021ec64b8d3cae93031bf9ff15ec7d1a270fa02e73e`.

Both PDFs were rendered and inspected page by page on 2026-08-13. This audit
also checked the current manuscript Introduction, Discussion, and Conclusion
and the frozen Study 1 protocol/results.

## What is genuinely continuous

The first outline asked why sophisticated graph models often fail to beat
simpler classifiers on functional connectomes and proposed BuNN orthogonal
transport as the object of study. It promised ABIDE evaluation, conventional
baselines, held-out-site testing, component ablations, and analysis of learned
orthogonal maps. The updated outline retained the same problem, architecture,
dataset family, GCN comparison, and representation question, but removed the
unsupported biological interpretation and made the operator comparison
measurable.

The completed Study 1 therefore is not a different project. It is the original
computational question tested under a narrower claim boundary. The parts that
were abandoned were the claim that negative functional correlations establish
heterophily or antagonistic coupling, the broad ADHD comparison, and an
uncontrolled catalogue of unrelated architectures.

## Extension-by-extension alignment

| Extension item | Original-outline connection | Updated-outline connection | Can it explain the Study 1 result? | Paper placement |
| --- | --- | --- | --- | --- |
| E0 evidence freeze and registry | supports the original promise of controlled evaluation | directly supports the pre-specified, site-aware design | no; it protects validity rather than explaining performance | Methods/reproducibility record only |
| E1 competing-hypothesis audit | directly answers the original question of why complex graph models may not outperform simple models | extends the density/representation analysis | yes, if interventions distinguish information redundancy, aggregation loss, map use, optimization, or site dependence | central Results and Discussion |
| E1 existing-artifact audit | checks whether training, tuning, or convergence failure could mimic an architectural null | validates the frozen nested evaluation | partly; it can rule in/out implementation and optimization explanations | concise main-text summary, full supplement |
| E1 identity-reset, shuffled-map, random-map, and topology interventions | directly implements the promised bundle-layer, orthogonal-map, and interpretability ablations | tests whether learned transport was computationally used | yes; this is the strongest real-data explanation family | main paper if complete and audited |
| E1 train-only probes and learning curves | not named explicitly, but serves the original request to investigate why performance changes | tests the proposed information-destruction interpretation | partly; it diagnoses information/data-regime behavior but does not prove causation | main paper only if it clearly separates hypotheses; otherwise supplement |
| E2 synthetic known-geometry suite | directly tests the original geometric operator idea and the claimed properties of BuNN | tests whether the proposed anti-collapse mechanism works under identifiable conditions | yes for computational boundary conditions; no for biological geometry in ABIDE | one main figure/section plus supplement |
| E3 matched depth and skip study | related to the original broad architecture comparison, but depth was not its central claim | post-hoc sensitivity beyond the updated two-layer shared backbone | only indirectly; it tests whether the conclusion was architecture-specific | supplement unless it materially changes the scientific conclusion |
| E3 capacity controls | aligns with the original component-ablation promise | required to keep the operator comparison fair | yes, by separating transport from parameter count | main methods/result summary, full supplement |
| E3 bundle dimension and orthogonality ablations | explicitly promised in the original outline as stalk-dimension and orthogonality tests | not included in the narrow updated core | yes, if they reveal that the chosen O(2) parameterization constrained or destabilized learning | supplement; main text only for a decisive result |
| E3 signed-association sensitivity | preserves the testable computational remnant of the original sign-bias idea | outside the positive-edge updated core and therefore post-hoc | possibly, but only as edge-representation sensitivity | supplement; never restore signed-heterophily or biological-antagonism claims |
| E4 faithful Wang temporal cGCN | fits the original wish to compare other GCN/GNN designs | changes atlas, input representation, graph construction, architecture, cohort, and selection protocol | no direct explanation of why AAL connectivity-row BuNN failed; it tests a different modeling route | separate benchmark/companion paper, or a short contextual comparison |
| E5 ABIDE-II external evaluation | restores the original multi-dataset/generalization ambition while staying in ASD | is the natural locked external test proposed by the updated design | no direct mechanism, but it tests whether the Study 1 ordering transports | main paper if preprocessing compatibility is established; otherwise separate cross-release study |
| Statistical decision table | formalizes the original intent to compare and explain | protects the updated narrow claims | yes, by preventing post-hoc storytelling | analysis contract and supplement; conclusions reflected in main text |
| Engineering/audit gates | not scientific content in the original outline | implements its pre-specification and leakage-control requirements | no | reproducibility package only |

## Original promises: preserved, repaired, or excluded

| Original element | Status | Reason |
| --- | --- | --- |
| Why graph models often trail simple classifiers | preserved and now empirically central | Study 1 directly compares graph operators with identity and elastic net |
| BuNN/bundle layer | preserved | remains the focal architecture |
| Learned orthogonal node maps | preserved and strengthened | common-frame diagnostics, learned-local control, and planned map interventions are more informative than a decorative visualization |
| Bundle-layer ablation | preserved | trivial-bundle and learned-local controls separate diffusion, transport, and extra capacity |
| Orthogonality ablation | not yet performed; restored as post-hoc sensitivity | needed to honor the original operator question without pretending it was confirmatory |
| Stalk/bundle dimension | not yet performed; restored as post-hoc sensitivity | tests whether the fixed O(2) design was restrictive |
| Sign-inducing bias | biological interpretation rejected; computational sensitivity optional | functional-correlation sign does not identify heterophily, excitation, inhibition, or antagonistic coupling |
| Visualize learned maps | replaced by gauge-aware stability and intervention analysis | raw map pictures are frame-dependent and can mislead; relative/invariant quantities are defensible |
| ABIDE | preserved | Study 1 uses ABIDE-I |
| ADHD-200 | excluded from this paper | it is a different diagnosis/task and would be transportability, not replication |
| GCN and simple classifiers | preserved | GCN, identity, covariates, and elastic net are included |
| GAT, Brain Network Transformer, Neural Sheaf Diffusion, SVM | excluded from the core leaderboard | adding architectures without matched inputs/tuning does not explain the operator result |
| k-fold accuracy | replaced | random participant folds are vulnerable to site leakage and are weaker than nested held-out-site evaluation for the stated question |
| leave-one-site-out evaluation | preserved and strengthened | nested training-only tuning and equal-site inference are used |
| site harmonization | not used | harmonization was not necessary to test the frozen operator contrast and can itself leak if fitted improperly; site-aware splitting was retained |
| interpretability | narrowed | computational map use and representation behavior are evaluated; biological localization is not claimed |

## What belongs in the final paper

The strongest coherent paper is:

1. **Study 1:** the completed frozen operator audit and null result;
2. **Mechanism audit:** learned-map/topology interventions plus training and
   representation diagnostics;
3. **Known-ground-truth test:** synthetic conditions showing when bundle
   transport should and should not help;
4. **External test:** ABIDE-II only if the preprocessing is genuinely
   compatible and all choices are locked before results.

Depth, bundle dimension, orthogonality, and signed-association sensitivities
belong primarily in the supplement. They can move into the main paper if they
produce a clear result that changes which explanation survives.

The faithful Wang experiment should not be sold as an explanation of Study 1.
It uses raw temporal CC200 signals and a training-derived k-nearest-neighbour
EdgeConv graph, whereas Study 1 uses AAL-116 Fisher-z connectivity rows and
density-thresholded participant graphs. It is valuable, but it is a separate
architecture/input-representation benchmark.

## Language required in the manuscript

Do not write that “BuNN failed.” The earned statement is:

> Under the frozen ABIDE-I connectivity-row pipeline, learned BuNN transport
> showed no detected predictive or gauge-aware representation-preservation
> advantage over the matched controls.

After the extension, the Discussion may say why only to the extent supported
by interventions:

- “The result is consistent with globally informed node features reducing the
  need for message transport” only if local/global and topology controls agree.
- “The learned maps were not materially used” only if identity-reset and map
  shuffling leave predictions unchanged within uncertainty.
- “Aggregation removed task-relevant information” only if train-only probes
  and propagation interventions show corresponding information loss.
- “The chosen bundle parameterization was restrictive” only if dimension or
  orthogonality controls improve results under the same evaluation.
- “Site heterogeneity contributed” only if pre-declared site-shift diagnostics
  and influence analyses support it; this still does not prove a causal site
  artifact.

If none of these controls distinguishes the explanations, the correct sentence
is that the mechanism remains unresolved. That is preferable to a confident
post-hoc story.
