# Post-hoc nonlinear connectome baseline results

## Evidence status

The full `rbf_svm_full_v1` run completed all 18 frozen outer sites. Its remote
and local score-blind audits passed before any metric was opened. The sealed
archive SHA-256 is
`a9bc9a5f2758591aaa7e5f0bb53ffcf7d9b0f970c26d56b92055267cb5fea477`.
The analysis contract was committed at `ed8e0de` before unblinding.

This is a post-hoc practical comparator. It cannot revise the confirmatory
Study 1 decision.

## RBF-SVM performance

The RBF-SVM reached:

- equal-site balanced accuracy: 0.6255;
- pooled balanced accuracy: 0.6565;
- pooled AUROC: 0.7014;
- sensitivity: 0.6577;
- specificity: 0.6554.

## Paired site comparisons

RBF-SVM minus connectome elastic net was -0.0146 under equal-site averaging
(95% paired site-bootstrap interval [-0.0555, 0.0206]; exact sign-flip
`p=0.498`). The analysis did not detect an RBF-SVM advantage over the strongest
classical baseline.

RBF-SVM minus the GCN density curve was +0.0310 (95% interval
[-0.0156, 0.0705]; exact `p=0.193`). RBF-SVM minus the learned-BuNN curve was
+0.0406 (95% interval [+0.0069, +0.0744]; exact `p=0.037`). The latter is a
post-hoc comparison and is not a new confirmatory endpoint.

## Weighting sensitivity

The RBF-SVM versus elastic-net ordering changed with the aggregation rule.
RBF-SVM was 0.0146 below elastic net under equal-site averaging but 0.0089
above it under participant weighting. The median paired site difference was
zero. Pooled performance therefore gives a different ordering because large
sites receive more influence.

## Interpretation

The nonlinear comparator does not explain the Study 1 result by itself. It
does rule out a simple claim that the linear baseline won only because no
nonlinear whole-connectome model was tested. Under the pre-specified equal-site
estimand, elastic net remained the strongest practical reference. RBF-SVM did
better than learned BuNN, which supports the narrower conclusion that adding
nonlinearity without graph propagation was still insufficient to beat the
regularized linear baseline.
