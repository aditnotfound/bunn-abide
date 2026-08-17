# YHSA-format revision draft

This directory is a non-destructive revision of `paper/yhsa-final`. It retains
the accepted ABIDE-I operator audit, the post-hoc E1 checkpoint interventions,
the controlled E2 synthetic study, and the post-hoc RBF-SVM comparator. It does
not alter the earlier report or the IEEE manuscript.

The 15 August revision adds analyses that required no model retraining:

- the complete participant and quality-control cascade;
- equal-site and participant-weighted site-cluster bootstrap intervals;
- a uniform-density-grid sensitivity check;
- all six E1 intervention comparisons with Holm correction; and
- graph-spectrum summaries that quantify how strongly the fixed heat operator
  attenuates modes at each density.

It also states the exact BuNN map-generator and heat-operator settings, removes
the decorative workflow diagram, separates confirmatory results from post-hoc
extensions, and narrows every conclusion to what the frozen analyses support.
No accepted model was retrained and no accepted prediction was changed.

The mechanical consistency audit is implemented in
`scripts/audit_yhsa_submission.py`.

Build with the project LaTeX workflow after regenerating the evidence-bound
tables. The reviewed PDF is written to
`output/pdf/yhsa-submission/Comp-183-Research Report.pdf`.
