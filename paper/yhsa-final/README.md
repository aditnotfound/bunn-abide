# Final YHSA-format evidence report

This directory contains the release-candidate A4, one-column rendering of
the frozen ABIDE-I operator study, the separately labelled E1
accepted-checkpoint intervention audit, and the E2 synthetic known-geometry
study. It was copied from `paper/yhsa-e2` and revised independently. It does
not replace the IEEE-style manuscript or any earlier report.

E1, E2, and the nonlinear connectome comparator were specified after Study 1
and are therefore presented as post-hoc evidence rather than independent
confirmation. The stopped E2 v1
run is disclosed as a pre-result implementation correction; only the fully
restarted and audited v2 run is reported. Wang-style, five-layer, and ABIDE-II
studies remain outside the completed evidence.

The final version adds a protocol-aware comparison with prior ABIDE graph
studies, corrects the evidence-boundary wording, and reports the failed
score-blind ABIDE-II derivative-compatibility gate. No ABIDE-II model was fitted
or scored. The integrity page still contains fields that the
researcher must review, complete, and sign before any actual competition
submission.

The 14 August editorial pass reviewed two user-supplied YHSA reports for broad
structural conventions, then rewrote this report independently. The revised
order is abstract, contents, introduction, related work, methods, the complete
Study 1 result, E1, E2, discussion, conclusion, acknowledgments and research
integrity material, appendices, and references. The pass removed formulaic
prose and repeated caveats without changing accepted scientific values,
citations, or the confirmatory and post-hoc boundary.

The complete Study 1 hyperparameter table is generated from the frozen model
contracts. The weighting-sensitivity table is generated from the frozen result
snapshot. Run `python scripts/build_final_report_tables.py` from the repository
root before compiling the report.

The 15 August update adds the audited RBF-SVM comparator. It uses the same 754
participants, vectorized connectomes, and nested held-out-site splits as the
classical baselines. Its result cannot revise the confirmatory Study 1
decision.
