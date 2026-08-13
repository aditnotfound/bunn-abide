# E2 pre-result implementation amendment v2

The first full launcher (`e2_synthetic_full_v1`) was stopped on 2026-08-14
before any test probability or metric was opened. Review of the generator found
that S0 constructed zero-angle maps through the direct O(2) parameterization.
That parameterization intentionally assigns reflections to half the bundles, so
zero angles did not produce identity maps in every bundle. This violated the
pre-specified role of S0 as the no-geometry condition.

The v1 run is quarantined and must never enter analysis. Version 2 replaces the
S0 maps with explicit identity matrices for every node, participant, and
bundle. A regression test asserts exact equality to identity. The scientific
question, families, samples, operators, primary estimand, inference, and success
rules are unchanged. The corrected full run starts from scratch under a new run
identifier, `e2_synthetic_full_v2`; no v1 checkpoint or prediction is reused.
