# Experiment log

| Run ID | Date | Commit | Dataset manifest | Model | Density | Held-out site | Seed | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `step7_4_smoke_nyu_caltech` | 2026-08-03 | `60c9bef` | `bef94eb5...` | All three baselines | N/A | NYU, CALTECH | 20260803 | Failed before predictions | Covariate-only tuning exposed an empty-`l1_ratio` tie-break bug. Fixed with a regression test; no test metric was produced. |
| `step7_4_smoke_nyu_caltech_retry1` | 2026-08-03 | `4654ee9` | `bef94eb5...` | All three baselines | N/A | NYU, CALTECH | 20260803 | Interrupted before predictions | Full-grid interactive smoke exceeded the appropriate cost/runtime budget on four vCPUs. No scientific protocol was altered. |
| `step7_4_fast_smoke_nyu_caltech` | 2026-08-03 | `cd0bee8` | `bef94eb5...` | All three baselines | N/A | NYU, CALTECH | 20260803 | Engineering smoke passed | Four inner grouped folds retained. Explicit `--fast-smoke` override: one lowest candidate/model, elastic-net max 100 iterations, no retry. 474 predictions and six finite metric rows saved; not a scientific result. |
| `checkpoint_resume_smoke` | 2026-08-03 | `a45e924` | `bef94eb5...` | All three baselines | N/A | NYU, CALTECH | 20260803 | Recovery test passed | NYU sealed; process deliberately interrupted during CALTECH; resume verified/skipped NYU and completed CALTECH. Explicit `--fast-smoke`; engineering evidence only. SNS was not configured because the instance role credentials were unavailable. |
