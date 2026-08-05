# AI assistance log

This is a local, version-controlled provenance record for AI-assisted work on
this project. It records assistance, not authorship: every technical claim,
implementation decision, experiment, and conclusion must be independently
reviewed and approved by the researcher.

## Maintenance rule

Codex will append an entry after each AI-assisted task it performs for this
repository. Assistance from tools Codex cannot observe (for example Cursor,
Claude, or another chat) must be supplied by the researcher with its date,
tool, and purpose before it can be logged accurately.

## Entries

| Date | Tool | Assistance provided | Researcher verification required | Evidence |
| --- | --- | --- | --- | --- |
| 2026-07-31 | Codex | Audited and narrowed the submitted BuNN/connectome proposal into an ABIDE-I operator-comparison study; identified claim and evaluation risks. | Verify all literature, protocol choices, and final scientific wording. | Prior Codex research-planning chats retained by the researcher. |
| 2026-08-03 | Codex | Produced an implementation plan; created the private repository and initial reproducibility scaffold. | Review the proposed project structure and decide the final protocol before experiments begin. | Git commit `6479bad`. |
| 2026-08-03 | Codex | Guided AWS EC2 GPU instance setup and SSH troubleshooting for the BuNN experiment environment. | Confirm the final active instance configuration, cost controls, and that credentials are kept private. | User confirmed SSH access to Ubuntu instance succeeded. |
| 2026-08-03 | Codex | Built and tested the ABIDE-I PCP manifest tool; compared QC subsets, verified all selected derivative URLs, and recorded the frozen primary cohort and exclusions. | Review the primary QC/site-eligibility decision before beginning data download and model training. | `scripts/prepare_abide_manifest.py`; `docs/step5_abide_manifest.md`; `configs/abide_i_primary_manifest.json`. |
| 2026-08-03 | Codex | Connected to the user-authorized AWS instance, checked storage and GPU-driver readiness, created a resumable public ABIDE-I ROI time-series downloader, and installed/validated the frozen cohort. | Confirm the retained AWS download summary before analysis; install and verify the NVIDIA driver before training. | `scripts/download_abide_timeseries.py`; 769 manifest-matched files, 0 malformed files. |
| 2026-08-03 | Codex | Installed the Ubuntu AWS NVIDIA driver/kernel module, rebooted the authorized instance, created an isolated PyTorch environment, and verified CUDA computation on the A10G. | Re-run the documented health checks if the instance image, kernel, or driver changes. | `docs/aws_environment.md`; `nvidia-smi`; PyTorch CUDA matrix-multiplication test. |
| 2026-08-03 | Codex | Implemented and executed ABIDE-I technical QC and Fisher-z connectome construction; identified 15 zero-variance-ROI files, created a separate analysis manifest, and validated the stored artifacts. | Review the technical eligibility rule and exclusions before interpreting or modeling the data. | `scripts/build_abide_connectomes.py`; `scripts/filter_abide_technical_qc.py`; `docs/step6_connectomes.md`. |
| 2026-08-03 | Codex | Froze the Step 7 baseline protocol, built the aligned baseline table, generated grouped site-held-out assignments, and independently audited their leakage and reproducibility properties. | Review the frozen protocol before fitting any baseline; no performance results exist yet. | `configs/baseline_protocol.json`; `configs/baseline_inputs_and_splits.json`; `docs/step7_baselines.md`. |
| 2026-08-03 | Codex | Implemented, tested, and ran the Step 7.3 baseline runner and Step 7.4 AWS engineering smoke test; recorded two implementation/runtime failures, their fixes, the smoke override, and artifact audit. | Treat the smoke artifacts as engineering evidence only. Run the full 18-site baseline evaluation without `--fast-smoke` before making predictive claims. | Commits `7dab13f`, `60c9bef`, `4654ee9`, `cd0bee8`; `scripts/run_baselines.py`; `docs/step7_baselines.md`; `outputs/runs/baselines/step7_4_fast_smoke_nyu_caltech/`. |
| 2026-08-03 | Codex | Implemented and AWS-tested atomic per-site checkpoints, immutable-contract resume, live status reading, detached launch control, and optional SNS notification hooks. | SNS creation/publish was blocked because the EC2 instance role supplied no usable credentials. Attach a least-privilege SNS role and confirm one emailed test alert before a full run. | Commit `a45e924`; `docs/step7_run_control.md`; run `checkpoint_resume_smoke`. |
| 2026-08-03 | Codex | Verified the user-created least-privilege EC2 SNS role, published a harmless test alert, configured the ignored owner-only alert file, and made managed runs fail closed when the alert configuration is absent. | Researcher confirmed that the test email reached the subscribed inbox. | Commit `7fdcf04`; `scripts/test_sns_alert.py`; `scripts/launch_managed_baseline.sh`; `docs/step7_run_control.md`. |
| 2026-08-05 | Codex | Reconciled the local repository, scientific protocol, and prior AWS handoff before continuation. | The prior public SSH endpoint did not respond; verify the instance's current state and public address in EC2 before any new remote action. | Local Git history through `7fdcf04`; current continuation chat. |
| 2026-08-05 | Codex | With the researcher's explicit authorization, added two SSH-only `/32` security-group rules: one for the researcher's current browser network and one for the authorized execution environment. Reconnected, completed the AWS preflight, normalized an ignored alert-file line-ending issue, and published one SNS test alert. | Confirm the test email arrives. No model-training process was started. | EC2 security group `sg-0f43fe350c37bbcfb`; `nvidia-smi`; PyTorch CUDA smoke; `scripts/check_baseline_run.py`; SNS message ID retained in the continuation chat. |
| 2026-08-05 | Codex | Audited the frozen Step 7.5 workload and existing AWS recovery/alert evidence, then wrote the prelaunch, monitoring, recovery, integrity, post-baseline, neural-model, statistical-analysis, and paper execution plans. | Review and explicitly authorize the Step 7.5 contract before launch; the estimates and later neural design remain plans until implemented and tested. | `docs/step7_5_full_baseline_plan.md`; `docs/research_execution_plan.md`. |
| 2026-08-05 | Codex | With explicit authorization, repeated AWS preflight, attempted the full Step 7.5 launch, and diagnosed a pre-fit SNS `NoRegionError`. Implemented a topic-ARN-derived SNS region fix, added a regression test, deployed it to AWS, and ran the intended test suite successfully. | `v1` produced no fit, prediction, or metric and must not be interpreted or resumed. Confirm the separately identified `v2` run starts before treating any work as active. | Commit `7ad7246`; AWS run `step7_5_full_baselines_v1`; `tests/test_run_baselines.py` (13 passed on AWS). |
| 2026-08-05 | Codex | Launched the authorized full 18-site Step 7.5 baseline evaluation after preflight and confirmed the detached runner, required STARTED alert, and initial full-grid tuning state. | Do not inspect test metrics until the documented completion integrity audit passes. On a terminal email, re-check status, manifests, hashes, and row coverage before analysis. | AWS run `step7_5_full_baselines_v2`; started `2026-08-05T08:01:22Z`; status message ID retained in the continuation chat. |

## Required fields for future entries

- Date and tool/version where known
- Specific task performed
- Files, code, results, or decisions affected
- What was independently checked
- Link to retained chat, prompt, or run artifact when available
