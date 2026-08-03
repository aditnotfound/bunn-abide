# Step 7.4.1: Durable Run Control

This layer strengthens execution reliability without changing the frozen data,
folds, models, hyperparameter grids, metrics, or interpretation rules.

## Components

- `scripts/run_baselines.py` now writes a hash-verified, atomic checkpoint
  directory after each held-out site. A site is never treated as complete until
  all of its predictions, metrics, tuning records, inner-site scores, warnings,
  and `complete.json` are present.
- `--resume` verifies the original run's immutable metadata (dataset/split
  hashes, protocol hash, code version, sources, selected models, sites, and
  overrides) before skipping only valid completed sites. Any incomplete site is
  rerun from the beginning.
- `status.json` is updated at site start, every tuning candidate, outer final
  fit, checkpoint, interruption, and completion. It is atomically replaced so
  a status reader never sees half-written JSON.
- `scripts/check_baseline_run.py --run-id RUN_ID` is a read-only status command.
  It reports progress, current site/model/stage, heartbeat age, PID liveness,
  and notification state. It labels a live run `stalled` only after a chosen
  heartbeat threshold.
- `scripts/launch_managed_baseline.sh` starts the runner detached from SSH,
  records a PID and log, and uses a lock to prevent duplicate managed launches.
- `scripts/setup_sns_alerts.py` can create a standard SNS topic plus a pending
  email subscription using an EC2 instance role. The runner can optionally
  publish STARTED, COMPLETE, FAILED, and INTERRUPTED alerts using only a topic
  ARN; it never stores access keys in the repository.

## AWS live recovery test

Run `checkpoint_resume_smoke` used the existing two-site `--fast-smoke`
engineering configuration only. It did not alter or replace the future full
baseline evaluation.

1. The detached manager started NYU and recorded live status.
2. NYU completed and was atomically sealed with a verified `complete.json`.
3. The process was intentionally terminated during CALTECH.
4. The root status became `interrupted`; NYU remained valid and CALTECH was not
   falsely marked complete.
5. The same run ID was resumed. It verified and skipped NYU, executed only
   CALTECH, then completed the final two-site artifact audit.

The source artifacts are retained under
`outputs/runs/baselines/checkpoint_resume_smoke/` on both AWS and the local
ignored results directory. They are engineering evidence only, not paper
results.

## Notification status

The AWS instance has no usable role credentials available to boto3, so no SNS
topic, email subscription, or test email has been created. The runner records
this accurately as `notification.status = "not_configured"`; it never claims a
message was sent.

To enable alerts securely, attach an EC2 instance profile that permits the
runner role to publish only to the chosen SNS topic. Create the topic and
subscribe/confirm the private email endpoint through AWS, then expose only its
topic ARN to the instance via `BUNN_SNS_TOPIC_ARN`. After that, run the setup
and publish test (`scripts/test_sns_alert.py`) before any full baseline job.

## Before the full run

1. Repair or attach the least-privilege AWS instance role for SNS.
2. Create and confirm the SNS email subscription; publish and receive one test
   message.
3. Start the full run without `--fast-smoke`, with an explicit run ID and
   `--require-notification`.
4. Treat `complete.json` files and the final all-site integrity audit as the
   source of truth; use email only as an alert to inspect them.
