#!/usr/bin/env bash
set -euo pipefail

project_dir=/home/ubuntu/bunn-abide-parallel-c246744
alert_config=/home/ubuntu/bunn-abide/.run-control/run-alerts.env
run_id=e1_full_v1
control_dir="$project_dir/outputs/extensions/e1_interventions_v1/control"
log_path="$control_dir/$run_id.log"
pid_path="$control_dir/$run_id.pid"

mkdir -p "$control_dir"
if [[ -f "$pid_path" ]]; then
  existing_pid="$(<"$pid_path")"
  if kill -0 "$existing_pid" 2>/dev/null; then
    echo "E1 full manager is already running with PID $existing_pid"
    exit 1
  fi
fi

set -a
source "$alert_config"
set +a
if [[ -z "${BUNN_SNS_TOPIC_ARN:-}" ]]; then
  echo "BUNN_SNS_TOPIC_ARN is not configured"
  exit 1
fi

cd "$project_dir"
nohup .venv/bin/python scripts/run_e1_full_manager.py \
  --run-id "$run_id" \
  --require-notification \
  >"$log_path" 2>&1 </dev/null &
manager_pid=$!
echo "$manager_pid" >"$pid_path"
sleep 2
if ! kill -0 "$manager_pid" 2>/dev/null; then
  echo "E1 full manager exited during launch; inspect $log_path"
  exit 1
fi
echo "E1 full manager launched with PID $manager_pid"
echo "Status: $project_dir/outputs/extensions/e1_interventions_v1/runs/$run_id/status.json"
echo "Log: $log_path"
