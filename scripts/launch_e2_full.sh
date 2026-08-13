#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/home/ubuntu/bunn-abide-parallel-c246744}"
RUN_ID="e2_synthetic_full_v2"
OUTPUT_ROOT="outputs/extensions/e2_synthetic"
RUN_DIR="${PROJECT_ROOT}/${OUTPUT_ROOT}/${RUN_ID}"
SNS_TOPIC="arn:aws:sns:us-east-1:020529562621:bunn-abide-run-alerts"

cd "${PROJECT_ROOT}"
mkdir -p "${RUN_DIR}"
if [[ -f "${RUN_DIR}/runner.pid" ]]; then
  existing_pid="$(tr -d '[:space:]' < "${RUN_DIR}/runner.pid")"
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "E2 full runner is already active with PID ${existing_pid}."
    exit 0
  fi
fi

nohup env PYTHONUNBUFFERED=1 .venv/bin/python scripts/run_e2_synthetic.py \
  --mode full \
  --run-id "${RUN_ID}" \
  --output-root "${OUTPUT_ROOT}" \
  --device cuda \
  --resume \
  --notification-topic-arn "${SNS_TOPIC}" \
  --require-notification \
  > "${RUN_DIR}/runner.log" 2>&1 < /dev/null &

runner_pid=$!
printf '%s\n' "${runner_pid}" > "${RUN_DIR}/runner.pid"
echo "E2 full runner started with PID ${runner_pid}."
