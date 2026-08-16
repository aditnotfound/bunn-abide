#!/usr/bin/env bash
# Launch the site-parallel neural coordinator detached from SSH with SNS alerts.
set -Eeuo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 RUN_ID RUN_KIND [run_neural_full_parallel.py arguments...]" >&2
  exit 64
fi

RUN_ID="$1"
RUN_KIND="$2"
shift 2
if [[ "$RUN_KIND" != "smoke" && "$RUN_KIND" != "full" ]]; then
  echo "RUN_KIND must be smoke or full" >&2
  exit 64
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALERT_CONFIG="$PROJECT_DIR/.run-control/run-alerts.env"
CONTROL_DIR="$PROJECT_DIR/outputs/runs/neural-full-parallel/.control/$RUN_ID"
LOG_FILE="$CONTROL_DIR/coordinator.log"
PID_FILE="$CONTROL_DIR/pid"
mkdir -p "$CONTROL_DIR"

if [[ ! -r "$ALERT_CONFIG" ]]; then
  echo "Missing required alert configuration: $ALERT_CONFIG" >&2
  exit 78
fi
# shellcheck source=/dev/null
source "$ALERT_CONFIG"
if [[ -z "${BUNN_SNS_TOPIC_ARN:-}" ]]; then
  echo "BUNN_SNS_TOPIC_ARN is missing from $ALERT_CONFIG" >&2
  exit 78
fi
for REQUIRED_PATH in \
  "$PROJECT_DIR/configs/neural_full_protocol.json" \
  "$PROJECT_DIR/configs/neural_operator_contract_v2.json" \
  "$PROJECT_DIR/configs/neural_analysis_protocol.json" \
  "$PROJECT_DIR/configs/neural_parallel_execution.json" \
  "$PROJECT_DIR/configs/baseline_inputs_and_splits.json" \
  "$PROJECT_DIR/data/processed/abide_i_baseline_table.csv" \
  "$PROJECT_DIR/data/processed/abide_i_connectomes_fisher_z.npz" \
  "$PROJECT_DIR/data/processed/splits/outer_loso_assignments.csv" \
  "$PROJECT_DIR/data/processed/splits/inner_grouped_assignments.csv"; do
  if [[ ! -r "$REQUIRED_PATH" ]]; then
    echo "Missing required parallel-run input: $REQUIRED_PATH" >&2
    exit 78
  fi
done
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "A managed process for $RUN_ID is already running (PID $(cat "$PID_FILE"))." >&2
  exit 1
fi

(
  exec 9>"$CONTROL_DIR/lock"
  flock -n 9 || { echo "Run lock is already held for $RUN_ID." >&2; exit 1; }
  printf 'started_utc=%s\n' "$(date --utc --iso-8601=seconds)"
  cd "$PROJECT_DIR"
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  exec "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/run_neural_full_parallel.py" \
    --run-id "$RUN_ID" --run-kind "$RUN_KIND" --require-notification \
    --notification-topic-arn "$BUNN_SNS_TOPIC_ARN" "$@"
) >> "$LOG_FILE" 2>&1 < /dev/null &

PID="$!"
echo "$PID" > "$PID_FILE"
echo "Launched parallel run $RUN_ID as coordinator PID $PID"
echo "Log: $LOG_FILE"
