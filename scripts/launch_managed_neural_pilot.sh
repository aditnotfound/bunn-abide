#!/usr/bin/env bash
# Launch the neural engineering pilot detached from SSH with mandatory SNS alerts.
set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 RUN_ID [run_neural_pilot.py arguments...]" >&2
  exit 64
fi

RUN_ID="$1"
shift
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALERT_CONFIG="$PROJECT_DIR/.run-control/run-alerts.env"
CONTROL_DIR="$PROJECT_DIR/outputs/runs/neural/.control/$RUN_ID"
LOG_FILE="$CONTROL_DIR/runner.log"
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
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "A managed neural pilot for $RUN_ID is already running (PID $(cat "$PID_FILE"))." >&2
  exit 1
fi

(
  exec 9>"$CONTROL_DIR/lock"
  flock -n 9 || { echo "Run lock is already held for $RUN_ID." >&2; exit 1; }
  printf 'started_utc=%s\n' "$(date --utc --iso-8601=seconds)"
  exec "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/run_neural_pilot.py" \
    --run-id "$RUN_ID" --require-notification \
    --notification-topic-arn "$BUNN_SNS_TOPIC_ARN" "$@"
) >> "$LOG_FILE" 2>&1 < /dev/null &

PID="$!"
echo "$PID" > "$PID_FILE"
echo "Launched $RUN_ID as PID $PID"
echo "Log: $LOG_FILE"
