#!/usr/bin/env bash
# Launch the runner detached from SSH while retaining a PID, lock, and log.
set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 RUN_ID [run_baselines.py arguments...]" >&2
  exit 64
fi

RUN_ID="$1"
shift
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTROL_DIR="$PROJECT_DIR/outputs/runs/baselines/.control/$RUN_ID"
LOG_FILE="$CONTROL_DIR/runner.log"
PID_FILE="$CONTROL_DIR/pid"
mkdir -p "$CONTROL_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "A managed process for $RUN_ID is already running (PID $(cat "$PID_FILE"))." >&2
  exit 1
fi

(
  exec 9>"$CONTROL_DIR/lock"
  flock -n 9 || { echo "Run lock is already held for $RUN_ID." >&2; exit 1; }
  echo "$$" > "$PID_FILE"
  printf 'started_utc=%s\n' "$(date --utc --iso-8601=seconds)"
  exec "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/run_baselines.py" \
    --run-id "$RUN_ID" "$@"
) >> "$LOG_FILE" 2>&1 < /dev/null &

PID="$!"
echo "$PID" > "$PID_FILE"
echo "Launched $RUN_ID as PID $PID"
echo "Log: $LOG_FILE"
