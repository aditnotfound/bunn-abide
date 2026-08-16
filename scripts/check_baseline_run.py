"""Read a durable baseline-run status without touching the experiment itself."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default="outputs/runs/baselines")
    parser.add_argument(
        "--stale-minutes",
        type=float,
        default=45.0,
        help="Mark a running status stale when its heartbeat is older than this value.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def status_report(run_dir: Path, stale_minutes: float) -> dict[str, Any]:
    status_path = run_dir / "status.json"
    metadata_path = run_dir / "metadata.json"
    if not status_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Run is missing status.json or metadata.json")
    status = read_json(status_path)
    metadata = read_json(metadata_path)
    updated = datetime.fromisoformat(status["last_updated_utc"])
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=UTC)
    age_seconds = max(0.0, (datetime.now(UTC) - updated.astimezone(UTC)).total_seconds())
    state = status.get("state", "unknown")
    stale = state == "running" and age_seconds > stale_minutes * 60
    notification_path = run_dir / "notification.json"
    return {
        "run_id": metadata["run_id"],
        "state": "stalled" if stale else state,
        "recorded_state": state,
        "stale": stale,
        "heartbeat_age_seconds": round(age_seconds, 1),
        "last_updated_utc": status["last_updated_utc"],
        "completed_site_count": status.get("completed_site_count", 0),
        "total_sites": status.get("total_sites"),
        "completed_sites": status.get("completed_sites", []),
        "current_site": status.get("current_site"),
        "current_model": status.get("current_model"),
        "current_stage": status.get("current_stage"),
        "current_candidate": status.get("current_candidate"),
        "notification": read_json(notification_path) if notification_path.exists() else None,
        "pid": status.get("pid"),
        "pid_alive": bool(status.get("pid") and os.path.exists(f"/proc/{status['pid']}")),
        "run_directory": str(run_dir),
    }


def main() -> int:
    args = parse_args()
    if args.stale_minutes <= 0:
        raise ValueError("--stale-minutes must be positive")
    report = status_report(Path(args.output_root) / args.run_id, args.stale_minutes)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"{report['run_id']}: {report['state']} | "
            f"{report['completed_site_count']}/{report['total_sites']} sites | "
            f"current={report['current_site']} / {report['current_model']} / {report['current_stage']} | "
            f"heartbeat_age={report['heartbeat_age_seconds']}s"
        )
    return 2 if report["state"] == "stalled" else 0


if __name__ == "__main__":
    raise SystemExit(main())
