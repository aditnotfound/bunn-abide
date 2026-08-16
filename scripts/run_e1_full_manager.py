"""Resumable, score-blind manager for all 18 E1 held-out sites."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_baselines import publish_sns_notification, sha256_file, write_json_atomic


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="e1_full_v1")
    parser.add_argument("--inputs", type=Path, default=Path("outputs/extensions/e1_interventions_v1/inputs"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/extensions/e1_interventions_v1/runs"))
    parser.add_argument("--notification-topic-arn", default=os.environ.get("BUNN_SNS_TOPIC_ARN"))
    parser.add_argument("--require-notification", action="store_true")
    return parser.parse_args()


def update_status(manager_dir: Path, **values: object) -> None:
    path = manager_dir / "status.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    current.update(values)
    current["updated_utc"] = utc_now()
    write_json_atomic(path, current)


def run(args: argparse.Namespace) -> Path:
    manifest_path = args.inputs / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sites = sorted(str(site) for site in manifest["sites"])
    if len(sites) != 18 or len(set(sites)) != 18:
        raise ValueError("Full manager requires the exact 18-site checkpoint manifest")
    manager_dir = args.output_root / args.run_id
    manager_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_id": args.run_id, "created_utc": utc_now(), "sites": sites,
        "input_manifest_sha256": sha256_file(manifest_path), "results_embargoed": True,
        "score_blind": True,
    }
    metadata_path = manager_dir / "manager_metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("run_id", "sites", "input_manifest_sha256", "results_embargoed", "score_blind"):
            if previous.get(key) != metadata[key]:
                raise ValueError("Manager resume contract changed")
    else:
        write_json_atomic(metadata_path, metadata)

    if args.require_notification:
        alert = publish_sns_notification(
            manager_dir, args.notification_topic_arn, f"BuNN E1 FULL STARTED: {args.run_id}",
            f"Score-blind E1 full intervention audit started across {len(sites)} sites. Results remain embargoed.",
        )
        if alert.get("status") != "published":
            raise RuntimeError("Required SNS start notification failed")

    completed: list[str] = []
    try:
        for site_index, site in enumerate(sites):
            site_run_id = f"{args.run_id}__{site}"
            site_run_dir = args.output_root / site_run_id
            audit_path = site_run_dir / "score_blind_audit.json"
            if audit_path.exists() and json.loads(audit_path.read_text(encoding="utf-8")).get("state") == "passed":
                completed.append(site)
                continue
            update_status(
                manager_dir, state="running", current_site=site, site_index=site_index + 1,
                total_sites=len(sites), completed_sites=completed, completed_site_count=len(completed),
            )
            if not (site_run_dir / "complete.json").exists():
                command = [
                    sys.executable, "scripts/run_e1_interventions.py", "--site", site,
                    "--run-id", site_run_id, "--inputs", str(args.inputs),
                    "--output-root", str(args.output_root),
                ]
                if (site_run_dir / "metadata.json").exists():
                    command.append("--resume")
                subprocess.run(command, check=True)
            subprocess.run(
                [
                    sys.executable, "scripts/audit_e1_interventions.py",
                    "--run-dir", str(site_run_dir),
                ],
                check=True,
            )
            completed.append(site)
            update_status(
                manager_dir, state="running", current_site=None,
                completed_sites=completed, completed_site_count=len(completed),
            )

        audit_hashes = {
            site: sha256_file(args.output_root / f"{args.run_id}__{site}" / "score_blind_audit.json")
            for site in sites
        }
        completion = {
            "state": "complete_all_sites_score_blind_audited", "completed_utc": utc_now(),
            "site_count": len(sites), "sites": sites, "audit_hashes": audit_hashes,
            "results_embargoed": True, "scientific_values_displayed": False,
        }
        write_json_atomic(manager_dir / "manager_complete.json", completion)
        update_status(
            manager_dir, state=completion["state"], current_site=None,
            completed_sites=completed, completed_site_count=len(completed),
        )
        if args.require_notification:
            publish_sns_notification(
                manager_dir, args.notification_topic_arn, f"BuNN E1 FULL COMPLETE: {args.run_id}",
                f"All {len(sites)} E1 sites completed and passed score-blind artifact audits. Results may now be analyzed.",
            )
        print(json.dumps({"state": completion["state"], "site_count": len(sites), "results_displayed": False}))
        return manager_dir
    except Exception as error:
        update_status(manager_dir, state="failed", error_type=type(error).__name__, error=str(error))
        if args.require_notification:
            publish_sns_notification(
                manager_dir, args.notification_topic_arn, f"BuNN E1 FULL FAILED: {args.run_id}",
                f"E1 full manager failed after {len(completed)} audited sites. Resume from the sealed site artifacts.",
            )
        raise


if __name__ == "__main__":
    run(parse_args())
