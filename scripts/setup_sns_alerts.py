"""Create an SNS topic and pending email subscription for private run alerts.

This script never accepts, stores, or prints AWS access keys. It relies on the
EC2 instance role or another standard AWS credential provider available to
boto3. The topic ARN can be kept in a private environment variable rather than
committed to the repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True)
    parser.add_argument("--topic-name", default="bunn-abide-run-alerts")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument(
        "--private-config",
        default=".run-control/sns-alerts.json",
        help="Ignored local config path; do not commit this file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if "@" not in args.email or args.email.startswith("@"):
        raise ValueError("--email must be a valid-looking email address")
    import boto3

    client = boto3.client("sns", region_name=args.region)
    topic_arn = client.create_topic(Name=args.topic_name)["TopicArn"]
    subscription = client.subscribe(
        TopicArn=topic_arn,
        Protocol="email",
        Endpoint=args.email,
        ReturnSubscriptionArn=True,
    )
    config = {
        "region": args.region,
        "topic_arn": topic_arn,
        "subscription_arn": subscription.get("SubscriptionArn"),
        "email_confirmation_required": True,
    }
    path = Path(args.private_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"topic_arn": topic_arn, "subscription": "confirmation email sent"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
