"""Publish one harmless SNS delivery test after the email subscription is confirmed."""

from __future__ import annotations

import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic-arn", required=True)
    parser.add_argument("--region", default="us-east-1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import boto3

    response = boto3.client("sns", region_name=args.region).publish(
        TopicArn=args.topic_arn,
        Subject="BuNN run-alert test",
        Message="This is a one-time test. Future messages report only baseline-run state and run ID.",
    )
    print(json.dumps({"status": "published", "message_id": response["MessageId"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
