"""Create and validate frozen outer LOSO and inner grouped-CV assignments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table", default="data/processed/abide_i_baseline_table.csv"
    )
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--outer-output", default="data/processed/splits/outer_loso_assignments.csv"
    )
    parser.add_argument(
        "--inner-output", default="data/processed/splits/inner_grouped_assignments.csv"
    )
    parser.add_argument(
        "--summary-output", default="data/processed/splits/split_summary.json"
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"connectome_row", "subject_id", "site_id", "label_asd"}
    if not rows:
        raise ValueError("Baseline table has no rows")
    if missing := required - set(rows[0]):
        raise ValueError(f"Baseline table lacks columns: {sorted(missing)}")
    if len({row["subject_id"] for row in rows}) != len(rows):
        raise ValueError("Baseline table has duplicate subject IDs")
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def class_counts(labels: np.ndarray) -> dict[str, int]:
    return {"asd": int(labels.sum()), "control": int((labels == 0).sum())}


def create_assignments(
    rows: list[dict[str, str]], inner_folds: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    labels = np.asarray([int(row["label_asd"]) for row in rows], dtype=np.int8)
    sites = np.asarray([row["site_id"] for row in rows])
    subject_ids = np.asarray([row["subject_id"] for row in rows])
    unique_sites = sorted(set(sites))
    if len(unique_sites) < inner_folds + 1:
        raise ValueError("Not enough sites for outer LOSO plus grouped inner CV")

    outer_assignments: list[dict[str, Any]] = []
    inner_assignments: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []

    for outer_fold, held_out_site in enumerate(unique_sites):
        test_mask = sites == held_out_site
        train_mask = ~test_mask
        test_indices = np.flatnonzero(test_mask)
        train_indices = np.flatnonzero(train_mask)
        test_counts = class_counts(labels[test_indices])
        if not test_counts["asd"] or not test_counts["control"]:
            raise ValueError(f"Outer held-out site {held_out_site} lacks a class")
        for index in test_indices:
            outer_assignments.append(
                {
                    "outer_fold": outer_fold,
                    "held_out_site": held_out_site,
                    "subject_id": subject_ids[index],
                    "site_id": sites[index],
                    "label_asd": int(labels[index]),
                    "role": "test",
                }
            )

        splitter = StratifiedGroupKFold(
            n_splits=inner_folds, shuffle=True, random_state=seed + outer_fold
        )
        inner_fold_for_train_index: dict[int, int] = {}
        inner_summary: list[dict[str, Any]] = []
        local_features = np.zeros((len(train_indices), 1), dtype=np.float32)
        for inner_fold, (inner_train_local, inner_val_local) in enumerate(
            splitter.split(local_features, labels[train_indices], sites[train_indices])
        ):
            inner_train_indices = train_indices[inner_train_local]
            inner_val_indices = train_indices[inner_val_local]
            train_sites = set(sites[inner_train_indices])
            val_sites = set(sites[inner_val_indices])
            if train_sites & val_sites:
                raise ValueError(
                    f"Outer fold {outer_fold}, inner fold {inner_fold} leaks a site"
                )
            val_counts = class_counts(labels[inner_val_indices])
            train_counts = class_counts(labels[inner_train_indices])
            if not val_counts["asd"] or not val_counts["control"]:
                raise ValueError(
                    f"Outer fold {outer_fold}, inner fold {inner_fold} validation lacks a class"
                )
            if not train_counts["asd"] or not train_counts["control"]:
                raise ValueError(
                    f"Outer fold {outer_fold}, inner fold {inner_fold} training lacks a class"
                )
            for index in inner_val_indices:
                if int(index) in inner_fold_for_train_index:
                    raise ValueError("A training participant appears in two inner validation folds")
                inner_fold_for_train_index[int(index)] = inner_fold
            inner_summary.append(
                {
                    "inner_fold": inner_fold,
                    "validation_sites": sorted(val_sites),
                    "validation_participants": int(len(inner_val_indices)),
                    "validation_class_counts": val_counts,
                    "training_participants": int(len(inner_train_indices)),
                    "training_class_counts": train_counts,
                }
            )
        if set(inner_fold_for_train_index) != set(map(int, train_indices)):
            raise ValueError(f"Outer fold {outer_fold} inner validation does not cover training rows")
        for index in train_indices:
            inner_assignments.append(
                {
                    "outer_fold": outer_fold,
                    "held_out_site": held_out_site,
                    "subject_id": subject_ids[index],
                    "site_id": sites[index],
                    "label_asd": int(labels[index]),
                    "inner_validation_fold": inner_fold_for_train_index[int(index)],
                }
            )
        fold_summaries.append(
            {
                "outer_fold": outer_fold,
                "held_out_site": held_out_site,
                "test_participants": int(len(test_indices)),
                "test_class_counts": test_counts,
                "training_participants": int(len(train_indices)),
                "training_class_counts": class_counts(labels[train_indices]),
                "inner_folds": inner_summary,
            }
        )

    if len(outer_assignments) != len(rows):
        raise ValueError("Outer test assignments do not cover every participant exactly once")
    if len({row["subject_id"] for row in outer_assignments}) != len(rows):
        raise ValueError("A participant appears in multiple outer test assignments")
    return outer_assignments, inner_assignments, fold_summaries


def main() -> int:
    args = parse_args()
    table_path = Path(args.table)
    outer_output = Path(args.outer_output)
    inner_output = Path(args.inner_output)
    summary_output = Path(args.summary_output)
    rows = read_table(table_path)
    outer_rows, inner_rows, fold_summaries = create_assignments(
        rows, args.inner_folds, args.seed
    )
    write_csv(
        outer_output,
        ["outer_fold", "held_out_site", "subject_id", "site_id", "label_asd", "role"],
        outer_rows,
    )
    write_csv(
        inner_output,
        [
            "outer_fold",
            "held_out_site",
            "subject_id",
            "site_id",
            "label_asd",
            "inner_validation_fold",
        ],
        inner_rows,
    )
    summary = {
        "baseline_table": str(table_path),
        "baseline_table_sha256": sha256_file(table_path),
        "seed": args.seed,
        "inner_folds": args.inner_folds,
        "participants": len(rows),
        "sites": len(fold_summaries),
        "outer_assignments": str(outer_output),
        "outer_assignments_sha256": sha256_file(outer_output),
        "inner_assignments": str(inner_output),
        "inner_assignments_sha256": sha256_file(inner_output),
        "outer_folds": fold_summaries,
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
