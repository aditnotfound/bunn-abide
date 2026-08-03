"""Build and validate an ABIDE-I PCP ROI time-series manifest.

The manifest is the contract for every downstream experiment: rows, labels,
sites, covariates, derivative URLs, and the QC rule are made explicit before
model training starts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PHENOTYPIC_URL = (
    "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/"
    "Phenotypic_V1_0b_preprocessed1.csv"
)
S3_OUTPUT_ROOT = (
    "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs"
)

QC_RULES = {
    "none": lambda row: True,
    "sub_in_smp": lambda row: row.get("SUB_IN_SMP") == "1",
    "rater1_ok": lambda row: row.get("qc_rater_1") == "OK",
    "func2_ok": lambda row: row.get("qc_func_rater_2") == "OK",
    "func3_ok": lambda row: row.get("qc_func_rater_3") == "OK",
    "rater1_and_func2_ok": lambda row: (
        row.get("qc_rater_1") == "OK" and row.get("qc_func_rater_2") == "OK"
    ),
}

MANIFEST_COLUMNS = [
    "subject_id",
    "site_id",
    "file_id",
    "dx_group",
    "label_asd",
    "age_at_scan",
    "sex",
    "fiq",
    "func_mean_fd",
    "func_num_fd",
    "func_perc_fd",
    "func_quality",
    "sub_in_smp",
    "qc_rater_1",
    "qc_func_rater_2",
    "qc_func_rater_3",
    "pipeline",
    "strategy",
    "derivative",
    "timeseries_url",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phenotypic-url", default=PHENOTYPIC_URL)
    parser.add_argument(
        "--raw-csv",
        default="data/raw/Phenotypic_V1_0b_preprocessed1.csv",
        help="Local cache path for the PCP phenotypic/QC table.",
    )
    parser.add_argument("--pipeline", default="cpac")
    parser.add_argument("--strategy", default="filt_noglobal")
    parser.add_argument("--derivative", default="rois_aal")
    parser.add_argument(
        "--qc-rule",
        default="sub_in_smp",
        choices=sorted(QC_RULES),
        help="Participant inclusion rule to apply after FILE_ID availability.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/abide_i_manifest_cpac_filt_noglobal_rois_aal.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="data/processed/abide_i_manifest_summary.json",
    )
    parser.add_argument(
        "--check-urls",
        action="store_true",
        help="HEAD-check derivative URLs and exclude unavailable files.",
    )
    parser.add_argument(
        "--max-url-checks",
        type=int,
        default=None,
        help="Optional limit for quick URL smoke checks.",
    )
    parser.add_argument("--url-timeout", type=float, default=10.0)
    parser.add_argument("--url-workers", type=int, default=16)
    parser.add_argument(
        "--min-site-class-size",
        type=int,
        default=1,
        help=(
            "Require at least this many ASD and control participants in each site. "
            "Sites failing the rule are excluded entirely; set 0 to disable."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_if_needed(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def available_file(row: dict[str, str]) -> bool:
    file_id = row.get("FILE_ID", "").strip()
    return bool(file_id) and file_id != "no_filename"


def derivative_url(row: dict[str, str], pipeline: str, strategy: str, derivative: str) -> str:
    file_id = row["FILE_ID"].strip()
    return f"{S3_OUTPUT_ROOT}/{pipeline}/{strategy}/{derivative}/{file_id}_{derivative}.1D"


def url_exists(url: str, timeout: float) -> bool:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 400
    except (urllib.error.URLError, TimeoutError):
        return False


def filter_by_url_availability(
    rows: list[dict[str, str]],
    timeout: float,
    workers: int,
    max_checks: int | None,
) -> tuple[list[dict[str, str]], int, int]:
    rows_to_check = rows if max_checks is None else rows[:max_checks]
    rows_to_keep_without_check = [] if max_checks is None else rows[max_checks:]
    availability: dict[str, bool] = {}
    max_workers = max(1, workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(url_exists, row["timeseries_url"], timeout): row["timeseries_url"]
            for row in rows_to_check
        }
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                availability[url] = future.result()
            except Exception:
                availability[url] = False

    checked_and_available = [
        row for row in rows_to_check if availability.get(row["timeseries_url"], False)
    ]
    missing_count = len(rows_to_check) - len(checked_and_available)
    return checked_and_available + rows_to_keep_without_check, len(rows_to_check), missing_count


def filter_sites_by_class_count(
    rows: list[dict[str, str]], min_per_class: int
) -> tuple[list[dict[str, str]], dict[str, dict[str, int]]]:
    if min_per_class < 0:
        raise ValueError("min_site_class_size cannot be negative")
    if min_per_class == 0:
        return rows, {}

    site_counts = nested_counter(rows)
    excluded = {
        site: counts
        for site, counts in site_counts.items()
        if counts["asd"] < min_per_class or counts["control"] < min_per_class
    }
    excluded_sites = set(excluded)
    return [row for row in rows if row["site_id"] not in excluded_sites], excluded


def make_manifest_row(
    row: dict[str, str],
    pipeline: str,
    strategy: str,
    derivative: str,
) -> dict[str, str]:
    dx_group = row.get("DX_GROUP", "").strip()
    return {
        "subject_id": row.get("SUB_ID", "").strip(),
        "site_id": row.get("SITE_ID", "").strip(),
        "file_id": row.get("FILE_ID", "").strip(),
        "dx_group": dx_group,
        "label_asd": "1" if dx_group == "1" else "0",
        "age_at_scan": row.get("AGE_AT_SCAN", "").strip(),
        "sex": row.get("SEX", "").strip(),
        "fiq": row.get("FIQ", "").strip(),
        "func_mean_fd": row.get("func_mean_fd", "").strip(),
        "func_num_fd": row.get("func_num_fd", "").strip(),
        "func_perc_fd": row.get("func_perc_fd", "").strip(),
        "func_quality": row.get("func_quality", "").strip(),
        "sub_in_smp": row.get("SUB_IN_SMP", "").strip(),
        "qc_rater_1": row.get("qc_rater_1", "").strip(),
        "qc_func_rater_2": row.get("qc_func_rater_2", "").strip(),
        "qc_func_rater_3": row.get("qc_func_rater_3", "").strip(),
        "pipeline": pipeline,
        "strategy": strategy,
        "derivative": derivative,
        "timeseries_url": derivative_url(row, pipeline, strategy, derivative),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def nested_counter(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[row["site_id"]][row["dx_group"]] += 1
    return {
        site: {
            "asd": counts[site].get("1", 0),
            "control": counts[site].get("2", 0),
            "total": sum(counts[site].values()),
        }
        for site in sorted(counts)
    }


def build_summary(
    *,
    raw_rows: list[dict[str, str]],
    selected_rows: list[dict[str, str]],
    raw_csv: Path,
    manifest_csv: Path,
    args: argparse.Namespace,
    url_checked: int,
    url_missing: int,
    pre_site_filter_rows: int,
    excluded_sites: dict[str, dict[str, int]],
) -> dict[str, object]:
    class_counts = Counter(row["dx_group"] for row in selected_rows)
    site_counts = nested_counter(selected_rows)
    return {
        "source": {
            "phenotypic_url": args.phenotypic_url,
            "raw_csv": str(raw_csv),
            "raw_csv_sha256": sha256_file(raw_csv),
            "raw_row_count": len(raw_rows),
        },
        "selection": {
            "pipeline": args.pipeline,
            "strategy": args.strategy,
            "derivative": args.derivative,
            "qc_rule": args.qc_rule,
            "requires_file_id": True,
            "url_checked": url_checked,
            "url_missing": url_missing,
            "min_site_class_size": args.min_site_class_size,
            "rows_before_site_filter": pre_site_filter_rows,
            "excluded_sites": excluded_sites,
        },
        "manifest": {
            "path": str(manifest_csv),
            "sha256": sha256_file(manifest_csv),
            "row_count": len(selected_rows),
            "site_count": len(site_counts),
            "class_counts": {
                "asd": class_counts.get("1", 0),
                "control": class_counts.get("2", 0),
            },
            "site_counts": site_counts,
        },
    }


def main() -> int:
    args = parse_args()
    raw_csv = Path(args.raw_csv)
    output_csv = Path(args.output)
    summary_output = Path(args.summary_output)

    download_if_needed(args.phenotypic_url, raw_csv)
    raw_rows = read_rows(raw_csv)

    qc_predicate = QC_RULES[args.qc_rule]
    candidates = [
        row
        for row in raw_rows
        if available_file(row) and qc_predicate(row) and row.get("DX_GROUP") in {"1", "2"}
    ]

    selected = [
        make_manifest_row(
            raw_row,
            pipeline=args.pipeline,
            strategy=args.strategy,
            derivative=args.derivative,
        )
        for raw_row in candidates
    ]
    url_checked = 0
    url_missing = 0
    if args.check_urls:
        selected, url_checked, url_missing = filter_by_url_availability(
            selected,
            timeout=args.url_timeout,
            workers=args.url_workers,
            max_checks=args.max_url_checks,
        )

    pre_site_filter_rows = len(selected)
    selected, excluded_sites = filter_sites_by_class_count(
        selected, args.min_site_class_size
    )

    selected.sort(key=lambda row: (row["site_id"], int(row["subject_id"])))
    write_csv(output_csv, selected)
    summary = build_summary(
        raw_rows=raw_rows,
        selected_rows=selected,
        raw_csv=raw_csv,
        manifest_csv=output_csv,
        args=args,
        url_checked=url_checked,
        url_missing=url_missing,
        pre_site_filter_rows=pre_site_filter_rows,
        excluded_sites=excluded_sites,
    )
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Wrote manifest: {output_csv}")
    print(f"Wrote summary:  {summary_output}")
    print(f"Rows: {summary['manifest']['row_count']}")
    print(f"Sites: {summary['manifest']['site_count']}")
    print(f"ASD/control: {summary['manifest']['class_counts']}")
    if args.check_urls:
        print(f"URL checks: {url_checked} checked, {url_missing} missing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
