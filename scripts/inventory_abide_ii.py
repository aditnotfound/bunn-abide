#!/usr/bin/env python3
"""Build a score-blind, source-hashed ABIDE-II compatibility inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "bunn-abide-inventory/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    temporary.replace(destination)


def read_csv(path: Path, encoding: str = "utf-8-sig") -> list[dict[str, str]]:
    with path.open("r", encoding=encoding, newline="") as stream:
        reader = csv.DictReader(stream)
        rows = []
        for row in reader:
            rows.append({(key or "").strip(): (value or "").strip() for key, value in row.items()})
        return rows


def normalized_id(value: str) -> str:
    return str(int(float(value)))


def list_s3(
    bucket_url: str,
    prefix: str,
    *,
    delimiter: str | None = None,
) -> tuple[list[str], list[str]]:
    keys: list[str] = []
    prefixes: list[str] = []
    token: str | None = None
    while True:
        parameters = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            parameters["delimiter"] = delimiter
        if token:
            parameters["continuation-token"] = token
        url = f"{bucket_url}?{urllib.parse.urlencode(parameters)}"
        request = urllib.request.Request(url, headers={"User-Agent": "bunn-abide-inventory/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            root = ET.fromstring(response.read())
        keys.extend(node.text or "" for node in root.findall("s3:Contents/s3:Key", S3_NS))
        prefixes.extend(
            node.text or "" for node in root.findall("s3:CommonPrefixes/s3:Prefix", S3_NS)
        )
        if root.findtext("s3:IsTruncated", default="false", namespaces=S3_NS) != "true":
            break
        token = root.findtext("s3:NextContinuationToken", namespaces=S3_NS)
        if not token:
            raise RuntimeError("S3 listing was truncated without a continuation token")
    return keys, prefixes


def pcp_subject_ids(keys: Iterable[str]) -> set[str]:
    ids: set[str] = set()
    for key in keys:
        match = re.search(r"_(\d+)_rois_aal\.1D$", key)
        if match:
            ids.add(normalized_id(match.group(1)))
    return ids


def decide_gate(
    *,
    exact_derivative_matches: int,
    main_participants: int,
    direct_prefix_key_count: int,
    lle_site_count: int,
) -> dict[str, object]:
    exact_complete = main_participants > 0 and exact_derivative_matches == main_participants
    passed = exact_complete
    reasons = []
    if not exact_complete:
        reasons.append(
            "No official complete C-PAC filt_noglobal AAL-116 ROI-time-series collection "
            "was found for the main ABIDE-II cohort."
        )
    if direct_prefix_key_count == 0:
        reasons.append("The tested ABIDE-II-specific public S3 prefixes contain no objects.")
    if lle_site_count:
        reasons.append(
            "The official LLE area contains ABIDE-II site directories, but its documented "
            "mixed C-PAC/SPM volume workflow is not the frozen Study 1 ROI derivative."
        )
    return {
        "decision": "PASS" if passed else "FAIL",
        "model_evaluation_authorized": passed,
        "exact_derivative_complete": exact_complete,
        "reasons": reasons,
    }


def build_inventory(config_path: Path, raw_dir: Path, output_path: Path, refresh: bool) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sources = config["official_sources"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "composite_phenotype": raw_dir / "ABIDEII_Composite_Phenotypic.csv",
        "longitudinal_phenotype": raw_dir / "ABIDEII_Long_Composite_Phenotypic.csv",
        "mri_quality_metrics": raw_dir / "ABIDEII_MRI_Quality_Metrics.zip",
        "release_page": raw_dir / "abide_II.html",
    }
    for name, path in files.items():
        if refresh or not path.exists():
            download(sources[name], path)

    quality_dir = raw_dir / "quality_metrics"
    functional_qap = quality_dir / "ABIDEII_MRI_Quality_Metrics" / "functional_qap.csv"
    if refresh or not functional_qap.exists():
        if quality_dir.exists():
            shutil.rmtree(quality_dir)
        with zipfile.ZipFile(files["mri_quality_metrics"]) as archive:
            archive.extractall(quality_dir)

    main_rows = read_csv(files["composite_phenotype"], encoding="latin1")
    long_rows = read_csv(files["longitudinal_phenotype"], encoding="latin1")
    qap_rows = read_csv(functional_qap)
    abide_i_rows = read_csv(ROOT / "data" / "raw" / "Phenotypic_V1_0b_preprocessed1.csv")

    main_ids = {normalized_id(row["SUB_ID"]) for row in main_rows}
    abide_i_ids = {normalized_id(row["SUB_ID"]) for row in abide_i_rows}
    longitudinal_ids = {normalized_id(row["SUB_ID"]) for row in long_rows}
    qap_ids = {normalized_id(row["Sub_ID"]) for row in qap_rows}

    pcp_prefix = config["frozen_study_1_representation"]["pcp_prefix"]
    pcp_keys, _ = list_s3(sources["pcp_s3_bucket"], pcp_prefix)
    pcp_ids = pcp_subject_ids(pcp_keys)

    direct_key_count = 0
    direct_prefix_results: dict[str, int] = {}
    for prefix in config["candidate_prefixes"]["abide_ii_direct"]:
        keys, _ = list_s3(sources["pcp_s3_bucket"], prefix)
        direct_prefix_results[prefix] = len(keys)
        direct_key_count += len(keys)

    _, lle_prefixes = list_s3(
        sources["pcp_s3_bucket"],
        config["candidate_prefixes"]["lle_abide_ii"],
        delimiter="/",
    )
    lle_sites = sorted(
        prefix.rstrip("/").split("/")[-1]
        for prefix in lle_prefixes
        if not prefix.rstrip("/").endswith("masks")
    )

    by_site: dict[str, dict[str, object]] = defaultdict(
        lambda: {"participant_count": 0, "diagnosis_counts": Counter(), "participant_ids": set()}
    )
    for row in main_rows:
        site = row["SITE_ID"]
        by_site[site]["participant_count"] = int(by_site[site]["participant_count"]) + 1
        by_site[site]["diagnosis_counts"][row["DX_GROUP"]] += 1
        by_site[site]["participant_ids"].add(normalized_id(row["SUB_ID"]))

    site_rows = []
    for site, values in sorted(by_site.items()):
        # Match quality records by participant ID. The official phenotype uses
        # OILH_2 while the corresponding quality file uses the renamed ONRC_2.
        qap_count = len(values["participant_ids"] & qap_ids)
        participant_count = int(values["participant_count"])
        site_rows.append(
            {
                "site_id": site,
                "participants": participant_count,
                "asd": int(values["diagnosis_counts"].get("1", 0)),
                "control": int(values["diagnosis_counts"].get("2", 0)),
                "functional_qap_participants": qap_count,
                "functional_qap_coverage": round(qap_count / participant_count, 6),
            }
        )

    exact_matches = len(main_ids & pcp_ids)
    gate = decide_gate(
        exact_derivative_matches=exact_matches,
        main_participants=len(main_ids),
        direct_prefix_key_count=direct_key_count,
        lle_site_count=len(lle_sites),
    )
    inventory = {
        "schema_version": "abide_ii_metadata_inventory_v1",
        "contract": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "score_blind": True,
        "model_outputs_read_or_created": False,
        "sources": {
            name: {"url": sources[name], "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in files.items()
        },
        "main_cohort": {
            "rows": len(main_rows),
            "unique_participants": len(main_ids),
            "sites": len(by_site),
            "diagnosis_counts": dict(sorted(Counter(row["DX_GROUP"] for row in main_rows).items())),
            "abide_i_numeric_id_overlap": len(main_ids & abide_i_ids),
        },
        "longitudinal_collection": {
            "rows": len(long_rows),
            "unique_participants": len(longitudinal_ids),
            "abide_i_numeric_id_overlap": len(longitudinal_ids & abide_i_ids),
            "frozen_rule": "Exclude the longitudinal collection from an independent external cohort.",
        },
        "functional_quality_metrics": {
            "rows": len(qap_rows),
            "unique_participants": len(qap_ids),
            "main_cohort_participants_covered": len(main_ids & qap_ids),
            "main_cohort_participants_missing": len(main_ids - qap_ids),
        },
        "derivative_inventory": {
            "study_1_pcp_prefix": pcp_prefix,
            "study_1_pcp_key_count": len(pcp_keys),
            "main_abide_ii_subject_id_matches": exact_matches,
            "direct_abide_ii_prefix_key_counts": direct_prefix_results,
            "lle_abide_ii_site_count_excluding_masks": len(lle_sites),
            "lle_abide_ii_sites": lle_sites,
            "lle_compatibility": "Documented mixed C-PAC/SPM preprocessed volumes; not the frozen C-PAC filt_noglobal AAL-116 ROI derivative.",
        },
        "site_inventory": site_rows,
        "gate": gate,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes((json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "abide_ii_inventory_v1.json"
    )
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "abide_ii")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reproducibility" / "abide_ii_gate_inventory.json",
    )
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inventory = build_inventory(args.config.resolve(), args.raw_dir.resolve(), args.output.resolve(), args.refresh)
    gate = inventory["gate"]
    print(f"ABIDE-II compatibility gate: {gate['decision']}")
    print(f"Model evaluation authorized: {gate['model_evaluation_authorized']}")
    print(f"Inventory: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
