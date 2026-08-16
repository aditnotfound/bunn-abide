"""Build a deterministic, privacy-scanned public release archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "release_v1.json"
DEFAULT_OUTPUT = ROOT / "output" / "release"


class ReleaseError(RuntimeError):
    """Raised when a release-safety or completeness check fails."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def git_candidates(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return sorted(item for item in result.stdout.decode("utf-8").split("\0") if item)


def ensure_tracked_tree_clean(root: Path) -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise ReleaseError(
            "tracked files differ from HEAD; commit or restore them before building a release"
        )


def collect_files(root: Path, config: dict) -> list[str]:
    prefixes = tuple(config["exclude_prefixes"])
    suffixes = tuple(config["exclude_suffixes"])
    files = []
    for relative in git_candidates(root):
        normalized = relative.replace("\\", "/")
        if normalized.startswith(prefixes) or normalized.endswith(suffixes):
            continue
        path = root / normalized
        if path.is_file():
            files.append(normalized)

    missing = sorted(set(config["required_files"]) - set(files))
    if missing:
        raise ReleaseError(f"required release files are missing: {missing}")
    return files


TEXT_SUFFIXES = {
    "",
    ".bib",
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}

PRIVATE_PATTERNS = {
    "Windows user path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+", re.IGNORECASE),
    "Linux home path": re.compile(r"/home/(?:ubuntu|ec2-user)/", re.IGNORECASE),
    "private key": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "AWS secret setting": re.compile(r"(?:aws_)?secret_access_key\s*[:=]", re.IGNORECASE),
    "private notification address": re.compile(r"\bme@adit\.email\b", re.IGNORECASE),
    "known public instance IP": re.compile(r"\b(?:35\.175\.173\.51|98\.89\.22\.112)\b"),
    "known AWS account": re.compile(r"\b020529562621\b"),
}


def scan_files(root: Path, files: Iterable[str]) -> None:
    findings: list[str] = []
    for relative in files:
        path = root / relative
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{relative}: {label}")
    if findings:
        raise ReleaseError("privacy scan failed:\n" + "\n".join(findings))


def source_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def manifest_payload(root: Path, config: dict, files: list[str]) -> dict:
    return {
        "schema_version": config["schema_version"],
        "release_name": config["release_name"],
        "release_date": config["release_date"],
        "source_commit": source_commit(root),
        "privacy_scan": "passed",
        "file_count": len(files),
        "files": [
            {
                "path": relative,
                "bytes": (root / relative).stat().st_size,
                "sha256": sha256_file(root / relative),
            }
            for relative in files
        ],
    }


def stable_json(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_zip(root: Path, archive: Path, release_name: str, files: list[str], manifest: bytes) -> None:
    fixed_time = (1980, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        entries = [(relative, (root / relative).read_bytes()) for relative in files]
        entries.append(("RELEASE_MANIFEST.json", manifest))
        for relative, data in sorted(entries):
            info = zipfile.ZipInfo(f"{release_name}/{relative}", fixed_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def build_release(
    root: Path,
    config_path: Path,
    output_dir: Path,
    *,
    require_clean: bool = True,
) -> tuple[Path, Path]:
    if require_clean:
        ensure_tracked_tree_clean(root)
    config = load_config(config_path)
    files = collect_files(root, config)
    scan_files(root, files)
    payload = manifest_payload(root, config, files)
    manifest_bytes = stable_json(payload)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "public_release_manifest.json"
    archive_path = output_dir / f"{config['release_name']}.zip"
    manifest_path.write_bytes(manifest_bytes)
    write_zip(root, archive_path, config["release_name"], files, manifest_bytes)

    summary = {
        "archive": archive_path.name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "file_count": len(files),
        "manifest": manifest_path.name,
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "privacy_scan": "passed",
    }
    (output_dir / "release_summary.json").write_bytes(stable_json(summary))
    return archive_path, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    archive, manifest = build_release(ROOT, args.config, args.output_dir, require_clean=True)
    print(f"Built {archive}")
    print(f"Manifest {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
