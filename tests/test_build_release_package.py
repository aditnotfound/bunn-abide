from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.build_release_package import (
    DEFAULT_CONFIG,
    ROOT,
    ReleaseError,
    build_release,
    collect_files,
    load_config,
    scan_files,
    sha256_file,
)


class ReleasePackageTests(unittest.TestCase):
    def test_release_selection_excludes_private_trees(self) -> None:
        config = load_config(DEFAULT_CONFIG)
        files = collect_files(ROOT, config)
        self.assertTrue(set(config["required_files"]).issubset(files))
        forbidden = ("data/", "outputs/", ".run-control/", ".venv/", "output/release/")
        self.assertFalse(any(path.startswith(forbidden) for path in files))

    def test_privacy_scan_rejects_private_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unsafe_path = "C:" + r"\Users" + r"\Researcher\secret.pem"
            (root / "unsafe.txt").write_text(unsafe_path, encoding="utf-8")
            with self.assertRaises(ReleaseError):
                scan_files(root, ["unsafe.txt"])

    def test_release_build_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            archive_a, manifest_a = build_release(ROOT, DEFAULT_CONFIG, Path(first))
            archive_b, manifest_b = build_release(ROOT, DEFAULT_CONFIG, Path(second))
            self.assertEqual(sha256_file(archive_a), sha256_file(archive_b))
            self.assertEqual(manifest_a.read_bytes(), manifest_b.read_bytes())
            payload = json.loads(manifest_a.read_text(encoding="utf-8"))
            self.assertEqual(payload["privacy_scan"], "passed")
            with zipfile.ZipFile(archive_a) as bundle:
                names = bundle.namelist()
            self.assertTrue(any(name.endswith("/RELEASE_MANIFEST.json") for name in names))
            self.assertFalse(any("outputs/" in name for name in names))


if __name__ == "__main__":
    unittest.main()
