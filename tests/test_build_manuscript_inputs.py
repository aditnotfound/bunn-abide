from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.build_manuscript_inputs import (
    ManuscriptInputError,
    build_manuscript_inputs,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = REPO_ROOT / "configs/manuscript_v1.json"
PAPER = REPO_ROOT / "paper"


class ManuscriptInputTests(unittest.TestCase):
    def test_frozen_inputs_generate_complete_manifest(self) -> None:
        result = build_manuscript_inputs(REPO_ROOT, CONTRACT)
        self.assertEqual(result["validated_frozen_inputs"], 4)
        self.assertEqual(result["validated_paper_assets"], 11)
        self.assertEqual(len(result["generated_outputs"]), 11)

        manifest = json.loads(
            (PAPER / "generated/manuscript_input_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["evidence_commit"], "a4def2a1f14f1bbff71356d0689eeeee4f405f4a")
        self.assertEqual(len(manifest["generated_inputs"]), 10)
        for record in manifest["generated_inputs"]:
            path = REPO_ROOT / record["path"]
            self.assertTrue(path.is_file())
            self.assertEqual(sha256_file(path), record["sha256"])

    def test_generation_is_byte_deterministic(self) -> None:
        first = build_manuscript_inputs(REPO_ROOT, CONTRACT)
        first_hashes = {
            path: sha256_file(REPO_ROOT / path)
            for path in first["generated_outputs"]
        }
        second = build_manuscript_inputs(REPO_ROOT, CONTRACT)
        second_hashes = {
            path: sha256_file(REPO_ROOT / path)
            for path in second["generated_outputs"]
        }
        self.assertEqual(first_hashes, second_hashes)

    def test_contract_digest_mismatch_stops_generation(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        first_path = next(iter(contract["frozen_inputs"]))
        contract["frozen_inputs"][first_path] = "0" * 64
        with TemporaryDirectory() as temporary:
            bad_contract = Path(temporary) / "manuscript.json"
            bad_contract.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaises(ManuscriptInputError):
                build_manuscript_inputs(REPO_ROOT, bad_contract)

    def test_citation_keys_resolve_and_claim_ids_are_covered(self) -> None:
        sources = [PAPER / "manuscript.tex", *sorted((PAPER / "sections").glob("*.tex"))]
        tex = "\n".join(path.read_text(encoding="utf-8") for path in sources)
        cited = set()
        for group in re.findall(r"\\cite\w*\{([^}]+)\}", tex):
            cited.update(key.strip() for key in group.split(","))
        bib = (PAPER / "references.bib").read_text(encoding="utf-8")
        available = set(re.findall(r"^@\w+\{([^,]+),", bib, flags=re.MULTILINE))
        self.assertTrue(cited)
        self.assertEqual(cited - available, set())

        comments = " ".join(re.findall(r"^%\s*(?:Claim|Claims)\s+(.+)$", tex, re.MULTILINE))
        for claim_id in json.loads(CONTRACT.read_text(encoding="utf-8"))["claim_ids"]:
            self.assertIn(claim_id, comments)

    def test_primary_results_are_not_duplicated_in_narrative_sources(self) -> None:
        forbidden_literals = {
            "0.6401",
            "-0.00958",
            "-0.03665",
            "0.01146",
            "-0.05516",
            "-0.08297",
            "-0.02752",
            "-0.00719",
            "-0.01310",
            "-0.00105",
            "8,529",
            "5,889",
            "31.14",
            "15.09",
        }
        narrative = "\n".join(
            path.read_text(encoding="utf-8")
            for path in [
                PAPER / "manuscript.tex",
                PAPER / "supplement.tex",
                *sorted((PAPER / "sections").glob("*.tex")),
            ]
        )
        for literal in forbidden_literals:
            self.assertNotIn(literal, narrative)


if __name__ == "__main__":
    unittest.main()
