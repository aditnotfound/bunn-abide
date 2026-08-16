from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.build_paper_assets import PaperAssetError, build_paper_assets, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = REPO_ROOT / "configs/paper_assets_v1.json"


def private_evidence_available() -> bool:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    return all((REPO_ROOT / relative).is_file() for relative in contract["frozen_inputs"])


PRIVATE_EVIDENCE_AVAILABLE = private_evidence_available()
PRIVATE_EVIDENCE_REASON = (
    "private Step 7/11/12 evidence archives are not distributed in the public package"
)


class PaperAssetTests(unittest.TestCase):
    def build_in_temporary_directory(self, root: Path) -> dict[str, object]:
        return build_paper_assets(
            REPO_ROOT,
            CONTRACT,
            root / "paper/generated",
            root / "reproducibility",
        )

    @unittest.skipUnless(PRIVATE_EVIDENCE_AVAILABLE, PRIVATE_EVIDENCE_REASON)
    def test_real_frozen_evidence_builds_expected_snapshot(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.build_in_temporary_directory(root)
            snapshot = json.loads(
                (root / "reproducibility/result_snapshot.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["validated_inputs"], 28)
            self.assertEqual(snapshot["cohort"]["participants"], 754)
            self.assertEqual(snapshot["cohort"]["held_out_sites"], 18)
            self.assertAlmostEqual(
                snapshot["classical_baselines"]["connectome_elastic_net_logistic"]
                ["equal_site_balanced_accuracy"],
                0.6401067135652798,
            )
            primary = snapshot["confirmatory_predictive_contrasts"][
                "learned_bunn_curve_minus_gcn_curve"
            ]
            self.assertAlmostEqual(primary["estimate"], -0.009578650270252929)
            self.assertEqual(
                primary["bootstrap_ci_95"],
                [-0.036648034920577756, 0.011455085043534615],
            )
            self.assertFalse(snapshot["step11_decision"]["all_three_conditions"])
            self.assertFalse(snapshot["claim_boundaries"]["general_bunn_inferiority_claim_allowed"])

    @unittest.skipUnless(PRIVATE_EVIDENCE_AVAILABLE, PRIVATE_EVIDENCE_REASON)
    def test_generation_is_byte_deterministic(self) -> None:
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            first_root = Path(first)
            second_root = Path(second)
            self.build_in_temporary_directory(first_root)
            self.build_in_temporary_directory(second_root)

            def tree_digests(root: Path) -> dict[str, str]:
                return {
                    path.relative_to(root).as_posix(): sha256_file(path)
                    for path in sorted(root.rglob("*"))
                    if path.is_file()
                }

            self.assertEqual(tree_digests(first_root), tree_digests(second_root))

    def test_digest_mismatch_fails_before_generation(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
            first_path = next(iter(contract["frozen_inputs"]))
            contract["frozen_inputs"][first_path] = "0" * 64
            bad_contract = root / "bad_contract.json"
            bad_contract.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaises(PaperAssetError):
                build_paper_assets(
                    REPO_ROOT,
                    bad_contract,
                    root / "paper/generated",
                    root / "reproducibility",
                )
            self.assertFalse((root / "paper").exists())
            self.assertFalse((root / "reproducibility").exists())

    @unittest.skipUnless(PRIVATE_EVIDENCE_AVAILABLE, PRIVATE_EVIDENCE_REASON)
    def test_publication_figures_are_valid_nonempty_pngs(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build_in_temporary_directory(root)
            contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
            for name in contract["generated_figures"]:
                payload = (root / "paper/generated/figures" / name).read_bytes()
                self.assertGreater(len(payload), 20_000)
                self.assertEqual(payload[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
