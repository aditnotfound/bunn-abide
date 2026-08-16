from __future__ import annotations

import unittest

from scripts.create_loso_splits import create_assignments


class LosoSplitTests(unittest.TestCase):
    def test_outer_test_once_and_inner_groups_are_disjoint(self) -> None:
        rows = []
        for site in ["A", "B", "C", "D", "E"]:
            for label in [0, 1]:
                rows.append(
                    {
                        "connectome_row": str(len(rows)),
                        "subject_id": f"{site}-{label}",
                        "site_id": site,
                        "label_asd": str(label),
                    }
                )
        outer, inner, summaries = create_assignments(rows, inner_folds=2, seed=9)
        self.assertEqual(len(outer), len(rows))
        self.assertEqual({row["subject_id"] for row in outer}, {row["subject_id"] for row in rows})
        self.assertEqual(len(summaries), 5)
        for summary in summaries:
            held_out = summary["held_out_site"]
            self.assertTrue(all(row["site_id"] != held_out for row in inner if row["outer_fold"] == summary["outer_fold"]))
            self.assertEqual(len(summary["inner_folds"]), 2)


if __name__ == "__main__":
    unittest.main()
