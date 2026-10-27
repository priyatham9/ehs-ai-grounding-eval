"""Guards scripts/item_analysis.py: committed outputs match regenerated output."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import item_analysis as ia  # noqa: E402


def _read(name: str) -> str:
    with open(os.path.join(ROOT, "results", name)) as fh:
        return fh.read()


class TestItemAnalysis(unittest.TestCase):
    def test_committed_outputs_match_regenerated(self) -> None:
        res = ia.analyse()
        self.assertEqual(_read("item_analysis.csv"), ia.render_csv(res), "run scripts/item_analysis.py")
        self.assertEqual(_read("item_analysis.md"), ia.render_md(res), "run scripts/item_analysis.py")

    def test_deterministic_and_clean(self) -> None:
        res = ia.analyse()
        self.assertEqual(ia.render_md(res), ia.render_md(ia.analyse()))
        self.assertNotIn("—", ia.render_md(res))

    def test_point_biserial_zero_variance(self) -> None:
        self.assertIsNone(ia.point_biserial([1, 1, 1, 1], [1.0, 2.0, 3.0, 4.0]))
        self.assertIsNone(ia.point_biserial([0, 1, 0, 1], [2.0, 2.0, 2.0, 2.0]))
        self.assertAlmostEqual(ia.point_biserial([0, 0, 1, 1], [1.0, 2.0, 3.0, 4.0]), 0.894427, places=5)

    def test_kappa(self) -> None:
        self.assertAlmostEqual(ia.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]), 1.0)
        self.assertIsNone(ia.cohen_kappa([1, 1], [1, 1]))
        self.assertAlmostEqual(ia.fleiss_kappa([[1, 1, 1], [0, 0, 0]]), 1.0)

    def test_difficulty_matches_summary(self) -> None:
        items = ia.analyse()["items"]
        self.assertEqual(len(items), 63)
        self.assertTrue(all(0.0 <= it["difficulty"] <= 1.0 for it in items))


if __name__ == "__main__":
    unittest.main()
