"""Tests for the manuscript draft: every number is sourced and re-derived from a file."""

from __future__ import annotations

import glob
import json
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import check_manuscript as cm  # noqa: E402

TEXT = open(cm.MANUSCRIPT, encoding="utf-8").read()


class ManuscriptNumberTests(unittest.TestCase):
    def test_appendix_has_many_rows_and_unique_ids(self) -> None:
        rows = cm.parse_appendix(TEXT)
        self.assertGreater(len(rows), 100)
        self.assertEqual(len({r["id"] for r in rows}), len(rows))

    def test_every_appendix_row_matches_its_source_file(self) -> None:
        self.assertEqual(cm.check_rows(), [])

    def test_every_number_in_the_body_has_an_appendix_row(self) -> None:
        self.assertEqual(cm.body_tokens_without_source(), [])

    def test_a_wrong_printed_value_is_detected(self) -> None:
        bad = {"id": "X1", "printed": "99.9%", "source": "results/baselines_summary.csv",
               "locator": "csv::results/baselines_summary.csv::adapter=retrieval_tfidf::accuracy"}
        self.assertFalse(cm.printed_matches(bad["printed"], cm.resolve(bad, {})))

    def test_clopper_pearson_known_values(self) -> None:
        lo, hi = cm.clopper_pearson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.3084971, places=5)  # 1 - 0.025 ** (1 / 10)
        lo, hi = cm.clopper_pearson(10, 10)
        self.assertEqual(hi, 1.0)
        self.assertAlmostEqual(lo, 1 - 0.3084971, places=5)
        lo, hi = cm.clopper_pearson(5, 10)
        self.assertAlmostEqual(lo + hi, 1.0, places=6)  # symmetric at k = n / 2


class ManuscriptHonestyTests(unittest.TestCase):
    def test_header_states_unsubmitted_draft_and_author(self) -> None:
        head = TEXT[:1500]
        self.assertIn("unsubmitted draft", head.lower())
        self.assertIn("Priyatham Chimmani", head)

    def test_no_language_model_claim_matches_the_repository(self) -> None:
        """The manuscript says no LLM was run; fail if a non-baseline run file appears."""
        self.assertIn("No language model was evaluated", TEXT)
        names = {os.path.basename(p) for p in glob.glob(os.path.join(REPO_ROOT, "results", "baselines_*.json"))}
        expected = {"baselines_%s.json" % a for a in (
            "random_floor", "retrieval_tfidf", "retrieval_bm25", "retrieval_bm25plus",
            "retrieval_rrf", "oracle")}
        self.assertEqual(names, expected)
        for name in names:
            with open(os.path.join(REPO_ROOT, "results", name), encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertNotIn("anthropic", json.dumps(data).lower(), msg=name)

    def test_every_numbered_reference_has_a_verification_status(self) -> None:
        apx = TEXT[TEXT.index("## Appendix B"):]
        cited = {int(n) for n in __import__("re").findall(r"\[R(\d+)", TEXT[:TEXT.index("## Appendix A")])}
        self.assertTrue(cited)
        for n in cited:
            line = next((ln for ln in apx.splitlines() if ln.startswith("* [R%d]" % n)), "")
            self.assertTrue(line, "reference R%d missing from Appendix B" % n)
            self.assertRegex(line, r"[Vv]erified|not independently verified|not verified")


if __name__ == "__main__":
    unittest.main()
