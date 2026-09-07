"""Tests for the non-LLM baseline adapters and the baselines runner."""

from __future__ import annotations

import os
import tempfile
import unittest

from grounding_eval.adapters.oracle import OracleAdapter
from grounding_eval.adapters.random_floor import RandomFloorAdapter
from grounding_eval.adapters.retrieval import RetrievalAdapter, TfidfIndex
from grounding_eval.corpus import load_corpus
from grounding_eval.harness import run_once
from grounding_eval.run_baselines import main, render_table, run_all, summarize
from grounding_eval.schema import ItemType, Outcome


class BaselineAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = load_corpus()

    def test_oracle_is_ceiling(self):
        run = run_once(OracleAdapter(), self.corpus)
        factual = [s for s in run.scores if s.item_type is ItemType.FACTUAL]
        self.assertTrue(all(s.outcome is Outcome.CORRECT for s in factual))
        self.assertTrue(all(s.citation_grade.value == "correct_clause" for s in run.scores))
        self.assertFalse(run.is_mock)

    def test_random_floor_is_seeded_and_order_independent(self):
        pool = [i.source.clause for i in self.corpus]
        a = RandomFloorAdapter(seed=1, clause_pool=pool)
        b = RandomFloorAdapter(seed=1, clause_pool=pool)
        items = list(self.corpus)
        self.assertEqual([r.text for r in a.answer_all(items)], [r.text for r in b.answer_all(reversed(items))][::-1])
        c = RandomFloorAdapter(seed=2, clause_pool=pool)
        self.assertNotEqual([r.text for r in a.answer_all(items)], [r.text for r in c.answer_all(items)])

    def test_random_floor_only_uses_item_options(self):
        pool = [i.source.clause for i in self.corpus]
        for item, resp in zip(self.corpus, RandomFloorAdapter(clause_pool=pool).answer_all(self.corpus)):
            self.assertIn(resp.metadata["option"], ("correct", "adjacent"))
            self.assertIn(resp.citations[0], pool)

    def test_tfidf_index_ranks_exact_match_first(self):
        idx = TfidfIndex([("a", "permit space forced air ventilation"), ("b", "rupture disk burst pressure"), ("c", "noise dosimeter exposure")])
        self.assertEqual(idx.query("what burst pressure applies to a rupture disk")[0][0], "b")

    def test_retrieval_cites_matched_clause_and_never_reads_key(self):
        adapter = RetrievalAdapter(self.corpus)
        by_id = {i.id: i for i in self.corpus}
        for item in self.corpus:
            resp = adapter.answer(item)
            matched = by_id[resp.metadata["matched_item"]]
            self.assertEqual(resp.citations, [matched.source.clause])
            self.assertIn(matched.source.anchor_text, resp.text)
            self.assertNotIn(item.adjacent_wrong.label, resp.text)


class RunnerTests(unittest.TestCase):
    def test_run_all_and_summary_shape(self):
        corpus = load_corpus()
        runs = run_all(corpus, n_seeds=2)
        self.assertEqual(len(runs["random_floor"]), 2)
        self.assertEqual(len(runs["oracle"]), 1)
        summary = summarize(runs, corpus)
        self.assertEqual(set(summary["adapter"]), {"random_floor", "retrieval_tfidf", "oracle"})
        for col in ("accuracy_ci_low", "adjacent_ci_high", "citation_ci_low", "safety_utility"):
            self.assertIn(col, summary.columns)
        oracle = summary[summary["adapter"] == "oracle"].iloc[0]
        self.assertEqual(oracle["accuracy"], 1.0)
        # One category-error reference answer is scored as an acceptable abstention (0.9), so the
        # oracle ceiling under the safety policy is just under 1.0.
        self.assertGreater(oracle["safety_utility"], 0.99)
        table = render_table(summary)
        self.assertIn("| oracle |", table)
        self.assertNotIn("\u2014", table)

    def test_main_writes_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main(["--out", tmp, "--seeds", "1"]), 0)
            for name in ("baselines_random_floor.json", "baselines_retrieval_tfidf.json", "baselines_oracle.json", "baselines_summary.csv", "baselines_table.md"):
                self.assertTrue(os.path.exists(os.path.join(tmp, name)), name)


if __name__ == "__main__":
    unittest.main()
