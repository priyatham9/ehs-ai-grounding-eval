"""Tests for the non-LLM baseline adapters and the baselines runner."""

from __future__ import annotations

import os
import tempfile
import unittest

from grounding_eval.adapters.bm25 import BM25Adapter, BM25Index
from grounding_eval.adapters.bm25plus import (
    BM25PlusAdapter,
    BM25PlusIndex,
    RRFAdapter,
    reciprocal_rank_fusion,
)
from grounding_eval.adapters.oracle import OracleAdapter
from grounding_eval.adapters.random_floor import RandomFloorAdapter
from grounding_eval.adapters.retrieval import RetrievalAdapter, TfidfIndex
from grounding_eval.corpus import load_corpus
from grounding_eval.harness import run_once
from grounding_eval.run_baselines import breakdown, main, render_breakdown, render_table, run_all, summarize
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

    def test_bm25_index_ranks_exact_match_first(self):
        idx = BM25Index([("a", "permit space forced air ventilation"), ("b", "rupture disk burst pressure"), ("c", "noise dosimeter exposure")])
        self.assertEqual(idx.query("what burst pressure applies to a rupture disk")[0][0], "b")

    def test_bm25_index_is_deterministic(self):
        docs = [("a", "permit space forced air ventilation"), ("b", "rupture disk burst pressure"), ("c", "noise dosimeter exposure")]
        idx1 = BM25Index(docs)
        idx2 = BM25Index(docs)
        result1 = idx1.query("what burst pressure applies to a rupture disk")
        result2 = idx2.query("what burst pressure applies to a rupture disk")
        self.assertEqual(result1, result2)

    def test_bm25_ranks_by_relevance(self):
        idx = BM25Index([("a", "rupture disk burst pressure safety"), ("b", "permit space ventilation"), ("c", "noise dosimeter")])
        ranked = idx.query("rupture disk burst pressure")
        self.assertEqual(ranked[0][0], "a")
        self.assertGreater(ranked[0][1], ranked[1][1])

    def test_bm25plus_index_ranks_exact_match_first(self):
        idx = BM25PlusIndex([("a", "permit space forced air ventilation"), ("b", "rupture disk burst pressure"), ("c", "noise dosimeter exposure")])
        self.assertEqual(idx.query("what burst pressure applies to a rupture disk")[0][0], "b")

    def test_bm25plus_parameters_are_the_preregistered_ones(self):
        idx = BM25PlusIndex([("a", "x")])
        self.assertEqual((idx.k1, idx.b, idx.delta), (1.2, 0.75, 1.0))
        self.assertEqual(RRFAdapter(self.corpus).k, 60)

    def test_bm25plus_never_penalises_a_common_term(self):
        # "pressure" is in 3 of 4 documents: Okapi IDF is negative there, BM25+ IDF is not.
        docs = [("a", "pressure valve"), ("b", "pressure disk"), ("c", "pressure tank"), ("d", "noise")]
        self.assertTrue((BM25PlusIndex(docs).idf >= 0).all())
        okapi = BM25Index(docs).query("pressure")
        self.assertLess(dict(okapi)["a"], 0.0)
        self.assertEqual(okapi[0][0], "d")  # Okapi ranks the one document without the term first.
        ranked = dict(BM25PlusIndex(docs).query("pressure"))
        self.assertGreater(ranked["a"], ranked["d"])
        self.assertEqual(ranked["d"], 0.0)

    def test_bm25plus_delta_lower_bounds_a_match(self):
        docs = [("short", "valve"), ("long", "valve " + "filler " * 200)]
        without = dict(BM25PlusIndex(docs, delta=0.0).query("valve"))
        with_delta = dict(BM25PlusIndex(docs).query("valve"))
        idf = BM25PlusIndex(docs).idf[0]
        self.assertAlmostEqual(with_delta["long"] - without["long"], idf)

    def test_reciprocal_rank_fusion(self):
        fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "c"]], k=60)
        # a and b tie (1/61 + 1/62); the tie keeps the first ranking's order.
        self.assertEqual([d for d, _ in fused], ["a", "b", "c"])
        self.assertAlmostEqual(fused[0][1], 1 / 61 + 1 / 62)
        self.assertEqual([d for d, _ in reciprocal_rank_fusion([["x", "y"], ["y", "z"]])][0], "y")

    def test_new_retrievers_cite_matched_clause_and_never_read_key(self):
        by_id = {i.id: i for i in self.corpus}
        for adapter in (BM25PlusAdapter(self.corpus), RRFAdapter(self.corpus)):
            self.assertEqual(adapter.arm, "grounded")
            for item in self.corpus:
                resp = adapter.answer(item)
                matched = by_id[resp.metadata["matched_item"]]
                self.assertEqual(resp.citations, [matched.source.clause])
                self.assertIn(matched.source.anchor_text, resp.text)
                self.assertNotIn(item.adjacent_wrong.label, resp.text)
                self.assertEqual(resp.provenance, "%s_baseline" % adapter.name)

    def test_retrieval_cites_matched_clause_and_never_reads_key(self):
        adapter = RetrievalAdapter(self.corpus)
        by_id = {i.id: i for i in self.corpus}
        for item in self.corpus:
            resp = adapter.answer(item)
            matched = by_id[resp.metadata["matched_item"]]
            self.assertEqual(resp.citations, [matched.source.clause])
            self.assertIn(matched.source.anchor_text, resp.text)
            self.assertNotIn(item.adjacent_wrong.label, resp.text)

    def test_bm25_adapter_cites_matched_clause_and_never_reads_key(self):
        adapter = BM25Adapter(self.corpus)
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
        self.assertEqual(set(summary["adapter"]), {"random_floor", "retrieval_tfidf", "retrieval_bm25", "retrieval_bm25plus", "retrieval_rrf", "oracle"})
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

    def test_breakdown_by_domain_and_minimal_pairs(self):
        corpus = load_corpus()
        runs = run_all(corpus, n_seeds=1)
        table = breakdown(runs, corpus)
        oracle = table[table["adapter"] == "oracle"].set_index("subset")
        self.assertTrue((oracle["accuracy"] == 1.0).all())
        self.assertEqual(int(oracle.loc["all_factual", "n_items"]), 63)
        self.assertEqual(int(oracle.loc["minimal_pair_items", "n_items"]) + int(oracle.loc["unpaired_items", "n_items"]), 63)
        self.assertEqual(int(oracle.loc["pairs_both_correct", "n_items"]), len(corpus.pairs()))
        domains = [s for s in oracle.index if s.startswith("domain:")]
        self.assertEqual(len(domains), len({i.domain for i in corpus}))
        self.assertEqual(sum(int(oracle.loc[d, "n_items"]) for d in domains), 63)
        for _, row in table.iterrows():
            self.assertLessEqual(row["ci_low"], row["accuracy"] + 1e-12)
            self.assertGreaterEqual(row["ci_high"], row["accuracy"] - 1e-12)
        # Domain breakdown agrees with the headline accuracy for every adapter.
        summary = summarize(runs, corpus).set_index("adapter")
        for adapter, group in table.groupby("adapter"):
            overall = group[group["subset"] == "all_factual"]["accuracy"].iloc[0]
            self.assertAlmostEqual(overall, summary.loc[adapter, "accuracy"])
        md = render_breakdown(table)
        self.assertIn("| pairs_both_correct | 23 |", md)
        self.assertIn("retrieval_bm25plus", md.splitlines()[0])

    def test_main_writes_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main(["--out", tmp, "--seeds", "1"]), 0)
            for name in ("baselines_random_floor.json", "baselines_retrieval_tfidf.json", "baselines_oracle.json", "baselines_summary.csv", "baselines_table.md",
                         "baselines_retrieval_bm25.json", "baselines_retrieval_bm25plus.json", "baselines_retrieval_rrf.json",
                         "baselines_breakdown.csv", "baselines_breakdown.md"):
                self.assertTrue(os.path.exists(os.path.join(tmp, name)), name)


if __name__ == "__main__":
    unittest.main()
