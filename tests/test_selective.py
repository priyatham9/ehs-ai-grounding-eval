"""Tests for answer-or-abstain scoring (risk-coverage, AURC, selective accuracy)."""

from __future__ import annotations

import math
import os
import unittest

from grounding_eval.corpus import load_corpus
from grounding_eval.run_baselines import render_selective, run_all, selective
from grounding_eval.selective import (
    aurc,
    aurc_optimal,
    aurc_random,
    cluster_bootstrap_selective,
    e_aurc,
    risk_coverage_points,
    selective_accuracy_at_coverage,
    summarize_selective,
)


class SelectiveMetricTests(unittest.TestCase):
    def test_hand_computed_aurc(self) -> None:
        # Ranks 1..4 have selective risk 0, 0, 1/3, 1/2; the mean is 5/24.
        conf, ok = [0.9, 0.8, 0.7, 0.1], [True, True, False, False]
        self.assertAlmostEqual(aurc(conf, ok), 5.0 / 24.0)
        self.assertAlmostEqual(aurc_optimal(ok), 5.0 / 24.0)
        self.assertAlmostEqual(e_aurc(conf, ok), 0.0)

    def test_perfect_and_worst_ordering(self) -> None:
        ok = [True, True, False, False]
        best = aurc([4, 3, 2, 1], ok)
        worst = aurc([1, 2, 3, 4], ok)
        self.assertAlmostEqual(best, aurc_optimal(ok))
        self.assertGreater(worst, aurc_random(ok))
        self.assertGreater(e_aurc([1, 2, 3, 4], ok), e_aurc([4, 3, 2, 1], ok))

    def test_constant_confidence_equals_overall_error_rate_in_expectation(self) -> None:
        ok = [True, False, False, False]
        # All tied: prefix k has expected accuracy 1/4 for every k, so risk is 3/4 throughout.
        self.assertAlmostEqual(aurc([0.5] * 4, ok), aurc_random(ok))
        self.assertAlmostEqual(aurc_random(ok), 0.75)

    def test_ties_enter_together_in_the_curve(self) -> None:
        pts = risk_coverage_points([0.9, 0.5, 0.5, 0.1], [True, True, False, False])
        self.assertEqual([p["n_answered"] for p in pts], [1.0, 3.0, 4.0])
        self.assertAlmostEqual(pts[1]["selective_accuracy"], 2.0 / 3.0)
        self.assertAlmostEqual(pts[-1]["coverage"], 1.0)
        self.assertAlmostEqual(pts[-1]["selective_accuracy"], 0.5)

    def test_selective_accuracy_at_coverage(self) -> None:
        conf, ok = [0.9, 0.8, 0.7, 0.1], [True, True, False, False]
        acc, cov = selective_accuracy_at_coverage(conf, ok, 0.5)
        self.assertEqual((acc, cov), (1.0, 0.5))
        # A tie group straddling the target is answered whole, so coverage overshoots.
        acc, cov = selective_accuracy_at_coverage([0.9, 0.5, 0.5, 0.5], [True, False, False, True], 0.5)
        self.assertAlmostEqual(cov, 1.0)
        self.assertAlmostEqual(acc, 0.5)
        with self.assertRaises(ValueError):
            selective_accuracy_at_coverage(conf, ok, 0.0)

    def test_missing_confidence_is_dropped_and_none_means_nan(self) -> None:
        self.assertTrue(math.isnan(aurc([None, None], [True, False])))
        self.assertEqual(risk_coverage_points([None], [True]), [])
        row = summarize_selective([None, None], [True, False], ["a", "b"])
        self.assertEqual(row["n_with_confidence"], 0)
        self.assertTrue(math.isnan(row["aurc"]))
        self.assertAlmostEqual(row["accuracy_full_coverage"], 0.5)

    def test_bootstrap_is_seeded_and_brackets_the_point(self) -> None:
        conf = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        ok = [True, True, False, True, False, False, True, False]
        fam = ["a", "a", "b", "b", "c", "c", "d", "d"]
        r1 = cluster_bootstrap_selective(conf, ok, fam, aurc, n_resamples=200, seed=1)
        r2 = cluster_bootstrap_selective(conf, ok, fam, aurc, n_resamples=200, seed=1)
        self.assertEqual(r1, r2)
        self.assertLessEqual(r1[1], r1[2])
        self.assertAlmostEqual(r1[0], aurc(conf, ok))

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            aurc([0.1], [True, False])


class SelectiveBaselineTests(unittest.TestCase):
    def test_baselines_scored_and_no_confidence_arms_reported_as_such(self) -> None:
        corpus = load_corpus()
        runs = run_all(corpus, n_seeds=1)
        summary, curves = selective(runs)
        by = summary.set_index("adapter")
        for name in ("random_floor", "oracle"):
            self.assertEqual(int(by.loc[name, "n_with_confidence"]), 0)
            self.assertTrue(math.isnan(by.loc[name, "aurc"]))
        for name in ("retrieval_tfidf", "retrieval_bm25", "retrieval_bm25plus", "retrieval_rrf"):
            row = by.loc[name]
            self.assertEqual(int(row["n_items"]), 63)
            self.assertEqual(int(row["n_with_confidence"]), 63)
            self.assertGreaterEqual(row["aurc"], row["aurc_optimal"] - 1e-12)
            self.assertLessEqual(row["aurc_ci_low"], row["aurc"] + 1e-12)
            self.assertGreaterEqual(row["aurc_ci_high"], row["aurc"] - 1e-12)
            c = curves[curves["adapter"] == name]
            self.assertAlmostEqual(c["coverage"].iloc[-1], 1.0)
            # Full-coverage selective accuracy is the headline accuracy.
            self.assertAlmostEqual(c["selective_accuracy"].iloc[-1], row["accuracy_full_coverage"])
        md = render_selective(summary)
        self.assertIn("| retrieval_rrf |", md)
        self.assertIn("n/a", md)

    def test_committed_results_match_the_runner(self) -> None:
        import pandas as pd

        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "selective_summary.csv")
        committed = pd.read_csv(path).set_index("adapter")
        summary, _ = selective(run_all(load_corpus(), n_seeds=1))
        fresh = summary.set_index("adapter")
        for name in ("retrieval_tfidf", "retrieval_bm25", "retrieval_bm25plus", "retrieval_rrf"):
            for col in ("aurc", "e_aurc", "sel_acc_at_50"):
                self.assertAlmostEqual(committed.loc[name, col], fresh.loc[name, col], places=9)


if __name__ == "__main__":
    unittest.main()
