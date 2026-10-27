"""Tests for calibration metrics (Brier score, ECE, reliability bins)."""

from __future__ import annotations

import math
import os
import unittest

from grounding_eval.calibration import (
    brier_constant,
    brier_score,
    expected_calibration_error,
    reliability_table,
    summarize_calibration,
)
from grounding_eval.corpus import load_corpus
from grounding_eval.run_baselines import CALIBRATION_SCORED, calibration, render_calibration, run_all


class CalibrationMetricTests(unittest.TestCase):
    def test_hand_computed_brier(self) -> None:
        # (0.9-1)^2 + (0.8-1)^2 + (0.3-0)^2 + (0.2-1)^2 = 0.01 + 0.04 + 0.09 + 0.64 = 0.78; mean 0.195.
        conf, ok = [0.9, 0.8, 0.3, 0.2], [True, True, False, True]
        self.assertAlmostEqual(brier_score(conf, ok), 0.195)

    def test_brier_extremes(self) -> None:
        self.assertAlmostEqual(brier_score([1.0, 0.0], [True, False]), 0.0)
        self.assertAlmostEqual(brier_score([0.0, 1.0], [True, False]), 1.0)
        # Always saying 0.5 scores 0.25 whatever the outcomes.
        self.assertAlmostEqual(brier_score([0.5] * 4, [True, False, True, True]), 0.25)

    def test_brier_constant_is_p_times_one_minus_p(self) -> None:
        self.assertAlmostEqual(brier_constant([0.1, 0.2, 0.3, 0.4], [True, False, False, False]), 0.25 * 0.75)
        # Reporting the base rate everywhere attains exactly that value.
        self.assertAlmostEqual(brier_score([0.25] * 4, [True, False, False, False]), 0.25 * 0.75)

    def test_hand_computed_ece(self) -> None:
        # Bin [0.8, 0.9): confidences 0.85, 0.85 with 1 of 2 correct -> |0.5 - 0.85| = 0.35, weight 2/4.
        # Bin [0.1, 0.2): confidence 0.15, 0 of 1 correct -> 0.15, weight 1/4.
        # Bin [0.5, 0.6): confidence 0.55, 1 of 1 correct -> 0.45, weight 1/4.
        conf, ok = [0.85, 0.85, 0.15, 0.55], [True, False, False, True]
        expected = 0.5 * 0.35 + 0.25 * 0.15 + 0.25 * 0.45
        self.assertAlmostEqual(expected_calibration_error(conf, ok), expected)

    def test_perfectly_calibrated_has_zero_ece(self) -> None:
        # Ten items at 0.8, eight correct; ten items at 0.2, two correct.
        conf = [0.8] * 10 + [0.2] * 10
        ok = [True] * 8 + [False] * 2 + [True] * 2 + [False] * 8
        self.assertAlmostEqual(expected_calibration_error(conf, ok), 0.0)

    def test_confidently_wrong_has_ece_equal_to_confidence(self) -> None:
        self.assertAlmostEqual(expected_calibration_error([0.95] * 5, [False] * 5), 0.95)

    def test_one_goes_in_the_last_bin_and_bins_partition(self) -> None:
        rows = reliability_table([1.0, 0.0, 0.1, 0.5], [True, False, False, True], n_bins=10)
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[9]["count"], 1)
        self.assertEqual(rows[0]["count"], 1)
        self.assertEqual(rows[1]["count"], 1)  # 0.1 opens the second bin
        self.assertEqual(rows[5]["count"], 1)
        self.assertEqual(sum(r["count"] for r in rows), 4)
        self.assertTrue(math.isnan(rows[3]["accuracy"]))

    def test_missing_confidence_dropped_and_out_of_range_rejected(self) -> None:
        self.assertTrue(math.isnan(brier_score([None], [True])))
        self.assertTrue(math.isnan(expected_calibration_error([None, None], [True, False])))
        self.assertAlmostEqual(brier_score([None, 1.0], [False, True]), 0.0)
        with self.assertRaises(ValueError):
            brier_score([1.2], [True])
        with self.assertRaises(ValueError):
            expected_calibration_error([-0.1], [True])
        with self.assertRaises(ValueError):
            brier_score([0.5], [True, False])
        with self.assertRaises(ValueError):
            reliability_table([0.5], [True], n_bins=0)

    def test_summary_is_seeded_and_brackets_the_point(self) -> None:
        conf = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        ok = [True, True, False, True, False, False, True, False]
        fam = ["a", "a", "b", "b", "c", "c", "d", "d"]
        r1 = summarize_calibration(conf, ok, fam, n_resamples=200, seed=1)
        r2 = summarize_calibration(conf, ok, fam, n_resamples=200, seed=1)
        self.assertEqual(r1, r2)
        self.assertAlmostEqual(r1["brier"], brier_score(conf, ok))
        self.assertAlmostEqual(r1["ece"], expected_calibration_error(conf, ok))
        self.assertLessEqual(r1["brier_ci_low"], r1["brier_ci_high"])
        empty = summarize_calibration([None, None], [True, False], ["a", "b"])
        self.assertEqual(empty["n_with_confidence"], 0)
        self.assertTrue(math.isnan(empty["brier"]))


class CalibrationBaselineTests(unittest.TestCase):
    def test_only_probability_scale_systems_are_scored(self) -> None:
        runs = run_all(load_corpus(), n_seeds=1)
        summary, bins = calibration(runs)
        by = summary.set_index("adapter")
        self.assertEqual(CALIBRATION_SCORED, ("retrieval_tfidf",))
        for name in ("random_floor", "oracle", "retrieval_bm25", "retrieval_bm25plus", "retrieval_rrf"):
            self.assertFalse(bool(by.loc[name, "computed"]))
            self.assertTrue(math.isnan(by.loc[name, "brier"]))
        row = by.loc["retrieval_tfidf"]
        self.assertTrue(bool(row["computed"]))
        self.assertEqual(int(row["n_with_confidence"]), 63)
        self.assertEqual(int(bins[bins["adapter"] == "retrieval_tfidf"]["count"].sum()), 63)
        self.assertGreaterEqual(row["ece"], 0.0)
        md = render_calibration(summary, bins)
        self.assertIn("not on a [0, 1] scale", md)
        self.assertIn("no confidence reported", md)

    def test_committed_results_match_the_runner(self) -> None:
        import pandas as pd

        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "calibration_summary.csv")
        committed = pd.read_csv(path).set_index("adapter")
        fresh, _ = calibration(run_all(load_corpus(), n_seeds=1))
        fresh = fresh.set_index("adapter")
        for col in ("brier", "ece", "mean_confidence"):
            self.assertAlmostEqual(committed.loc["retrieval_tfidf", col], fresh.loc["retrieval_tfidf", col], places=9)


if __name__ == "__main__":
    unittest.main()
