"""Known-answer tests for paired significance (McNemar, paired bootstrap, Holm)."""

from __future__ import annotations

import unittest

from grounding_eval.significance import compare_arms, holm_adjust, paired_difference


class HolmTests(unittest.TestCase):
    def test_hand_computed(self) -> None:
        # Sorted 0.01, 0.02, 0.03, 0.04 -> x4, x3, x2, x1 = 0.04, 0.06, 0.06, 0.04;
        # the running maximum lifts the last to 0.06.
        got = holm_adjust([0.04, 0.01, 0.03, 0.02])
        for a, b in zip(got, [0.06, 0.04, 0.06, 0.06]):
            self.assertAlmostEqual(a, b)

    def test_caps_at_one_and_keeps_order(self) -> None:
        self.assertEqual(holm_adjust([0.9, 0.6]), [1.0, 1.0])
        self.assertEqual(holm_adjust([]), [])
        self.assertAlmostEqual(holm_adjust([0.2])[0], 0.2)

    def test_never_below_raw_and_monotone(self) -> None:
        raw = [0.001, 0.2, 0.049, 0.5, 0.01]
        adj = holm_adjust(raw)
        self.assertTrue(all(a >= r for a, r in zip(adj, raw)))
        ranked = [a for _, a in sorted(zip(raw, adj))]
        self.assertEqual(ranked, sorted(ranked))


class PairedDifferenceTests(unittest.TestCase):
    def test_mcnemar_known_value(self) -> None:
        # 8 discordant, all favouring A: two-sided exact p = 2 * (1/2)**8 = 1/128.
        a = [True] * 8 + [True] * 4
        b = [False] * 8 + [True] * 4
        r = paired_difference(a, b, ["f%d" % i for i in range(12)], n_resamples=200)
        self.assertEqual((r["n_a_only"], r["n_b_only"]), (8, 0))
        self.assertAlmostEqual(r["p_mcnemar"], 1.0 / 128.0)
        self.assertAlmostEqual(r["diff"], 8.0 / 12.0)
        self.assertGreater(r["diff_ci_low"], 0.0)

    def test_balanced_discordance_is_not_significant(self) -> None:
        a = [True, True, True, False, False, False]
        b = [False, False, False, True, True, True]
        r = paired_difference(a, b, ["f%d" % i for i in range(6)], n_resamples=200)
        self.assertAlmostEqual(r["p_mcnemar"], 1.0)
        self.assertAlmostEqual(r["diff"], 0.0)
        self.assertLess(r["diff_ci_low"], 0.0)
        self.assertGreater(r["diff_ci_high"], 0.0)

    def test_identical_arms_have_zero_width_interval(self) -> None:
        a = [True, False, True, False]
        r = paired_difference(a, list(a), ["x", "x", "y", "y"], n_resamples=100)
        self.assertEqual((r["diff"], r["diff_ci_low"], r["diff_ci_high"]), (0.0, 0.0, 0.0))
        self.assertEqual(r["p_mcnemar"], 1.0)

    def test_bootstrap_resamples_whole_clusters(self) -> None:
        # One cluster carries the whole difference; resamples that omit it give 0,
        # so the interval must reach 0 even though the point estimate is positive.
        a = [True, True, False, False]
        b = [False, False, False, False]
        r = paired_difference(a, b, ["f1", "f1", "f2", "f2"], n_resamples=500)
        self.assertAlmostEqual(r["diff"], 0.5)
        self.assertEqual(r["diff_ci_low"], 0.0)

    def test_misaligned_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            paired_difference([True], [True, False], ["a"])

    def test_seeded_interval_is_reproducible(self) -> None:
        a = [True, False, True, True, False, True]
        b = [False, False, True, False, False, False]
        c = ["a", "a", "b", "b", "c", "c"]
        self.assertEqual(paired_difference(a, b, c, 300, 7), paired_difference(a, b, c, 300, 7))


class CompareArmsTests(unittest.TestCase):
    def test_holm_applied_across_family_and_flags(self) -> None:
        strong = [True] * 10 + [False] * 2
        weak = [False] * 10 + [False] * 2
        same = list(weak)
        rows = compare_arms(
            {"s": strong, "w": weak, "x": same},
            ["f%d" % i for i in range(12)],
            [("s", "w"), ("w", "x")],
            n_resamples=100,
        )
        self.assertEqual(rows[0]["n_comparisons"], 2)
        self.assertAlmostEqual(rows[0]["p_mcnemar"], 2 * 0.5 ** 10)
        self.assertAlmostEqual(rows[0]["p_holm"], 2 * 2 * 0.5 ** 10)
        self.assertTrue(rows[0]["significant"])
        self.assertFalse(rows[1]["significant"])
        self.assertEqual(rows[1]["p_holm"], 1.0)


class CommittedResultsTests(unittest.TestCase):
    def test_retriever_pairs_not_significant_and_floor_contrasts_are(self) -> None:
        import pandas as pd
        import os

        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "significance_pairwise.csv")
        t = pd.read_csv(path)
        self.assertEqual(len(t), 10)
        rr = t[t["arm_b"] != "random_floor"]
        self.assertEqual(len(rr), 6)
        self.assertFalse(rr["significant"].any())
        self.assertTrue((rr["n_discordant"] <= 1).all())
        self.assertTrue(t[t["arm_b"] == "random_floor"]["significant"].all())


if __name__ == "__main__":
    unittest.main()
