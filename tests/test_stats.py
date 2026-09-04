"""Statistics tests.

The statistics here are simple enough to check against values computed by hand,
which is the point: a benchmark whose inference machinery cannot be verified by
hand is one more thing the reader has to take on trust.
"""

from __future__ import annotations

import unittest

from grounding_eval.stats import (
    cluster_bootstrap_ci,
    mcnemar_exact,
    paired_accuracy,
    risk_coverage_curve,
    wilson_interval,
)


class TestMcNemar(unittest.TestCase):
    def test_counts_partition_the_items(self) -> None:
        a = [True, True, False, False, True]
        b = [True, False, True, False, False]
        result = mcnemar_exact(a, b)
        self.assertEqual(result.n_both, 1)
        self.assertEqual(result.n_a_only, 2)
        self.assertEqual(result.n_b_only, 1)
        self.assertEqual(result.n_neither, 1)
        self.assertEqual(
            result.n_both + result.n_a_only + result.n_b_only + result.n_neither,
            len(a),
        )

    def test_identical_arms_give_p_of_one(self) -> None:
        a = [True, False, True, True]
        result = mcnemar_exact(a, list(a))
        self.assertEqual(result.n_discordant, 0)
        self.assertEqual(result.p_value, 1.0)

    def test_known_value_ten_versus_zero(self) -> None:
        """10 discordant items all favouring one arm: p = 2 * 0.5**10."""
        a = [True] * 10
        b = [False] * 10
        result = mcnemar_exact(a, b)
        self.assertAlmostEqual(result.p_value, 2 * (0.5 ** 10), places=12)

    def test_symmetric_discordance_is_not_significant(self) -> None:
        a = [True] * 5 + [False] * 5
        b = [False] * 5 + [True] * 5
        self.assertAlmostEqual(mcnemar_exact(a, b).p_value, 1.0, places=12)

    def test_p_value_is_bounded(self) -> None:
        for n_a in range(0, 12):
            result = mcnemar_exact([True] * n_a + [False] * 3, [False] * n_a + [False] * 3)
            self.assertGreaterEqual(result.p_value, 0.0)
            self.assertLessEqual(result.p_value, 1.0)

    def test_misaligned_sequences_raise(self) -> None:
        with self.assertRaises(ValueError):
            mcnemar_exact([True, False], [True])


class TestWilson(unittest.TestCase):
    def test_interval_brackets_the_point_estimate(self) -> None:
        ci = wilson_interval(7, 10)
        self.assertLessEqual(ci.low, ci.point)
        self.assertGreaterEqual(ci.high, ci.point)

    def test_bounds_stay_inside_zero_one_at_the_extremes(self) -> None:
        """The reason Wilson is used rather than the Wald interval."""
        for successes, n in ((0, 20), (20, 20), (1, 3)):
            ci = wilson_interval(successes, n)
            self.assertGreaterEqual(ci.low, 0.0)
            self.assertLessEqual(ci.high, 1.0)

    def test_zero_n_is_not_a_crash(self) -> None:
        ci = wilson_interval(0, 0)
        self.assertEqual(ci.n, 0)


class TestClusterBootstrap(unittest.TestCase):
    def test_is_reproducible_from_its_seed(self) -> None:
        values = [1.0, 0.0] * 20
        clusters = ["f%d" % (i // 2) for i in range(40)]
        first = cluster_bootstrap_ci(values, clusters, n_resamples=200, seed=7)
        second = cluster_bootstrap_ci(values, clusters, n_resamples=200, seed=7)
        self.assertEqual(first.low, second.low)
        self.assertEqual(first.high, second.high)

    def test_point_estimate_is_the_observed_mean(self) -> None:
        values = [1.0, 1.0, 0.0, 0.0]
        clusters = ["a", "a", "b", "b"]
        result = cluster_bootstrap_ci(values, clusters, n_resamples=100, seed=1)
        self.assertAlmostEqual(result.point, 0.5)

    def test_clustering_widens_the_interval_versus_ignoring_it(self) -> None:
        """The reason the clustered interval is the one reported.

        With outcomes perfectly correlated inside clusters, resampling clusters
        must produce a wider interval than treating every item as independent.
        Reporting the narrow one would overstate the precision of every result.
        """
        values = [1.0] * 20 + [0.0] * 20
        clustered = ["c%d" % (i // 4) for i in range(40)]
        singleton = ["s%d" % i for i in range(40)]
        wide = cluster_bootstrap_ci(values, clustered, n_resamples=800, seed=3)
        narrow = cluster_bootstrap_ci(values, singleton, n_resamples=800, seed=3)
        self.assertGreater(wide.high - wide.low, narrow.high - narrow.low)

    def test_misaligned_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            cluster_bootstrap_ci([1.0, 0.0], ["a"], n_resamples=10)


class TestPairedAccuracy(unittest.TestCase):
    def test_requires_both_members(self) -> None:
        success = {"a1": True, "b1": False, "a2": True, "b2": True}
        both, evaluated = paired_accuracy(success, [("a1", "b1"), ("a2", "b2")])
        self.assertEqual((both, evaluated), (1, 2))

    def test_is_stricter_than_per_item_accuracy(self) -> None:
        """The property that makes paired accuracy the primary statistic."""
        success = {"a1": True, "b1": False, "a2": True, "b2": False}
        item_accuracy = sum(success.values()) / len(success)
        both, evaluated = paired_accuracy(success, [("a1", "b1"), ("a2", "b2")])
        self.assertEqual(item_accuracy, 0.5)
        self.assertEqual(both / evaluated, 0.0)

    def test_missing_items_shrink_the_denominator_rather_than_counting_as_failures(self) -> None:
        success = {"a1": True, "b1": True}
        both, evaluated = paired_accuracy(success, [("a1", "b1"), ("a2", "b2")])
        self.assertEqual((both, evaluated), (1, 1))


class TestRiskCoverage(unittest.TestCase):
    def test_curve_is_ordered_by_descending_confidence(self) -> None:
        curve = risk_coverage_curve([0.9, 0.5, 0.7], [True, False, True])
        thresholds = [p["confidence_threshold"] for p in curve]
        self.assertEqual(thresholds, sorted(thresholds, reverse=True))

    def test_full_coverage_accuracy_equals_overall_accuracy(self) -> None:
        curve = risk_coverage_curve([0.9, 0.5, 0.7, 0.2], [True, False, True, False])
        self.assertAlmostEqual(curve[-1]["coverage"], 1.0)
        self.assertAlmostEqual(curve[-1]["accuracy"], 0.5)

    def test_no_confidence_reported_yields_an_empty_curve(self) -> None:
        # A system that reports no confidence offers no way to trade coverage
        # for precision. The empty curve is the honest representation of that.
        self.assertEqual(risk_coverage_curve([None, None], [True, False]), [])


if __name__ == "__main__":
    unittest.main()
