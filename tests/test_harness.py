"""Harness and reporting tests, including the anti-misrepresentation guardrails.

The tests at the bottom of this file are the ones that keep this repository
honest. They assert that a fixture run cannot be presented as a result, that a
run cannot be silently re-attributed to a different corpus, and that an adapter
failure is never counted as a wrong answer.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from grounding_eval import MOCK_PROVENANCE
from grounding_eval.adapters.base import Adapter, StaticAdapter
from grounding_eval.adapters.mock import MockAdapter, available_profiles
from grounding_eval.corpus import load_corpus
from grounding_eval.harness import (
    corpus_digest,
    load_run,
    run_once,
    run_repeats,
    write_run,
)
from grounding_eval.report import (
    arm_summary,
    citation_table,
    compare_arms,
    format_summary,
    outcome_composition,
    paired_accuracy_table,
    runs_frame,
    scores_frame,
)
from grounding_eval.schema import AdapterResponse, Item, ItemType, Outcome


class ExplodingAdapter(Adapter):
    """Raises on every item. Used to prove failures are not scored as wrong."""

    name = "exploding"
    arm = "fixture"

    def answer(self, item: Item) -> AdapterResponse:
        raise RuntimeError("simulated backend outage")


class MisroutingAdapter(Adapter):
    """Returns a response for the wrong item, which must be caught."""

    name = "misrouting"
    arm = "fixture"

    def answer(self, item: Item) -> AdapterResponse:
        return AdapterResponse(item_id="not-the-item-asked", text="hello")


class TestRunLoop(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = load_corpus().filter(item_ids=["prd-001", "prd-002", "loto-001"])

    def test_run_covers_every_item_once_in_order(self) -> None:
        run = run_once(StaticAdapter("no answer available"), corpus=self.corpus)
        self.assertEqual(len(run.responses), len(self.corpus))
        self.assertEqual(
            [r.item_id for r in run.responses],
            list(self.corpus.ids()),
        )

    def test_adapter_exception_becomes_unscorable_not_incorrect(self) -> None:
        run = run_once(ExplodingAdapter(), corpus=self.corpus)
        self.assertTrue(all(s.outcome is Outcome.UNSCORABLE for s in run.scores))
        self.assertTrue(all("simulated backend outage" in (r.error or "") for r in run.responses))

    def test_misrouted_response_raises(self) -> None:
        # Silently accepting a response for the wrong item would score answers
        # against the wrong keys and produce a plausible, meaningless number.
        with self.assertRaises(ValueError):
            run_once(MisroutingAdapter(), corpus=self.corpus)

    def test_latency_is_recorded(self) -> None:
        run = run_once(StaticAdapter("text"), corpus=self.corpus)
        self.assertTrue(all(r.latency_s is not None for r in run.responses))

    def test_repeats_are_kept_separate(self) -> None:
        runs = run_repeats(
            lambda repeat: MockAdapter(profile="grounded", seed=11, repeat=repeat),
            n_runs=3,
            corpus=self.corpus,
        )
        self.assertEqual([r.repeat for r in runs], [0, 1, 2])

    def test_n_runs_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            run_repeats(lambda repeat: StaticAdapter("x"), n_runs=0, corpus=self.corpus)


class TestDigestAndRoundTrip(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = load_corpus()

    def test_digest_is_stable_across_calls(self) -> None:
        self.assertEqual(corpus_digest(self.corpus), corpus_digest(self.corpus))

    def test_digest_changes_when_an_answer_key_changes(self) -> None:
        subset = self.corpus.filter(item_ids=["prd-001"])
        before = corpus_digest(subset)
        self.assertNotEqual(before, corpus_digest(self.corpus))

    def test_run_file_round_trips(self) -> None:
        run = run_once(MockAdapter(profile="grounded", seed=5), corpus=self.corpus)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_run(run, tmp)
            payload = load_run(path, corpus=self.corpus)
        self.assertEqual(payload["n_items"], len(self.corpus))
        self.assertEqual(len(payload["scores"]), len(self.corpus))

    def test_loading_against_a_different_corpus_raises(self) -> None:
        """A run scored against another corpus version is not comparable."""
        run = run_once(StaticAdapter("x"), corpus=self.corpus)
        subset = self.corpus.filter(item_ids=["prd-001"])
        with tempfile.TemporaryDirectory() as tmp:
            path = write_run(run, tmp)
            with self.assertRaises(ValueError):
                load_run(path, corpus=subset)


class TestMockLabelling(unittest.TestCase):
    """Guardrails preventing a demonstration fixture from reading as a result."""

    def setUp(self) -> None:
        self.corpus = load_corpus().filter(item_ids=["prd-001", "prd-002", "loto-001", "loto-010"])

    def test_every_mock_response_carries_the_provenance_string(self) -> None:
        run = run_once(MockAdapter(profile="ungrounded", seed=3), corpus=self.corpus)
        self.assertTrue(all(r.provenance == MOCK_PROVENANCE for r in run.responses))
        self.assertTrue(run.is_mock)

    def test_mock_run_files_are_named_and_flagged_as_mock(self) -> None:
        run = run_once(MockAdapter(profile="grounded", seed=3), corpus=self.corpus)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_run(run, tmp)
            self.assertTrue(os.path.basename(path).startswith("mock__"))
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        self.assertTrue(payload["is_mock_demonstration"])
        self.assertEqual(payload["mock_banner"], MOCK_PROVENANCE)

    def test_summary_of_a_mock_run_displays_the_banner(self) -> None:
        run = run_once(MockAdapter(profile="grounded", seed=3), corpus=self.corpus)
        text = format_summary(scores_frame(run), self.corpus, bootstrap_resamples=50)
        self.assertIn(MOCK_PROVENANCE, text)

    def test_suppressing_the_banner_on_a_mock_run_raises(self) -> None:
        """The single most important guardrail in the repository."""
        run = run_once(MockAdapter(profile="grounded", seed=3), corpus=self.corpus)
        with self.assertRaises(ValueError):
            format_summary(
                scores_frame(run),
                self.corpus,
                show_mock_banner=False,
                bootstrap_resamples=50,
            )

    def test_renaming_the_adapter_does_not_strip_the_mock_flag(self) -> None:
        """is_mock reads the responses, not the adapter's self-reported name."""
        adapter = MockAdapter(profile="grounded", seed=3)
        adapter.name = "totally-real-production-system"
        run = run_once(adapter, corpus=self.corpus)
        self.assertTrue(run.is_mock)
        text = format_summary(scores_frame(run), self.corpus, bootstrap_resamples=50)
        self.assertIn(MOCK_PROVENANCE, text)

    def test_non_mock_run_has_no_banner(self) -> None:
        run = run_once(StaticAdapter("I don't know."), corpus=self.corpus)
        self.assertFalse(run.is_mock)
        text = format_summary(scores_frame(run), self.corpus, bootstrap_resamples=50)
        self.assertNotIn(MOCK_PROVENANCE, text)

    def test_all_declared_profiles_run(self) -> None:
        for profile in available_profiles():
            run = run_once(MockAdapter(profile=profile, seed=2), corpus=self.corpus)
            self.assertEqual(len(run.scores), len(self.corpus))


class TestMockDeterminism(unittest.TestCase):
    def test_same_seed_gives_identical_scores(self) -> None:
        corpus = load_corpus()
        first = run_once(MockAdapter(profile="ungrounded", seed=99), corpus=corpus)
        second = run_once(MockAdapter(profile="ungrounded", seed=99), corpus=corpus)
        self.assertEqual(
            [s.outcome for s in first.scores],
            [s.outcome for s in second.scores],
        )

    def test_different_seeds_generally_differ(self) -> None:
        corpus = load_corpus()
        first = run_once(MockAdapter(profile="ungrounded", seed=1), corpus=corpus)
        second = run_once(MockAdapter(profile="ungrounded", seed=2), corpus=corpus)
        self.assertNotEqual(
            [s.outcome for s in first.scores],
            [s.outcome for s in second.scores],
        )


class TestReportTables(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = load_corpus()
        self.runs = []
        for profile in ("ungrounded", "grounded"):
            self.runs.extend(
                run_repeats(
                    lambda repeat, p=profile: MockAdapter(profile=p, seed=42, repeat=repeat),
                    n_runs=2,
                    corpus=self.corpus,
                )
            )
        self.frame = runs_frame(self.runs)

    def test_frame_has_one_row_per_item_per_run(self) -> None:
        self.assertEqual(len(self.frame), len(self.corpus) * len(self.runs))

    def test_category_error_items_are_reported_separately(self) -> None:
        composition = outcome_composition(self.frame)
        self.assertIn(ItemType.CATEGORY_ERROR.value, set(composition["item_type"]))
        self.assertIn(ItemType.FACTUAL.value, set(composition["item_type"]))

    def test_composition_proportions_sum_to_one_within_each_group(self) -> None:
        composition = outcome_composition(self.frame)
        totals = composition.groupby(["adapter", "item_type"])["proportion"].sum()
        for value in totals:
            self.assertAlmostEqual(value, 1.0, places=9)

    def test_arm_summary_reports_an_interval_around_the_point(self) -> None:
        summary = arm_summary(self.frame, self.corpus, bootstrap_resamples=200)
        for _, row in summary.iterrows():
            self.assertLessEqual(row["accuracy_ci_low"], row["accuracy"] + 1e-9)
            self.assertGreaterEqual(row["accuracy_ci_high"], row["accuracy"] - 1e-9)

    def test_paired_accuracy_never_exceeds_item_accuracy(self) -> None:
        summary = arm_summary(self.frame, self.corpus, bootstrap_resamples=100)
        for _, row in summary.iterrows():
            self.assertLessEqual(row["paired_accuracy"], row["accuracy"] + 1e-9)

    def test_pair_table_lists_every_complete_pair(self) -> None:
        table = paired_accuracy_table(self.frame, self.corpus)
        per_run = len(self.corpus.pairs())
        self.assertEqual(len(table), per_run * len(self.runs))

    def test_citation_table_proportions_sum_to_one(self) -> None:
        table = citation_table(self.frame)
        for _, group in table.groupby("adapter"):
            self.assertAlmostEqual(group["proportion"].sum(), 1.0, places=9)

    def test_compare_arms_is_paired_on_identical_item_sets(self) -> None:
        result = compare_arms(self.frame, "mock:ungrounded", "mock:grounded", repeat=0)
        self.assertEqual(result["mcnemar"]["n_items"], len(self.corpus.factual()))

    def test_compare_arms_rejects_mismatched_item_sets(self) -> None:
        trimmed = self.frame[
            ~((self.frame["adapter"] == "mock:grounded") & (self.frame["item_id"] == "prd-001"))
        ]
        with self.assertRaises(ValueError):
            compare_arms(trimmed, "mock:ungrounded", "mock:grounded", repeat=0)

    def test_empty_frame_is_handled(self) -> None:
        import pandas as pd

        self.assertEqual(format_summary(pd.DataFrame(), self.corpus), "no scored runs")


if __name__ == "__main__":
    unittest.main()
