"""Tests for the human-adjudication export.

The property that matters here is blinding: a rating sheet that leaks the arm or
the automatic outcome is not an independent check on the scorer, and a benchmark
whose only validation is the scorer checking itself has no validation at all.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from grounding_eval.adjudication import agreement, build_sample, write_sample
from grounding_eval.adapters.mock import MockAdapter
from grounding_eval.corpus import load_corpus
from grounding_eval.harness import run_once

LEAKY_FIELDS = ("arm", "adapter", "automatic_outcome", "repeat")


class TestSampleConstruction(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = load_corpus()
        self.runs = [
            run_once(MockAdapter(profile=profile, seed=13), corpus=self.corpus)
            for profile in ("ungrounded", "grounded")
        ]

    def test_sample_is_non_empty(self) -> None:
        sample = build_sample(self.runs, self.corpus)
        self.assertGreater(len(sample), 0)

    def test_sheet_rows_do_not_leak_arm_or_automatic_outcome(self) -> None:
        """The core blinding guarantee."""
        sample = build_sample(self.runs, self.corpus)
        for row in sample.sheet:
            for field in LEAKY_FIELDS:
                self.assertNotIn(field, row, msg="rating sheet leaks %r" % field)

    def test_sheet_carries_what_a_rater_needs(self) -> None:
        sample = build_sample(self.runs, self.corpus)
        for row in sample.sheet:
            for field in ("question", "system_response", "governing_clause", "rating"):
                self.assertIn(field, row)

    def test_key_covers_every_sheet_row_exactly_once(self) -> None:
        sample = build_sample(self.runs, self.corpus)
        sheet_ids = [row["rating_id"] for row in sample.sheet]
        key_ids = [row["rating_id"] for row in sample.key]
        self.assertEqual(sorted(sheet_ids), sorted(key_ids))
        self.assertEqual(len(set(sheet_ids)), len(sheet_ids))

    def test_correct_responses_are_sampled_too(self) -> None:
        """Sampling only errors would leave the scorer's false-negative rate unknown."""
        sample = build_sample(self.runs, self.corpus)
        outcomes = {row["automatic_outcome"] for row in sample.key}
        self.assertIn("correct", outcomes)

    def test_sample_is_reproducible_from_its_seed(self) -> None:
        first = build_sample(self.runs, self.corpus, seed=5)
        second = build_sample(self.runs, self.corpus, seed=5)
        self.assertEqual(
            [r["rating_id"] for r in first.key],
            [r["rating_id"] for r in second.key],
        )
        self.assertEqual(
            [r["item_id"] for r in first.key],
            [r["item_id"] for r in second.key],
        )

    def test_different_seeds_draw_differently(self) -> None:
        first = build_sample(self.runs, self.corpus, seed=1)
        second = build_sample(self.runs, self.corpus, seed=2)
        self.assertNotEqual(
            [r["item_id"] for r in first.key],
            [r["item_id"] for r in second.key],
        )


class TestWriteSample(unittest.TestCase):
    def test_sheet_and_key_are_written_to_separate_files(self) -> None:
        corpus = load_corpus()
        runs = [run_once(MockAdapter(profile="ungrounded", seed=4), corpus=corpus)]
        sample = build_sample(runs, corpus)
        with tempfile.TemporaryDirectory() as tmp:
            sheet_path, key_path = write_sample(sample, tmp)
            self.assertNotEqual(sheet_path, key_path)
            with open(sheet_path, "r", encoding="utf-8") as handle:
                sheet_text = handle.read()
            with open(key_path, "r", encoding="utf-8") as handle:
                key = json.load(handle)
        # The rendered sheet must not contain the arm labels anywhere, including
        # inside free-text fields.
        self.assertNotIn("ungrounded", sheet_text)
        self.assertEqual(len(key["key"]), len(sample))


class TestAgreement(unittest.TestCase):
    def test_perfect_agreement_gives_kappa_of_one(self) -> None:
        human = {"a": "correct", "b": "adjacent_substitution", "c": "correct"}
        result = agreement(human, dict(human))
        self.assertAlmostEqual(result["raw_agreement"], 1.0)
        self.assertAlmostEqual(result["cohens_kappa"], 1.0)

    def test_disagreements_are_listed(self) -> None:
        human = {"a": "correct", "b": "correct"}
        auto = {"a": "correct", "b": "adjacent_substitution"}
        result = agreement(human, auto)
        self.assertEqual(result["n_disagreements"], 1)
        self.assertEqual(result["disagreements"][0]["rating_id"], "b")

    def test_kappa_corrects_for_chance_on_a_dominant_class(self) -> None:
        """Raw agreement can be high while kappa shows no real skill."""
        human = {str(i): "correct" for i in range(20)}
        auto = {str(i): "correct" for i in range(20)}
        result = agreement(human, auto)
        self.assertAlmostEqual(result["raw_agreement"], 1.0)
        # With one class only, expected agreement is 1.0 and kappa is undefined.
        self.assertNotEqual(result["cohens_kappa"], result["cohens_kappa"])  # NaN

    def test_no_shared_ids_is_not_a_crash(self) -> None:
        result = agreement({"a": "correct"}, {"b": "correct"})
        self.assertEqual(result["n"], 0)


if __name__ == "__main__":
    unittest.main()
