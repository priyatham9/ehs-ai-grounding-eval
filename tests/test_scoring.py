"""Scoring tests.

The scoring rule is the benchmark's measuring instrument. These tests pin down
each branch of the decision procedure with hand-written responses, so that a
change in scoring behaviour shows up as a failing test rather than as a shifted
number in a report.
"""

from __future__ import annotations

import unittest

from grounding_eval.corpus import load_corpus
from grounding_eval.schema import AdapterResponse, ItemType, Outcome
from grounding_eval.scoring import (
    ScoringConfig,
    assess_adjacency,
    assess_concepts,
    detect_abstention,
    score_item,
    score_responses,
)
from grounding_eval.scoring.score import utility_of


def response(item_id: str, text: str, **kwargs) -> AdapterResponse:
    return AdapterResponse(item_id=item_id, text=text, **kwargs)


class TestAbstention(unittest.TestCase):
    def test_plain_decline_is_an_abstention(self) -> None:
        signal = detect_abstention("I don't know the answer to that.")
        self.assertTrue(signal.abstained)
        self.assertFalse(signal.empty)

    def test_empty_text_is_empty_not_abstention(self) -> None:
        # An empty response is an infrastructure event, not a considered decline.
        signal = detect_abstention("   ")
        self.assertTrue(signal.empty)

    def test_a_decline_followed_by_a_substantive_answer_is_not_an_abstention(self) -> None:
        """Hedging then answering is answering.

        Crediting "I'm not certain, but the set pressure is 150 psig" as an
        abstention would let a system collect abstention credit for a confident
        wrong answer wearing a disclaimer.
        """
        text = (
            "I am not certain about this, but the blowdown is 7 percent and the "
            "valve reseats at that point, per the applicable standard."
        )
        self.assertFalse(detect_abstention(text).abstained)


class TestConceptsAndAdjacency(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = load_corpus()
        self.item = self.corpus.by_id("prd-001")

    def test_reference_answer_satisfies_its_own_key(self) -> None:
        """Every item's own reference answer must score as correct.

        If a reference answer fails its own key, the key is wrong, and every
        system measured against it is being measured against a broken target.
        """
        for item in self.corpus:
            assessment = assess_concepts(item.correct_answer, item.answer_key)
            self.assertGreaterEqual(
                assessment.coverage,
                ScoringConfig().coverage_threshold,
                msg="%s: its own reference answer does not satisfy its key "
                "(coverage %.2f)" % (item.id, assessment.coverage),
            )
            self.assertFalse(
                assessment.has_forbidden,
                msg="%s: its own reference answer trips a forbidden concept (%s)"
                % (item.id, assessment.forbidden_asserted),
            )

    def test_reference_answers_score_correct_end_to_end(self) -> None:
        """The full decision procedure, not only the concept component."""
        for item in self.corpus:
            if item.item_type is ItemType.CATEGORY_ERROR:
                continue  # correct behaviour there is to decline, not to answer
            score = score_item(item, response(item.id, item.correct_answer))
            self.assertIs(
                score.outcome,
                Outcome.CORRECT,
                msg="%s: reference answer scored %s" % (item.id, score.outcome),
            )

    def test_adjacent_answer_is_detected(self) -> None:
        text = (
            "The device is set at the maximum allowable working pressure with a "
            "blowdown of 7 percent, and it reseats once pressure falls. The spring "
            "determines the set pressure."
        )
        assessment = assess_adjacency(text, self.item.adjacent_wrong)
        self.assertGreaterEqual(assessment.n_asserted, 2)

    def test_negated_adjacent_term_is_not_an_assertion(self) -> None:
        """Saying a rupture disk has no blowdown is correct, not a substitution.

        Without this, a system that explicitly draws the right distinction would
        be penalised for naming the thing it is ruling out.
        """
        text = "A rupture disk is non-reclosing: it has no blowdown and it does not reseat."
        assessment = assess_adjacency(text, self.item.adjacent_wrong)
        self.assertEqual(assessment.n_asserted, 0, msg=str(assessment.asserted_terms))


class TestScoreBranches(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = load_corpus()
        self.item = self.corpus.by_id("prd-001")

    def test_adapter_error_is_unscorable_not_wrong(self) -> None:
        score = score_item(self.item, response(self.item.id, "", error="HTTP 503"))
        self.assertIs(score.outcome, Outcome.UNSCORABLE)
        self.assertIn("503", score.unscorable_reason or "")

    def test_empty_response_is_unscorable(self) -> None:
        score = score_item(self.item, response(self.item.id, ""))
        self.assertIs(score.outcome, Outcome.UNSCORABLE)

    def test_abstention_branch(self) -> None:
        score = score_item(self.item, response(self.item.id, "I don't know."))
        self.assertIs(score.outcome, Outcome.ABSTAINED)

    def test_adjacent_substitution_branch(self) -> None:
        text = (
            "Set the device at 150 psig with a 7 percent blowdown; the valve will "
            "reseat after the event and the spring holds the disc on its seat."
        )
        score = score_item(self.item, response(self.item.id, text))
        self.assertIs(score.outcome, Outcome.ADJACENT_SUBSTITUTION)

    def test_other_incorrect_branch(self) -> None:
        score = score_item(self.item, response(self.item.id, "Consult your site engineer."))
        self.assertIs(score.outcome, Outcome.OTHER_INCORRECT)

    def test_wrong_number_is_not_partial_credit(self) -> None:
        """A right explanation with a wrong number is not a correct answer."""
        text = (
            "The normal maximum operating pressure multiplied by 1.9 must not exceed "
            "the nominal disk burst pressure, to avoid fatigue of the disk at "
            "operating pressure."
        )
        score = score_item(self.item, response(self.item.id, text))
        self.assertIsNot(score.outcome, Outcome.CORRECT)

    def test_score_responses_rejects_unknown_item_ids(self) -> None:
        with self.assertRaises(KeyError):
            score_responses(self.corpus, [response("no-such-item", "text")])


class TestUtilityPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = load_corpus()

    def test_abstention_beats_confabulation_under_the_safety_policy(self) -> None:
        """The stated ordering the benchmark exists to encode."""
        item = self.corpus.by_id("prd-001")
        config = ScoringConfig(policy="safety")
        abstain = score_item(item, response(item.id, "I don't know."))
        adjacent = score_item(
            item,
            response(item.id, "Set pressure with 7 percent blowdown; the valve reseats and the spring resets."),
        )
        self.assertGreater(utility_of(abstain, item, config), utility_of(adjacent, item, config))

    def test_declining_a_category_error_item_earns_near_full_credit(self) -> None:
        item = self.corpus.category_errors()[0]
        config = ScoringConfig(policy="safety")
        score = score_item(item, response(item.id, "That question rests on a false premise; I can't answer it as asked."))
        self.assertIs(score.outcome, Outcome.ABSTAINED)
        self.assertGreaterEqual(utility_of(score, item, config), 0.5)

    def test_unknown_policy_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ScoringConfig(policy="nonexistent").utility_table()

    def test_accuracy_policy_ignores_abstention(self) -> None:
        item = self.corpus.by_id("prd-001")
        config = ScoringConfig(policy="accuracy")
        score = score_item(item, response(item.id, "I don't know."))
        self.assertEqual(utility_of(score, item, config), 0.0)


class TestDeterminism(unittest.TestCase):
    def test_scoring_the_same_text_twice_gives_the_same_result(self) -> None:
        corpus = load_corpus()
        item = corpus.by_id("loto-001")
        text = "Some partially relevant answer mentioning periodic inspection annually."
        first = score_item(item, response(item.id, text))
        second = score_item(item, response(item.id, text))
        self.assertEqual(first.to_dict(), second.to_dict())


if __name__ == "__main__":
    unittest.main()
