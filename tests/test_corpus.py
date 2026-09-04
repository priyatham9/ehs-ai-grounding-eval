"""Corpus integrity tests.

These are the tests that matter most. The corpus is the part of this repository
that makes factual assertions about federal regulations, so it is the part that
can be wrong in a way that misleads someone. Everything here checks a property
that, if violated, would mean the benchmark is asserting something it has not
established.
"""

from __future__ import annotations

import json
import os
import unittest

from grounding_eval.corpus import (
    DEFAULT_ITEMS_DIR,
    QUARANTINE_PATH,
    REPO_ROOT,
    load_corpus,
    load_quarantine,
    summarize,
)
from grounding_eval.schema import ItemType, VerificationStatus, parse_citations

VERIFICATION_PATH = os.path.join(REPO_ROOT, "corpus", "verification", "source_verification.json")


class TestCorpusLoads(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = load_corpus()

    def test_corpus_is_large_enough_to_be_worth_running(self) -> None:
        # The design target was at least 40 items. Falling below that would mean
        # items were silently dropped.
        self.assertGreaterEqual(len(self.corpus), 40)

    def test_ids_are_unique_and_sorted(self) -> None:
        ids = list(self.corpus.ids())
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, sorted(ids))

    def test_every_item_is_verified(self) -> None:
        # The loader is supposed to refuse unverified items. This asserts the
        # outcome directly rather than trusting the loader.
        for item in self.corpus:
            self.assertIs(
                item.verification.status,
                VerificationStatus.VERIFIED,
                msg="%s is not marked verified" % item.id,
            )

    def test_minimal_pairs_are_complete_and_symmetric(self) -> None:
        pairs = self.corpus.pairs()
        self.assertGreaterEqual(len(pairs), 15)
        for a, b in pairs:
            self.assertEqual(a.pair_id, b.pair_id)
            self.assertEqual(a.pair_role, "a")
            self.assertEqual(b.pair_role, "b")
            self.assertNotEqual(a.id, b.id)
            # A pair whose two questions are identical is not a minimal pair.
            self.assertNotEqual(a.question.strip(), b.question.strip())

    def test_corpus_contains_category_error_items(self) -> None:
        # Without false-premise items, a system that answers everything
        # confidently can outscore a calibrated one.
        self.assertGreaterEqual(len(self.corpus.category_errors()), 3)

    def test_category_error_items_accept_abstention(self) -> None:
        for item in self.corpus.category_errors():
            self.assertTrue(
                item.abstention_acceptable,
                msg="%s is a category error but does not accept declining" % item.id,
            )

    def test_answer_keys_are_usable(self) -> None:
        for item in self.corpus:
            self.assertTrue(
                item.answer_key.required_concepts,
                msg="%s has no required concepts, so it can never be scored correct" % item.id,
            )
            for group in item.answer_key.required_concepts:
                self.assertTrue(all(s.strip() for s in group), msg=item.id)

    def test_adjacent_wrong_is_populated(self) -> None:
        # The adjacency signature is the instrument. An item without one cannot
        # detect the failure mode this benchmark exists to measure.
        for item in self.corpus:
            self.assertTrue(item.adjacent_wrong.signature_terms, msg=item.id)
            self.assertTrue(item.adjacent_wrong.why_dangerous.strip(), msg=item.id)

    def test_adjacent_signature_terms_do_not_appear_in_the_correct_answer(self) -> None:
        """An adjacency term that occurs in the reference answer is a broken item.

        If the correct answer itself contains the wrong entity's signature
        vocabulary, then a system that answers correctly could be scored as an
        adjacent substitution. That would make the headline statistic wrong in
        the direction that flatters the benchmark, so it is checked explicitly.
        """
        for item in self.corpus:
            reference = item.correct_answer.lower()
            for term in item.adjacent_wrong.signature_terms:
                self.assertNotIn(
                    term.lower(),
                    reference,
                    msg="%s: adjacency term %r appears in its own correct answer" % (item.id, term),
                )

    def test_source_clause_parses_into_a_citation(self) -> None:
        for item in self.corpus:
            refs = parse_citations(item.source.clause)
            self.assertTrue(
                refs,
                msg="%s: source clause %r does not parse" % (item.id, item.source.clause),
            )

    def test_sources_are_public(self) -> None:
        """No item's answer key may depend on copyrighted standard text."""
        for item in self.corpus:
            self.assertEqual(
                item.source.access,
                "public",
                msg="%s cites a non-public source as its primary authority" % item.id,
            )

    def test_anchor_text_is_present_and_substantial(self) -> None:
        for item in self.corpus:
            anchor = item.source.anchor_text.strip()
            self.assertGreaterEqual(
                len(anchor), 20, msg="%s: anchor text is too short to verify" % item.id
            )

    def test_summary_is_self_consistent(self) -> None:
        info = summarize(self.corpus)
        self.assertEqual(info["n_items"], len(self.corpus))
        self.assertEqual(info["n_factual"] + info["n_category_error"], info["n_items"])
        self.assertEqual(sum(info["by_domain"].values()), info["n_items"])


class TestSourceVerification(unittest.TestCase):
    """The corpus must be covered by a stored, dated verification report."""

    def setUp(self) -> None:
        self.corpus = load_corpus()
        with open(VERIFICATION_PATH, "r", encoding="utf-8") as handle:
            self.report = json.load(handle)

    def test_report_exists_and_records_no_failures(self) -> None:
        self.assertEqual(self.report["n_failed"], 0, msg=str(self.report.get("results"))[:2000])

    def test_every_corpus_item_appears_in_the_report(self) -> None:
        verified = {r["item_id"] for r in self.report["results"]}
        missing = sorted(set(self.corpus.ids()) - verified)
        self.assertEqual(missing, [], msg="items with no source verification: %s" % missing)

    def test_report_clause_matches_current_corpus_clause(self) -> None:
        """Catches an item whose clause was edited after it was verified."""
        by_id = {r["item_id"]: r for r in self.report["results"]}
        for item in self.corpus:
            self.assertEqual(
                by_id[item.id]["clause"],
                item.source.clause,
                msg="%s: clause changed since verification" % item.id,
            )

    def test_offline_recheck_passes(self) -> None:
        import sys

        sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
        from verify_sources import recheck_offline

        ok, problems = recheck_offline(VERIFICATION_PATH)
        self.assertTrue(ok, msg="\n".join(problems))

    def test_no_anchor_sits_outside_its_cited_paragraph(self) -> None:
        """An item must not cite one paragraph and anchor on another's text.

        The anchor-present check alone does not catch this: it only asks whether
        the anchor appears somewhere in the section. One item in this corpus
        cited 1910.157(d) while anchoring on text that lives in (e)(2), and it
        passed verification. The online verifier now records locality per item
        and this test holds the stored report to it.
        """
        warnings = self.report.get("anchor_locality_warnings")
        self.assertIsNotNone(
            warnings,
            msg="verification report predates the anchor-locality check; "
            "re-run tools/verify_sources.py",
        )
        self.assertEqual(
            warnings,
            [],
            msg="anchor text found outside the cited paragraph: %s" % warnings,
        )


class TestQuarantine(unittest.TestCase):
    def test_quarantine_is_declared_excluded(self) -> None:
        payload = load_quarantine()
        self.assertEqual(payload["status"], "EXCLUDED_FROM_SCORING")

    def test_quarantined_ids_are_not_in_the_scored_corpus(self) -> None:
        payload = load_quarantine()
        scored = set(load_corpus().ids())
        for entry in payload.get("items", []):
            self.assertNotIn(
                entry.get("id"),
                scored,
                msg="%s is quarantined but is also being scored" % entry.get("id"),
            )

    def test_quarantined_items_state_why(self) -> None:
        """A quarantined item must record why it could not be verified.

        Exclusion without a stated reason is indistinguishable from an item that
        was dropped because it was inconvenient.
        """
        for entry in load_quarantine().get("items", []):
            reason = str(entry.get("why_excluded") or entry.get("reason") or "").strip()
            self.assertTrue(reason, msg="%s states no exclusion reason" % entry.get("id"))
            blocking = entry.get("verification", {}).get("blocking_reason")
            self.assertTrue(blocking, msg="%s states no blocking reason" % entry.get("id"))


if __name__ == "__main__":
    unittest.main()


class TestQuarantineAssertsNoUnverifiedIdentifiers(unittest.TestCase):
    """Quarantined drafts must not assert the thing that could not be checked.

    The reason these items are excluded is that their clause identifiers and
    edition-specific values could not be confirmed against a public primary
    source. An excluded item that still carried a clause identifier would put an
    unverifiable citation in the repository, which is the exact failure the
    quarantine exists to prevent.
    """

    def setUp(self) -> None:
        with open(QUARANTINE_PATH, "r", encoding="utf-8") as handle:
            self.payload = json.load(handle)

    def test_no_quarantined_item_asserts_a_clause(self) -> None:
        for item in self.payload["items"]:
            source = item.get("source") or {}
            self.assertFalse(
                source.get("clause"),
                "%s asserts a clause identifier that was never verified" % item["id"],
            )

    def test_quarantined_items_cannot_be_mistaken_for_scorable_items(self) -> None:
        # They store text under 'draft_question', so the loader cannot pick
        # them up even if this file were pointed at the items directory.
        for item in self.payload["items"]:
            self.assertNotIn("question", item, "%s looks scorable" % item["id"])
            self.assertIn("draft_question", item)
            self.assertTrue(item.get("why_excluded"), "%s lacks a reason" % item["id"])
