"""Finite-space bounds, threshold realizability, and scorer-validity checks."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from grounding_eval.answerability import (
    ROOT, analyse, analyse_space, diagnose, excerpt_text, load_baseline,
    pair_counts, render_markdown, threshold_curve,
)
from grounding_eval.corpus import Corpus, load_corpus


class TestAnswerability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = load_corpus()
        cls.result = analyse()

    def test_committed_artifact_is_reproducible_and_traceable(self):
        committed = json.loads((ROOT / "results/answerability.json").read_text())
        self.assertEqual(committed, self.result)
        self.assertEqual((ROOT / "results/answerability.json").read_bytes(),
                         (ROOT / "docs/evidence-audit-data.json").read_bytes())
        self.assertEqual((ROOT / "results/answerability.md").read_text(), render_markdown(self.result))
        self.assertIn("grounding_eval/scoring/score.py", committed["input_sha256"])
        self.assertIn("corpus/items/confined_space.json", committed["input_sha256"])

    def test_finite_space_bound_contains_every_actual_baseline_pass(self):
        space = self.result["answer_space"]
        possible = {row["item_id"]: row["passing_excerpt_ids"] for row in space["items"]}
        self.assertEqual(space["n_combinations_per_format"], 63 * 68)
        self.assertEqual(space["n_attainable_wrapped"], 27)
        self.assertEqual(space["n_attainable_bare"], 24)
        for baseline in self.result["baselines"]:
            self.assertEqual(baseline["n_pass"] + baseline["n_recoverable_by_reranking"], 27)
            self.assertEqual(baseline["n_no_excerpt_passes"], 36)
            for row in baseline["items"]:
                self.assertEqual(row["passes"], row["matched_item_id"] in possible[row["item_id"]])

    def test_exhaustion_is_not_just_own_source_scoring(self):
        # A tiny synthetic test fixture: the target's own excerpt lacks its
        # required number, while a different indexed excerpt contains it.
        base = self.corpus.by_id("hc-005")
        target = replace(base, id="test-a", source=replace(base.source, anchor_text="No count given."))
        alternate = replace(base, id="test-b", source=replace(base.source, anchor_text="Eight pictograms."))
        result = analyse_space(Corpus((target, alternate), ()))
        row = next(row for row in result["items"] if row["item_id"] == "test-a")
        self.assertFalse(row["own_wrapped"]["passes"])
        self.assertEqual(row["passing_excerpt_ids"], ["test-b"])

    def test_thresholds_do_not_break_ties_or_reward_fractured_pairs(self):
        pairs = [(SimpleNamespace(id="a"), SimpleNamespace(id="b"))]
        rows = [
            {"item_id": "a", "score": .9, "passes": True},
            {"item_id": "b", "score": .5, "passes": False},
            {"item_id": "c", "score": .5, "passes": True},
        ]
        curve = threshold_curve(rows, pairs)
        self.assertEqual([point["n_answered"] for point in curve], [1, 3])
        self.assertEqual(curve[0]["complete_pairs"], 0)
        self.assertEqual(curve[0]["fractured_pairs"], 1)
        self.assertEqual(curve[1]["complete_pairs"], 1)
        self.assertEqual(curve[1]["both_pass_pairs"], 0)
        # A monotone rescaling preserves each answered set and all its counts.
        squared = threshold_curve([{**row, "score": row["score"] ** 2} for row in rows], pairs)
        for original, rescaled in zip(curve, squared):
            self.assertEqual({k: v for k, v in original.items() if k != "threshold"},
                             {k: v for k, v in rescaled.items() if k != "threshold"})

    def test_pairs_partition_joint_outcomes(self):
        pairs = [(SimpleNamespace(id=a), SimpleNamespace(id=b)) for a, b in (("a", "b"), ("c", "d"), ("e", "f"))]
        self.assertEqual(pair_counts(pairs, dict(a=True, b=True, c=True, d=False, e=False, f=False)),
                         dict(n_pairs=3, both_pass=1, one_pass=1, neither_pass=1))

    def test_metadata_and_numeric_counterexamples_are_reproducible(self):
        oxygen = self.corpus.by_id("cs-004")
        self.assertFalse(diagnose(oxygen, excerpt_text(oxygen, wrapped=False))["passes"])
        self.assertTrue(diagnose(oxygen, excerpt_text(oxygen))["passes"])
        pictograms = self.corpus.by_id("hc-005")
        for source_id in ("hn-001", "rk-001"):
            wrong_topic = self.corpus.by_id(source_id)
            self.assertNotEqual(pictograms.domain, wrong_topic.domain)
            self.assertTrue(diagnose(pictograms, excerpt_text(wrong_topic))["passes"])
        self.assertTrue(all(example["diagnostics"]["passes"] for example in self.result["score_validity_examples"]))

    def test_stale_anchor_rejected_even_when_legacy_digest_omits_anchor(self):
        changed = replace(self.corpus.items[0], source=replace(self.corpus.items[0].source, anchor_text="Altered source."))
        corpus = replace(self.corpus, items=(changed,) + self.corpus.items[1:])
        with self.assertRaisesRegex(ValueError, "stored excerpt no longer matches"):
            load_baseline("retrieval_tfidf", corpus, ROOT / "results")

    def test_duplicate_and_mock_runs_are_rejected(self):
        original = json.loads((ROOT / "results/baselines_retrieval_tfidf.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baselines_retrieval_tfidf.json"
            original["runs"][0]["is_mock_demonstration"] = True
            path.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, "mock"):
                load_baseline("retrieval_tfidf", self.corpus, Path(directory))
            original["runs"][0]["is_mock_demonstration"] = False
            original["runs"][0]["scores"].append(original["runs"][0]["scores"][0])
            path.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, "duplicated"):
                load_baseline("retrieval_tfidf", self.corpus, Path(directory))


if __name__ == "__main__":
    unittest.main()
