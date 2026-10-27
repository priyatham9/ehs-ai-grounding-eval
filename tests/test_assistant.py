"""Guards the grounded assistant: committed results match regeneration; core behaviours."""

from __future__ import annotations

import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_assistant_page as bp  # noqa: E402
import run_assistant_eval as rae  # noqa: E402
from grounding_eval.assistant import ABSTAIN_THRESHOLD, Assistant  # noqa: E402
from grounding_eval.corpus import load_corpus  # noqa: E402


def _read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts)) as fh:
        return fh.read()


class TestAssistant(unittest.TestCase):
    def test_committed_results_match_regeneration(self) -> None:
        res = rae.evaluate()
        self.assertEqual(_read("results", "assistant_eval.md"), rae.render_md(res), "run scripts/run_assistant_eval.py")
        self.assertEqual(_read("results", "assistant_eval.csv"), rae.render_csv(res), "run scripts/run_assistant_eval.py")
        self.assertEqual(_read("results", "assistant_audit.jsonl"), rae.render_audit(res), "run scripts/run_assistant_eval.py")

    def test_page_embeds_current_corpus(self) -> None:
        html = _read("docs", "assistant.html")
        self.assertEqual(bp.render(html), html, "run scripts/build_assistant_page.py")
        self.assertNotIn("http://", html.replace("http://www.w3.org", ""))
        self.assertNotIn("https://", html)

    def test_abstains_off_topic_and_cites_on_answer(self) -> None:
        a = Assistant()
        off = a.ask("What is the best pizza topping?")
        self.assertTrue(off.abstained)
        self.assertEqual(off.citations, [])
        item = load_corpus().by_id("cs-001")
        on = a.ask(item.question)
        self.assertEqual(on.audit["retrieved"][0]["id"], "cs-001")
        if not on.abstained:
            self.assertEqual(on.citations, [item.source.clause])
            self.assertIn(item.source.anchor_text, on.text)

    def test_answer_is_extractive_and_deterministic(self) -> None:
        a = Assistant()
        q = load_corpus().by_id("cs-001").question
        r1, r2 = a.ask(q), a.ask(q)
        self.assertEqual(json.dumps(r1.audit, sort_keys=True), json.dumps(r2.audit, sort_keys=True))
        self.assertFalse(r1.audit["llm_used"])
        self.assertEqual(r1.audit["threshold"], ABSTAIN_THRESHOLD)
        self.assertEqual(len(r1.audit["retrieved"]), 3)

    def test_threshold_one_always_abstains_zero_never(self) -> None:
        q = load_corpus().by_id("cs-001").question
        self.assertTrue(Assistant(threshold=1.01).ask(q).abstained)
        self.assertFalse(Assistant(threshold=0.0).ask("What is the best pizza topping?").abstained)


if __name__ == "__main__":
    unittest.main()
