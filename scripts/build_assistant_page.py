#!/usr/bin/env python3
"""Re-embed the governed corpus into docs/assistant.html (between the CORPUS markers).

Usage: python3 scripts/build_assistant_page.py [--check]
"""

from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from grounding_eval.adapters.retrieval import corpus_documents  # noqa: E402
from grounding_eval.assistant import ABSTAIN_THRESHOLD  # noqa: E402
from grounding_eval.corpus import load_corpus  # noqa: E402
from grounding_eval.harness import corpus_digest  # noqa: E402

PAGE = os.path.join(ROOT, "docs", "assistant.html")
PATTERN = re.compile(r"(/\*CORPUS_BEGIN\*/).*?(/\*CORPUS_END\*/)", re.S)
EXAMPLE_IDS = ["cs-001", "loto-001", "prd-001"]


def payload() -> str:
    corpus = load_corpus()
    docs = dict(corpus_documents(corpus))
    data = {
        "digest": corpus_digest(corpus),
        "threshold": ABSTAIN_THRESHOLD,
        "passages": [
            {"id": i.id, "standard": i.source.standard, "cite": i.source.clause,
             "title": i.source.section_title, "text": i.source.anchor_text, "doc": docs[i.id]}
            for i in corpus
        ],
        "examples": [corpus.by_id(x).question for x in EXAMPLE_IDS],
    }
    return "const CORPUS = " + json.dumps(data, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/") + ";"


def render(html: str) -> str:
    return PATTERN.sub(lambda m: m.group(1) + "\n" + payload() + "\n" + m.group(2), html)


def main() -> int:
    html = open(PAGE).read()
    new = render(html)
    if "--check" in sys.argv:
        return 0 if new == html else 1
    open(PAGE, "w").write(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
