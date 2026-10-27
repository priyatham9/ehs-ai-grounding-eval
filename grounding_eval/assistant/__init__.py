"""Grounded assistant: citation-first Q&A over the governed corpus.

Retrieval (BM25+ or Okapi BM25, the same indexes the baselines use), an
extractive answer composer, a documented abstain rule and one audit record per
answer. No language model is involved; see docs/assistant.md.
"""

from .assistant import (
    ABSTAIN_THRESHOLD,
    Assistant,
    AssistantAdapter,
    AssistantAnswer,
    coverage_of,
)
from .model import AnswerModel, MockModel, Passage

__all__ = [
    "ABSTAIN_THRESHOLD", "Assistant", "AssistantAdapter", "AssistantAnswer",
    "AnswerModel", "MockModel", "Passage", "coverage_of",
]
