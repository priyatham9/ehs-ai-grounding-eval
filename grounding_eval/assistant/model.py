"""Answer composers.

``AnswerModel`` is the seam where a real LLM would plug in (see docs/assistant.md).
``MockModel`` is NOT a language model: it is an extractive, deterministic composer
that quotes the top retrieved passage verbatim and attaches its citation. It needs
no keys and cannot say anything the corpus does not say.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Passage:
    """One retrieved, governed passage. ``cite`` is the CFR/standard clause."""

    item_id: str
    standard: str
    cite: str
    title: str
    text: str
    score: float


class AnswerModel(Protocol):
    name: str

    def compose(self, query: str, passages: Sequence[Passage]) -> str:
        """Return answer text built only from ``passages``; must cite ``cite`` of each used."""


class MockModel:
    """Extractive composer: quote the top passage, cite it. Deterministic, no network."""

    name = "mock-extractive"

    def compose(self, query: str, passages: Sequence[Passage]) -> str:
        top = passages[0]
        return '%s (%s, %s) states: "%s" [1] Source: %s, %s.' % (
            top.standard, top.cite, top.title, top.text, top.standard, top.cite,
        )
