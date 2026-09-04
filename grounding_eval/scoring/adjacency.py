"""Adjacent-substitution detection.

The failure this benchmark exists to measure is a system answering about the
semantically adjacent wrong entity: pressure relief valve parameters returned for
a rupture disk question, entry supervisor duties returned for an attendant
question, the process safety management audit interval returned for the lockout
periodic inspection interval.

Detection works on ``adjacent_wrong.signature_terms``: vocabulary that belongs to
the wrong entity and would not normally appear in a correct answer. A term counts
only when asserted, not when contrasted, so an answer that explicitly says the
wrong entity's parameter does not apply is not penalised for naming it.

This is a heuristic. It is reported alongside the evidence (which terms fired) so
that a human adjudicator can check it, and docs/methodology.md states that the
human-adjudicated subsample, not this function, is the primary measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..schema import AdjacentWrong
from ..text import find_asserted, find_phrase, normalize


@dataclass(frozen=True)
class AdjacencyAssessment:
    """Evidence that a response answered about the adjacent wrong entity."""

    asserted_terms: Tuple[str, ...]
    contrasted_terms: Tuple[str, ...]

    @property
    def n_asserted(self) -> int:
        return len(self.asserted_terms)


def assess_adjacency(response_text: str, adjacent: AdjacentWrong) -> AdjacencyAssessment:
    """Find which of the adjacent entity's signature terms the response asserts."""
    text = normalize(response_text)
    asserted = []
    contrasted = []
    for term in adjacent.signature_terms:
        spans = find_phrase(text, term)
        if not spans:
            continue
        if find_asserted(text, term):
            asserted.append(term)
        else:
            contrasted.append(term)
    return AdjacencyAssessment(
        asserted_terms=tuple(sorted(set(asserted))),
        contrasted_terms=tuple(sorted(set(contrasted))),
    )
