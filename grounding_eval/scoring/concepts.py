"""Concept coverage and forbidden-concept detection.

Coverage is the fraction of required alternation groups satisfied. A group is
satisfied when any one of its alternatives appears anywhere in the response,
including inside a contrast construction: stating "the interval is annual, not
triennial" satisfies the group ``["annual"]``.

Forbidden concepts are handled asymmetrically. A forbidden phrase counts against
the response only when it is asserted, that is, when it appears outside a
contrast construction. "A rupture disk has no blowdown" does not assert blowdown;
"set the blowdown to 7 percent" does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..schema import AnswerKey
from ..text import any_asserted, any_present, has_number, normalize


@dataclass(frozen=True)
class ConceptAssessment:
    """Result of comparing a response against an item's answer key."""

    coverage: float
    satisfied_groups: Tuple[int, ...]
    missing_groups: Tuple[Tuple[str, ...], ...]
    forbidden_asserted: Tuple[str, ...]
    numeric_coverage: Optional[float]
    missing_numeric: Tuple[str, ...]

    @property
    def has_forbidden(self) -> bool:
        return bool(self.forbidden_asserted)


def assess_concepts(response_text: str, key: AnswerKey) -> ConceptAssessment:
    """Score one response against one answer key.

    ``numeric_coverage`` is ``None`` when the key declares no numeric facts, so
    that "no numeric requirement" and "numeric requirement fully met" stay
    distinguishable downstream.
    """
    text = normalize(response_text)

    satisfied: List[int] = []
    missing: List[Tuple[str, ...]] = []
    for index, group in enumerate(key.required_concepts):
        if any_present(text, group):
            satisfied.append(index)
        else:
            missing.append(group)
    total = len(key.required_concepts)
    coverage = (len(satisfied) / total) if total else 1.0

    forbidden: List[str] = []
    for group in key.forbidden_concepts:
        forbidden.extend(any_asserted(text, group))

    numeric_coverage: Optional[float] = None
    missing_numeric: List[str] = []
    if key.numeric_facts:
        hits = 0
        for fact in key.numeric_facts:
            if has_number(text, fact.value, fact.tolerance_abs):
                hits += 1
            else:
                missing_numeric.append("%s=%s %s" % (fact.label, fact.value, fact.unit))
        numeric_coverage = hits / len(key.numeric_facts)

    return ConceptAssessment(
        coverage=coverage,
        satisfied_groups=tuple(satisfied),
        missing_groups=tuple(missing),
        forbidden_asserted=tuple(sorted(set(forbidden))),
        numeric_coverage=numeric_coverage,
        missing_numeric=tuple(missing_numeric),
    )


def coverage_of(response_text: str, groups: Sequence[Sequence[str]]) -> float:
    """Standalone coverage helper, used by tests and by the adjudication export."""
    if not groups:
        return 1.0
    text = normalize(response_text)
    hits = sum(1 for group in groups if any_present(text, group))
    return hits / len(groups)
