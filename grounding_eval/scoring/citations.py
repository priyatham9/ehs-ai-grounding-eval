"""Citation presence and correctness.

Three things are measured separately, because they fail separately:

1. Presence. Did the response cite anything at all?
2. Target correctness. Did it name the right document and section?
3. Clause correctness. Did it name the right paragraph within that section?

A fourth quantity is tracked because of what it costs a reader to check:
``unsupported_paywalled_citations`` counts references to copyrighted consensus
standards (ASME, API, NFPA, ISO) that the item does not list as its source or as
a corroborating cross-reference. A fabricated CFR citation can be checked for free
in thirty seconds. A fabricated ASME paragraph number cannot, which makes it a
more expensive error to catch and therefore worth counting on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ..schema import CitationGrade, CitationRef, Item, is_paywalled_family, parse_citations


@dataclass(frozen=True)
class CitationAssessment:
    grade: CitationGrade
    found: Tuple[str, ...]
    matched_clause: Tuple[str, ...]
    matched_section: Tuple[str, ...]
    cited_cross_reference: bool
    unsupported_paywalled: Tuple[str, ...]

    @property
    def has_any(self) -> bool:
        return bool(self.found)


def _collect(response_text: str, explicit_citations: Sequence[str]) -> Tuple[CitationRef, ...]:
    """Parse citations from the answer text and from any structured citation list."""
    refs: List[CitationRef] = list(parse_citations(response_text))
    seen = {(r.family, r.section, r.chain) for r in refs}
    for raw in explicit_citations or ():
        for ref in parse_citations(str(raw)):
            key = (ref.family, ref.section, ref.chain)
            if key not in seen:
                seen.add(key)
                refs.append(ref)
    return tuple(refs)


def assess_citations(
    item: Item,
    response_text: str,
    explicit_citations: Sequence[str] = (),
) -> CitationAssessment:
    """Grade the citations in one response against one item's authoritative source."""
    found = _collect(response_text, explicit_citations)
    expected = item.expected_citations
    acceptable = item.acceptable_citations

    matched_clause: List[str] = []
    matched_section: List[str] = []
    for ref in found:
        for want in expected:
            if ref.same_clause(want):
                matched_clause.append(ref.as_text())
                break
            if ref.same_target(want):
                matched_section.append(ref.as_text())
                break

    cited_cross_reference = any(
        any(ref.same_target(x) for x in acceptable if x not in expected) for ref in found
    )

    unsupported: List[str] = []
    for ref in found:
        if not is_paywalled_family(ref.family):
            continue
        if any(ref.same_target(ok) for ok in acceptable):
            continue
        unsupported.append(ref.as_text())

    if not found:
        grade = CitationGrade.NONE
    elif matched_clause:
        grade = CitationGrade.CLAUSE
    elif matched_section:
        grade = CitationGrade.SECTION
    else:
        grade = CitationGrade.WRONG

    return CitationAssessment(
        grade=grade,
        found=tuple(ref.as_text() for ref in found),
        matched_clause=tuple(sorted(set(matched_clause))),
        matched_section=tuple(sorted(set(matched_section))),
        cited_cross_reference=cited_cross_reference,
        unsupported_paywalled=tuple(sorted(set(unsupported))),
    )


def grade_rank(grade: CitationGrade) -> int:
    """Ordinal rank for aggregation. Higher is better."""
    return {
        CitationGrade.NONE: 0,
        CitationGrade.WRONG: 1,
        CitationGrade.SECTION: 2,
        CitationGrade.CLAUSE: 3,
    }[grade]


def grades_in_order() -> Tuple[CitationGrade, ...]:
    return (CitationGrade.NONE, CitationGrade.WRONG, CitationGrade.SECTION, CitationGrade.CLAUSE)
