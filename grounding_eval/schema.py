"""Typed representations of corpus items, adapter responses and scores.

Everything in this module is a plain dataclass or an Enum. There is no behaviour
here beyond parsing and validation, so that the data model can be read on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "1.0"


class ItemType(str, Enum):
    """What kind of question an item is.

    FACTUAL
        There is a correct answer stated in an authoritative source, and the
        system is expected to produce it.
    CATEGORY_ERROR
        The question contains a false or unanswerable premise. The expected
        behaviour is to reject the premise. Producing a confident answer is the
        worst outcome, not a partially correct one. These items are reported
        separately and are never folded into overall accuracy.
    """

    FACTUAL = "factual"
    CATEGORY_ERROR = "category_error"


class RiskTier(str, Enum):
    """Consequence class of getting the item wrong.

    These are an author judgement, not a measured quantity. They are used only
    for an explicitly optional risk-weighted summary and never for the headline
    outcome counts.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"


class Outcome(str, Enum):
    """The scored outcome for one response to one item.

    CORRECT
        Answer key satisfied: required concept coverage at or above threshold,
        no forbidden concept asserted, and every numeric fact present.
    ADJACENT_SUBSTITUTION
        The response answered about the semantically adjacent wrong entity. This
        is the failure class the benchmark exists to measure, and it is reported
        separately from other errors because its consequences differ.
    OTHER_INCORRECT
        Wrong, but not by substituting the adjacent entity.
    ABSTAINED
        The system declined to answer or expressed that it did not know.
    UNSCORABLE
        Empty response, or the adapter reported an error. Never silently counted
        as either correct or incorrect.
    """

    CORRECT = "correct"
    ADJACENT_SUBSTITUTION = "adjacent_substitution"
    OTHER_INCORRECT = "other_incorrect"
    ABSTAINED = "abstained"
    UNSCORABLE = "unscorable"


class CitationGrade(str, Enum):
    """How well the response's citations match the item's authoritative source."""

    NONE = "none"
    WRONG = "wrong"
    SECTION = "correct_section"
    CLAUSE = "correct_clause"


@dataclass(frozen=True)
class CitationRef:
    """A parsed reference to a clause in a standard.

    ``family`` is the document family, normalised (for example ``29 cfr``,
    ``asme ug``, ``nfpa 70e``). ``section`` is the section or paragraph
    identifier within it (``1910.147``, ``127``). ``chain`` is the ordered list
    of subordinate paragraph designators (``("c", "6", "i")``).
    """

    family: str
    section: str
    chain: Tuple[str, ...] = ()
    paywalled: bool = False

    def as_text(self) -> str:
        tail = "".join("(%s)" % part for part in self.chain)
        return "%s %s%s" % (self.family.upper(), self.section, tail)

    def same_target(self, other: "CitationRef") -> bool:
        return self.family == other.family and self.section == other.section

    def same_clause(self, other: "CitationRef") -> bool:
        return self.same_target(other) and self.chain == other.chain


@dataclass(frozen=True)
class NumericFact:
    label: str
    value: float
    unit: str
    tolerance_abs: float = 0.0


@dataclass(frozen=True)
class AnswerKey:
    """Concept-coverage answer key.

    ``required_concepts`` is a list of alternation groups. A group is satisfied
    when any one of its alternatives appears in the response. Coverage is the
    fraction of groups satisfied.

    ``forbidden_concepts`` is a list of alternation groups whose assertion makes
    the answer wrong. A forbidden phrase appearing inside a contrast construction
    ("unlike a relief valve, a rupture disk has no blowdown") is not counted as
    an assertion; see grounding_eval.scoring.concepts.
    """

    required_concepts: Tuple[Tuple[str, ...], ...] = ()
    forbidden_concepts: Tuple[Tuple[str, ...], ...] = ()
    numeric_facts: Tuple[NumericFact, ...] = ()


@dataclass(frozen=True)
class AdjacentWrong:
    """Description of the plausible-but-wrong adjacent answer for an item."""

    label: str
    answer: str
    signature_terms: Tuple[str, ...]
    why_dangerous: str


@dataclass(frozen=True)
class SourceRef:
    standard: str
    clause: str
    section_title: str
    url: str
    access: str
    anchor_text: str
    value_from: Optional[str] = None


@dataclass(frozen=True)
class CrossReference:
    standard: str
    clause: str
    access: str
    corroboration: str


@dataclass(frozen=True)
class Verification:
    status: VerificationStatus
    method: str
    source_edition_date: str


@dataclass(frozen=True)
class Item:
    """One benchmark question with its answer key and provenance."""

    id: str
    domain: str
    item_type: ItemType
    family: str
    risk_tier: RiskTier
    question: str
    correct_answer: str
    answer_key: AnswerKey
    adjacent_wrong: AdjacentWrong
    source: SourceRef
    cross_reference: Tuple[CrossReference, ...]
    verification: Verification
    abstention_acceptable: bool
    pair_id: Optional[str] = None
    pair_role: Optional[str] = None
    notes: Optional[str] = None

    @property
    def expected_citations(self) -> Tuple[CitationRef, ...]:
        return parse_citations(self.source.clause)

    @property
    def acceptable_citations(self) -> Tuple[CitationRef, ...]:
        refs = list(self.expected_citations)
        for xref in self.cross_reference:
            refs.extend(parse_citations(xref.clause))
        return tuple(refs)


@dataclass
class AdapterResponse:
    """What a system under test returned for one item.

    ``confidence`` is optional. When absent, selective-prediction analysis is
    skipped rather than imputed. ``error`` set to a non-empty string makes the
    response unscorable; it is never treated as a wrong answer, because an
    infrastructure failure and a confabulation are different events.
    """

    item_id: str
    text: str
    citations: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    latency_s: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: str = "unspecified"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "text": self.text,
            "citations": list(self.citations),
            "confidence": self.confidence,
            "latency_s": self.latency_s,
            "error": self.error,
            "metadata": dict(self.metadata),
            "provenance": self.provenance,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "AdapterResponse":
        return AdapterResponse(
            item_id=str(payload["item_id"]),
            text=str(payload.get("text", "")),
            citations=list(payload.get("citations") or []),
            confidence=payload.get("confidence"),
            latency_s=payload.get("latency_s"),
            error=payload.get("error"),
            metadata=dict(payload.get("metadata") or {}),
            provenance=str(payload.get("provenance", "unspecified")),
        )


@dataclass
class ItemScore:
    """The scored result for one (item, response) pair."""

    item_id: str
    domain: str
    family: str
    item_type: ItemType
    risk_tier: RiskTier
    pair_id: Optional[str]
    outcome: Outcome
    concept_coverage: float
    forbidden_asserted: Tuple[str, ...]
    numeric_coverage: Optional[float]
    adjacent_hits: Tuple[str, ...]
    abstained: bool
    citation_grade: CitationGrade
    cited_cross_reference: bool
    unsupported_paywalled_citations: Tuple[str, ...]
    confidence: Optional[float]
    unscorable_reason: Optional[str] = None

    @property
    def is_correct(self) -> bool:
        return self.outcome is Outcome.CORRECT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "domain": self.domain,
            "family": self.family,
            "item_type": self.item_type.value,
            "risk_tier": self.risk_tier.value,
            "pair_id": self.pair_id,
            "outcome": self.outcome.value,
            "concept_coverage": self.concept_coverage,
            "forbidden_asserted": list(self.forbidden_asserted),
            "numeric_coverage": self.numeric_coverage,
            "adjacent_hits": list(self.adjacent_hits),
            "abstained": self.abstained,
            "citation_grade": self.citation_grade.value,
            "cited_cross_reference": self.cited_cross_reference,
            "unsupported_paywalled_citations": list(self.unsupported_paywalled_citations),
            "confidence": self.confidence,
            "unscorable_reason": self.unscorable_reason,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "ItemScore":
        """Rebuild a score from its serialised form.

        Used when re-reporting stored run files. Round-tripping is exercised in
        the test suite so that a report regenerated from disk is identical to the
        one printed when the run happened.
        """
        return ItemScore(
            item_id=payload["item_id"],
            domain=payload["domain"],
            family=payload["family"],
            item_type=ItemType(payload["item_type"]),
            risk_tier=RiskTier(payload["risk_tier"]),
            pair_id=payload.get("pair_id"),
            outcome=Outcome(payload["outcome"]),
            concept_coverage=float(payload["concept_coverage"]),
            forbidden_asserted=tuple(payload.get("forbidden_asserted") or ()),
            numeric_coverage=payload.get("numeric_coverage"),
            adjacent_hits=tuple(payload.get("adjacent_hits") or ()),
            abstained=bool(payload["abstained"]),
            citation_grade=CitationGrade(payload["citation_grade"]),
            cited_cross_reference=bool(payload["cited_cross_reference"]),
            unsupported_paywalled_citations=tuple(
                payload.get("unsupported_paywalled_citations") or ()
            ),
            confidence=payload.get("confidence"),
            unscorable_reason=payload.get("unscorable_reason"),
        )


# --------------------------------------------------------------------------
# Citation parsing
# --------------------------------------------------------------------------

_PAYWALLED_FAMILIES = ("asme ug", "nfpa", "api", "iso", "ansi", "asme bpvc")

# 29 CFR 1910.147(c)(6)(i)  /  40 CFR 68.79(a)  /  46 CFR 54.15-13(b)(3)
_CFR_RE = re.compile(
    r"(?P<title>\d{1,2})\s*c\.?f\.?r\.?\s*(?:part\s*)?"
    r"(?P<section>\d{2,4}(?:\.\d+)?(?:-\d+)?)"
    r"(?P<chain>(?:\s*\([0-9a-z]{1,4}\))*)",
    re.IGNORECASE,
)
# Bare section reference such as "1910.147(c)(6)" or "§ 1910.147(c)(6)".
_BARE_SECTION_RE = re.compile(
    r"(?<![\d.])(?:§\s*)?(?P<section>19\d{2}\.\d+|68\.\d+|54\.\d+(?:-\d+)?)"
    r"(?P<chain>(?:\s*\([0-9a-z]{1,4}\))*)",
    re.IGNORECASE,
)
_ASME_RE = re.compile(r"\bug-?\s?(?P<num>\d{2,3})\b", re.IGNORECASE)
_NFPA_RE = re.compile(r"\bnfpa\s*(?P<num>\d{1,4}[a-z]?)\b", re.IGNORECASE)
_API_RE = re.compile(r"\bapi\s*(?:rp|std|standard|recommended practice)?\s*(?P<num>\d{2,4})\b", re.IGNORECASE)
_ISO_RE = re.compile(r"\biso\s*(?P<num>\d{3,5}(?:-\d+)?)\b", re.IGNORECASE)

_CHAIN_RE = re.compile(r"\(([0-9a-z]{1,4})\)", re.IGNORECASE)

_CFR_TITLE_NAMES = {"29": "29 cfr", "40": "40 cfr", "46": "46 cfr", "49": "49 cfr", "30": "30 cfr"}

# Sections we know belong to a CFR title, used to resolve bare references such as
# "1910.147(c)(6)" that omit the title. Restricted to the titles this corpus uses.
_BARE_SECTION_TITLE = (
    ("1904.", "29 cfr"),
    ("1910.", "29 cfr"),
    ("1926.", "29 cfr"),
    ("68.", "40 cfr"),
    ("54.", "46 cfr"),
)


def _chain_of(raw: str) -> Tuple[str, ...]:
    return tuple(m.group(1).lower() for m in _CHAIN_RE.finditer(raw or ""))


def parse_citations(text: str) -> Tuple[CitationRef, ...]:
    """Extract every recognised standard reference from free text.

    Recognises CFR references with or without an explicit title, ASME UG
    paragraph identifiers, and NFPA / API / ISO document numbers. Returns
    references in order of appearance with duplicates removed.

    The parser is deliberately conservative. It does not attempt to resolve a
    bare section number outside the small set of prefixes this corpus uses,
    because guessing a CFR title is exactly the kind of plausible inference the
    benchmark is trying to detect in the systems under test.
    """
    found: List[CitationRef] = []
    seen = set()
    if not text:
        return ()

    def add(ref: CitationRef) -> None:
        key = (ref.family, ref.section, ref.chain)
        if key not in seen:
            seen.add(key)
            found.append(ref)

    covered_spans: List[Tuple[int, int]] = []
    for match in _CFR_RE.finditer(text):
        title = match.group("title")
        family = _CFR_TITLE_NAMES.get(title, "%s cfr" % title)
        add(CitationRef(family, match.group("section"), _chain_of(match.group("chain"))))
        covered_spans.append(match.span())

    for match in _BARE_SECTION_RE.finditer(text):
        start, end = match.span()
        if any(s <= start and end <= e for s, e in covered_spans):
            continue
        section = match.group("section")
        family = None
        for prefix, fam in _BARE_SECTION_TITLE:
            if section.startswith(prefix):
                family = fam
                break
        if family is None:
            continue
        add(CitationRef(family, section, _chain_of(match.group("chain"))))

    for match in _ASME_RE.finditer(text):
        add(CitationRef("asme ug", match.group("num"), (), paywalled=True))
    for match in _NFPA_RE.finditer(text):
        add(CitationRef("nfpa", match.group("num").lower(), (), paywalled=True))
    for match in _API_RE.finditer(text):
        add(CitationRef("api", match.group("num"), (), paywalled=True))
    for match in _ISO_RE.finditer(text):
        add(CitationRef("iso", match.group("num"), (), paywalled=True))

    return tuple(found)


def is_paywalled_family(family: str) -> bool:
    return any(family.startswith(prefix) for prefix in _PAYWALLED_FAMILIES)


# --------------------------------------------------------------------------
# Item construction from JSON payloads
# --------------------------------------------------------------------------


class CorpusValidationError(ValueError):
    """Raised when a corpus file does not satisfy the schema."""


def _groups(raw: Optional[Sequence[Sequence[str]]], field_name: str, item_id: str) -> Tuple[Tuple[str, ...], ...]:
    if raw is None:
        return ()
    out: List[Tuple[str, ...]] = []
    for group in raw:
        if isinstance(group, str):
            raise CorpusValidationError(
                "%s: %s must be a list of alternation groups (lists of strings), "
                "not a flat list of strings" % (item_id, field_name)
            )
        alts = tuple(str(alt).strip().lower() for alt in group if str(alt).strip())
        if not alts:
            raise CorpusValidationError("%s: empty alternation group in %s" % (item_id, field_name))
        out.append(alts)
    return tuple(out)


def item_from_dict(payload: Dict[str, Any], domain: str) -> Item:
    """Build an :class:`Item` from a corpus JSON object, validating as we go."""
    try:
        item_id = str(payload["id"])
    except KeyError as exc:  # pragma: no cover - defensive
        raise CorpusValidationError("item is missing 'id'") from exc

    for required in ("question", "correct_answer", "answer_key", "adjacent_wrong", "source", "verification"):
        if required not in payload:
            raise CorpusValidationError("%s: missing required field %r" % (item_id, required))

    key_raw = payload["answer_key"] or {}
    numeric_facts = tuple(
        NumericFact(
            label=str(nf.get("label", "")),
            value=float(nf["value"]),
            unit=str(nf.get("unit", "")),
            tolerance_abs=float(nf.get("tolerance_abs", 0.0)),
        )
        for nf in (key_raw.get("numeric_facts") or [])
    )
    answer_key = AnswerKey(
        required_concepts=_groups(key_raw.get("required_concepts"), "required_concepts", item_id),
        forbidden_concepts=_groups(key_raw.get("forbidden_concepts"), "forbidden_concepts", item_id),
        numeric_facts=numeric_facts,
    )
    if not answer_key.required_concepts:
        raise CorpusValidationError("%s: answer_key.required_concepts must be non-empty" % item_id)

    adj_raw = payload["adjacent_wrong"]
    adjacent = AdjacentWrong(
        label=str(adj_raw["label"]),
        answer=str(adj_raw["answer"]),
        signature_terms=tuple(str(t).strip().lower() for t in adj_raw.get("signature_terms", []) if str(t).strip()),
        why_dangerous=str(adj_raw["why_dangerous"]),
    )

    src_raw = payload["source"]
    source = SourceRef(
        standard=str(src_raw["standard"]),
        clause=str(src_raw["clause"]),
        section_title=str(src_raw.get("section_title", "")),
        url=str(src_raw.get("url", "")),
        access=str(src_raw.get("access", "unknown")),
        anchor_text=str(src_raw["anchor_text"]),
        value_from=src_raw.get("value_from"),
    )
    if not source.anchor_text.strip():
        raise CorpusValidationError("%s: source.anchor_text must be non-empty" % item_id)

    cross = tuple(
        CrossReference(
            standard=str(x["standard"]),
            clause=str(x["clause"]),
            access=str(x.get("access", "unknown")),
            corroboration=str(x.get("corroboration", "")),
        )
        for x in (payload.get("cross_reference") or [])
    )

    ver_raw = payload["verification"]
    verification = Verification(
        status=VerificationStatus(str(ver_raw["status"])),
        method=str(ver_raw.get("method", "")),
        source_edition_date=str(ver_raw.get("source_edition_date", "")),
    )
    if verification.status is not VerificationStatus.VERIFIED:
        raise CorpusValidationError(
            "%s: only verified items may live in corpus/items/. Unverified drafts belong "
            "in corpus/quarantine/unverified.json" % item_id
        )

    item_type = ItemType(str(payload.get("item_type", "factual")))
    if item_type is ItemType.CATEGORY_ERROR and not bool(payload.get("abstention_acceptable", False)):
        raise CorpusValidationError(
            "%s: category_error items must set abstention_acceptable=true" % item_id
        )

    pair_id = payload.get("pair_id")
    pair_role = payload.get("pair_role")
    if (pair_id is None) != (pair_role is None):
        raise CorpusValidationError("%s: pair_id and pair_role must be set together" % item_id)
    if pair_role is not None and pair_role not in ("a", "b"):
        raise CorpusValidationError("%s: pair_role must be 'a' or 'b'" % item_id)

    if not parse_citations(source.clause):
        raise CorpusValidationError(
            "%s: source.clause %r did not parse into any recognised citation" % (item_id, source.clause)
        )

    return Item(
        id=item_id,
        domain=domain,
        item_type=item_type,
        family=str(payload.get("family", domain)),
        risk_tier=RiskTier(str(payload.get("risk_tier", "medium"))),
        question=str(payload["question"]),
        correct_answer=str(payload["correct_answer"]),
        answer_key=answer_key,
        adjacent_wrong=adjacent,
        source=source,
        cross_reference=cross,
        verification=verification,
        abstention_acceptable=bool(payload.get("abstention_acceptable", False)),
        pair_id=str(pair_id) if pair_id is not None else None,
        pair_role=str(pair_role) if pair_role is not None else None,
        notes=payload.get("notes"),
    )
