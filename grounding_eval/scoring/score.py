"""Composition of the scoring components into a single outcome per response.

The decision procedure below is the whole of the automatic scoring rule. It is
short on purpose: anyone disputing a score should be able to read this function
and the item's answer key and reconstruct the result by hand.

Thresholds are configuration, not findings. The defaults were chosen before any
system was run against the corpus, they are recorded in docs/preregistration.md,
and every reported number should name the configuration that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from ..corpus import Corpus
from ..schema import (
    AdapterResponse,
    CitationGrade,
    Item,
    ItemScore,
    ItemType,
    Outcome,
    RiskTier,
)
from .abstention import detect_abstention
from .adjacency import assess_adjacency
from .citations import assess_citations
from .concepts import assess_concepts

# Utility weights are a stated policy, not an estimated quantity. They encode a
# judgement about the relative cost of outcomes in a safety setting: a confident
# wrong answer about the adjacent device is worse than a plainly wrong answer,
# and both are worse than declining. Report the policy name with any number
# derived from it, and do not present a utility as a measurement.
POLICIES: Mapping[str, Mapping[str, float]] = {
    "safety": {
        "correct": 1.0,
        "abstained": 0.25,
        "abstained_acceptable": 0.9,
        "other_incorrect": -0.5,
        "adjacent_substitution": -1.0,
        "unscorable": 0.0,
    },
    "accuracy": {
        "correct": 1.0,
        "abstained": 0.0,
        "abstained_acceptable": 0.0,
        "other_incorrect": 0.0,
        "adjacent_substitution": 0.0,
        "unscorable": 0.0,
    },
}

# Risk weights are likewise a stated policy used only for an optional weighted
# summary. Headline outcome counts are never risk-weighted.
DEFAULT_RISK_WEIGHTS: Mapping[str, float] = {
    RiskTier.HIGH.value: 1.0,
    RiskTier.MEDIUM.value: 0.6,
    RiskTier.LOW.value: 0.3,
}


@dataclass(frozen=True)
class ScoringConfig:
    """Thresholds and policy weights for one scoring run.

    coverage_threshold
        Fraction of required concept groups that must be satisfied for an answer
        to be eligible for CORRECT.
    require_full_numeric
        When True, every numeric fact in the answer key must be present. A wrong
        number in a safety answer is not a partial credit situation.
    adjacent_min_terms
        Number of asserted adjacent-entity signature terms that on their own
        classify a response as an adjacent substitution.
    adjacent_low_coverage
        Below this concept coverage, a single asserted signature term is enough.
    policy
        Key into POLICIES for the utility table.
    """

    coverage_threshold: float = 0.75
    require_full_numeric: bool = True
    adjacent_min_terms: int = 2
    adjacent_low_coverage: float = 0.5
    policy: str = "safety"
    risk_weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_RISK_WEIGHTS))

    def utility_table(self) -> Mapping[str, float]:
        try:
            return POLICIES[self.policy]
        except KeyError as exc:
            raise ValueError(
                "unknown scoring policy %r; available: %s" % (self.policy, ", ".join(sorted(POLICIES)))
            ) from exc

    def describe(self) -> Dict[str, object]:
        return {
            "coverage_threshold": self.coverage_threshold,
            "require_full_numeric": self.require_full_numeric,
            "adjacent_min_terms": self.adjacent_min_terms,
            "adjacent_low_coverage": self.adjacent_low_coverage,
            "policy": self.policy,
            "policy_weights": dict(self.utility_table()),
            "risk_weights": dict(self.risk_weights),
        }


def score_item(item: Item, response: AdapterResponse, config: Optional[ScoringConfig] = None) -> ItemScore:
    """Score one response against one item.

    The order of the decision is fixed and is as follows.

    1. An adapter error or an empty response is UNSCORABLE. It is never counted
       as wrong, because an infrastructure failure and a confabulation are
       different events and mixing them corrupts both rates.
    2. An explicit declining response with no substantive commitment is
       ABSTAINED.
    3. Otherwise the answer key decides CORRECT: concept coverage at or above the
       threshold, no forbidden concept asserted, and, when the key declares
       numeric facts, all of them present.
    4. A response that is not correct and that asserts the adjacent entity's
       signature vocabulary is ADJACENT_SUBSTITUTION.
    5. Everything else is OTHER_INCORRECT.
    """
    cfg = config or ScoringConfig()

    base = dict(
        item_id=item.id,
        domain=item.domain,
        family=item.family,
        item_type=item.item_type,
        risk_tier=item.risk_tier,
        pair_id=item.pair_id,
        confidence=response.confidence,
    )

    if response.error:
        return ItemScore(
            outcome=Outcome.UNSCORABLE,
            concept_coverage=0.0,
            forbidden_asserted=(),
            numeric_coverage=None,
            adjacent_hits=(),
            abstained=False,
            citation_grade=CitationGrade.NONE,
            cited_cross_reference=False,
            unsupported_paywalled_citations=(),
            unscorable_reason="adapter_error: %s" % response.error,
            **base
        )

    abstention = detect_abstention(response.text)
    if abstention.empty:
        return ItemScore(
            outcome=Outcome.UNSCORABLE,
            concept_coverage=0.0,
            forbidden_asserted=(),
            numeric_coverage=None,
            adjacent_hits=(),
            abstained=False,
            citation_grade=CitationGrade.NONE,
            cited_cross_reference=False,
            unsupported_paywalled_citations=(),
            unscorable_reason="empty_response",
            **base
        )

    concepts = assess_concepts(response.text, item.answer_key)
    adjacency = assess_adjacency(response.text, item.adjacent_wrong)
    citations = assess_citations(item, response.text, response.citations)

    if abstention.abstained:
        outcome = Outcome.ABSTAINED
    else:
        numeric_ok = True
        if concepts.numeric_coverage is not None and cfg.require_full_numeric:
            numeric_ok = concepts.numeric_coverage >= 1.0
        correct = (
            concepts.coverage >= cfg.coverage_threshold
            and not concepts.has_forbidden
            and numeric_ok
        )
        if correct:
            outcome = Outcome.CORRECT
        elif adjacency.n_asserted >= cfg.adjacent_min_terms or (
            adjacency.n_asserted >= 1 and concepts.coverage < cfg.adjacent_low_coverage
        ):
            outcome = Outcome.ADJACENT_SUBSTITUTION
        else:
            outcome = Outcome.OTHER_INCORRECT

    return ItemScore(
        outcome=outcome,
        concept_coverage=concepts.coverage,
        forbidden_asserted=concepts.forbidden_asserted,
        numeric_coverage=concepts.numeric_coverage,
        adjacent_hits=adjacency.asserted_terms,
        abstained=abstention.abstained,
        citation_grade=citations.grade,
        cited_cross_reference=citations.cited_cross_reference,
        unsupported_paywalled_citations=citations.unsupported_paywalled,
        unscorable_reason=None,
        **base
    )


def utility_of(score: ItemScore, item: Item, config: Optional[ScoringConfig] = None) -> float:
    """Policy-weighted utility for one scored response.

    Abstention on an item whose correct behaviour is to decline (a category-error
    item) earns nearly full credit; abstention on an answerable item earns partial
    credit. Both weights come from the named policy and are not estimates.
    """
    cfg = config or ScoringConfig()
    table = cfg.utility_table()
    if score.outcome is Outcome.ABSTAINED and item.abstention_acceptable:
        return table["abstained_acceptable"]
    return table[score.outcome.value]


def score_responses(
    corpus: Corpus,
    responses: Iterable[AdapterResponse],
    config: Optional[ScoringConfig] = None,
) -> List[ItemScore]:
    """Score a batch of responses. Responses for unknown item ids are an error."""
    cfg = config or ScoringConfig()
    by_id = {item.id: item for item in corpus}
    out: List[ItemScore] = []
    for response in responses:
        item = by_id.get(response.item_id)
        if item is None:
            raise KeyError(
                "response references unknown item id %r; the responses file and the "
                "corpus are out of step" % response.item_id
            )
        out.append(score_item(item, response, cfg))
    return out
