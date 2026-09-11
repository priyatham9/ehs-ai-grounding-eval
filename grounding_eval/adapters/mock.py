"""Mock adapter: a demonstration fixture, not a system under test.

READ THIS BEFORE INTERPRETING ANY NUMBER PRODUCED WITH THIS ADAPTER.

The mock composes its answers out of the corpus item it is answering. Its
"correct" responses are built from the item's own reference answer, and its
"wrong" responses are built from the item's own adjacent-wrong description. It
therefore scores exactly as well as its profile probabilities say it will, by
construction. It measures nothing about any real system, and the numbers it
produces are not evidence about grounding, retrieval, or hallucination.

It exists for three reasons, all of them mechanical:

1. The harness must be runnable end to end with no API key and no network.
2. The scoring path must be exercised across every outcome branch in tests.
3. A reader evaluating this repository should be able to see the report format
   without being asked to trust an unverifiable result.

Every response and every artifact derived from this adapter carries the
provenance string ``MOCK_DEMONSTRATION_FIXTURE_NOT_RESULTS``. The reporting code
refuses to print a run summary without displaying that banner. Tests assert on
both. Do not remove either.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

from .. import MOCK_PROVENANCE
from ..schema import AdapterResponse, Item
from .base import Adapter


@dataclass(frozen=True)
class MockProfile:
    """Outcome probabilities and citation behaviour for one simulated arm.

    The probabilities are inputs to a simulation, chosen to make the harness
    exercise every branch. They are not estimates of how any real system behaves,
    and nothing in this repository claims otherwise.
    """

    key: str
    arm: str
    description: str
    p_correct: float
    p_adjacent: float
    p_abstain: float
    p_cite_correct_clause: float
    p_cite_correct_section: float
    p_cite_wrong: float
    p_fabricate_paywalled: float
    confidence_correct: Tuple[float, float]
    confidence_wrong: Tuple[float, float]

    @property
    def p_other(self) -> float:
        return max(0.0, 1.0 - (self.p_correct + self.p_adjacent + self.p_abstain))


PROFILES: Mapping[str, MockProfile] = {
    "ungrounded": MockProfile(
        key="ungrounded",
        arm="ungrounded",
        description=(
            "Simulates a system with no retrieval: a persona prompt over parametric "
            "memory. Weighted towards adjacent substitution and towards citing "
            "nothing or citing a plausible wrong clause."
        ),
        p_correct=0.18,
        p_adjacent=0.60,
        p_abstain=0.03,
        p_cite_correct_clause=0.04,
        p_cite_correct_section=0.16,
        p_cite_wrong=0.35,
        p_fabricate_paywalled=0.22,
        confidence_correct=(0.72, 0.95),
        confidence_wrong=(0.62, 0.93),
    ),
    "pseudo_grounded": MockProfile(
        key="pseudo_grounded",
        arm="pseudo_grounded",
        description=(
            "Simulates retrieval over a plausible but unauthoritative corpus: vendor "
            "pages, training decks, internal wikis. Cites confidently and often "
            "cites the right document but the wrong clause."
        ),
        p_correct=0.44,
        p_adjacent=0.34,
        p_abstain=0.04,
        p_cite_correct_clause=0.18,
        p_cite_correct_section=0.46,
        p_cite_wrong=0.24,
        p_fabricate_paywalled=0.10,
        confidence_correct=(0.75, 0.97),
        confidence_wrong=(0.68, 0.95),
    ),
    "grounded": MockProfile(
        key="grounded",
        arm="grounded",
        description=(
            "Simulates retrieval over a version-pinned authoritative corpus with "
            "clause-level metadata. Higher clause-level citation accuracy and more "
            "abstention on items whose correct behaviour is to decline."
        ),
        p_correct=0.80,
        p_adjacent=0.07,
        p_abstain=0.07,
        p_cite_correct_clause=0.66,
        p_cite_correct_section=0.24,
        p_cite_wrong=0.06,
        p_fabricate_paywalled=0.02,
        confidence_correct=(0.70, 0.96),
        confidence_wrong=(0.40, 0.75),
    ),
    "abstainer": MockProfile(
        key="abstainer",
        arm="fixture",
        description=(
            "Always declines. Included as the degenerate baseline that any useful "
            "scoring policy must rank below a system that answers correctly, and "
            "above one that confabulates."
        ),
        p_correct=0.0,
        p_adjacent=0.0,
        p_abstain=1.0,
        p_cite_correct_clause=0.0,
        p_cite_correct_section=0.0,
        p_cite_wrong=0.0,
        p_fabricate_paywalled=0.0,
        confidence_correct=(0.0, 0.1),
        confidence_wrong=(0.0, 0.1),
    ),
}

_ABSTAIN_TEMPLATES = (
    "I do not know the answer to this with enough certainty to state it. "
    "You should consult the governing standard directly.",
    "I cannot confirm this from a source I can cite, so I am declining to answer "
    "rather than guess.",
    "I am unable to answer this reliably. The premise or the clause would need to "
    "be checked against the primary source.",
)

_OTHER_TEMPLATES = (
    "This is governed by the employer's written program, which should be reviewed "
    "by the site safety committee before any determination is made.",
    "The applicable requirement depends on facility-specific factors and on the "
    "jurisdiction, so a general statement would not be meaningful here.",
    "Industry practice varies. Most facilities handle this through their existing "
    "management system rather than by reference to a single provision.",
)

_FABRICATED_PAYWALLED = (
    "NFPA 70E Section 130.7",
    "API 520 Part I Section 5.4",
    "ASME BPVC UG-131",
    "ISO 4126-9",
    "API RP 754 Section 4.2",
)

_WRONG_CLAUSES = (
    "29 CFR 1910.132(d)(1)",
    "29 CFR 1910.38(b)",
    "40 CFR 68.15(a)",
    "29 CFR 1904.35(b)(1)",
    "46 CFR 54.01-1",
)


class MockAdapter(Adapter):
    """Deterministic simulated system. See the module docstring before using output."""

    def __init__(self, profile: str = "ungrounded", seed: int = 20260903, repeat: int = 0) -> None:
        if profile not in PROFILES:
            raise ValueError(
                "unknown mock profile %r; available: %s" % (profile, ", ".join(sorted(PROFILES)))
            )
        self.profile = PROFILES[profile]
        self.seed = int(seed)
        self.repeat = int(repeat)
        self.name = "mock:%s" % profile
        self.arm = self.profile.arm

    # -- determinism ------------------------------------------------------
    def _rng(self, item: Item) -> random.Random:
        """A per-item RNG, so results do not depend on iteration order.

        Seeding on (seed, profile, repeat, item id) means an item's simulated
        outcome is the same whether it is scored alone or in a batch, and a run
        can be reproduced from the seed printed in the run file.
        """
        return random.Random("%d|%s|%d|%s" % (self.seed, self.profile.key, self.repeat, item.id))

    def describe(self) -> Dict[str, object]:
        meta = super().describe()
        meta.update(
            {
                "profile": self.profile.key,
                "profile_description": self.profile.description,
                "seed": self.seed,
                "repeat": self.repeat,
                "provenance": MOCK_PROVENANCE,
                "warning": (
                    "Mock responses are composed from the corpus item being answered. "
                    "They demonstrate that the harness runs. They are not results and "
                    "say nothing about any real system."
                ),
            }
        )
        return meta

    # -- response construction -------------------------------------------
    def _correct_text(self, item: Item) -> str:
        return item.correct_answer

    def _adjacent_text(self, item: Item, rng: random.Random) -> str:
        terms = list(item.adjacent_wrong.signature_terms)
        rng.shuffle(terms)
        chosen = terms[: min(3, len(terms))]
        if not chosen:
            return (
                "This is determined by the %s, and the governing parameters follow "
                "from that device class." % item.adjacent_wrong.label
            )
        joined = ", ".join(chosen)
        return (
            "This is governed by the %s. The controlling parameters here are %s, and "
            "the value should be selected on that basis." % (item.adjacent_wrong.label, joined)
        )

    def _citation_text(self, item: Item, rng: random.Random) -> Tuple[str, List[str]]:
        p = self.profile
        draw = rng.random()
        if draw < p.p_cite_correct_clause:
            clause = item.source.clause
        elif draw < p.p_cite_correct_clause + p.p_cite_correct_section:
            expected = item.expected_citations
            clause = expected[0].as_text().split("(")[0].strip() if expected else item.source.clause
        elif draw < p.p_cite_correct_clause + p.p_cite_correct_section + p.p_cite_wrong:
            clause = rng.choice(_WRONG_CLAUSES)
        else:
            return "", []
        extra: List[str] = [clause]
        if rng.random() < p.p_fabricate_paywalled:
            extra.append(rng.choice(_FABRICATED_PAYWALLED))
        return " See %s." % "; ".join(extra), extra

    def answer(self, item: Item) -> AdapterResponse:
        rng = self._rng(item)
        p = self.profile
        draw = rng.random()

        if draw < p.p_correct:
            branch = "correct"
            body = self._correct_text(item)
        elif draw < p.p_correct + p.p_adjacent:
            branch = "adjacent"
            body = self._adjacent_text(item, rng)
        elif draw < p.p_correct + p.p_adjacent + p.p_abstain:
            branch = "abstain"
            body = rng.choice(_ABSTAIN_TEMPLATES)
        else:
            branch = "other"
            body = rng.choice(_OTHER_TEMPLATES)

        citation_suffix, citations = ("", [])
        if branch != "abstain":
            citation_suffix, citations = self._citation_text(item, rng)

        lo, hi = p.confidence_correct if branch == "correct" else p.confidence_wrong
        confidence = round(rng.uniform(lo, hi), 4)
        if branch == "abstain":
            confidence = round(rng.uniform(0.05, 0.35), 4)

        return AdapterResponse(
            item_id=item.id,
            text=body + citation_suffix,
            citations=citations,
            confidence=confidence,
            latency_s=None,
            error=None,
            metadata={
                "simulated_branch": branch,
                "profile": p.key,
                "item_type": item.item_type.value,
            },
            provenance=MOCK_PROVENANCE,
        )


def available_profiles() -> Tuple[str, ...]:
    return tuple(sorted(PROFILES))
