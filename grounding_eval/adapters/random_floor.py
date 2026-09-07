"""Random floor baseline.

Picks uniformly, per item, between the two answer options the corpus item
carries (the reference answer and the adjacent-wrong answer) and cites a clause
drawn uniformly from the pool of clauses cited anywhere in the corpus. It is a
floor: a system that knows nothing but the shape of the item scores here.

Like the mock fixture, this adapter reads the item beyond ``question`` and
``id``. It is a baseline for calibrating the scale, not a system under test, and
its arm is labelled ``fixture`` so reports cannot present it as one.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

from ..schema import AdapterResponse, Item
from .base import Adapter


def _adjacent_text(item: Item, rng: random.Random) -> str:
    terms = list(item.adjacent_wrong.signature_terms)
    rng.shuffle(terms)
    chosen = terms[: min(3, len(terms))]
    if not chosen:
        return "This is determined by the %s." % item.adjacent_wrong.label
    return "This is governed by the %s. The controlling parameters here are %s." % (
        item.adjacent_wrong.label,
        ", ".join(chosen),
    )


class RandomFloorAdapter(Adapter):
    """Uniform choice between the item's answer options; uniform random citation."""

    name = "random_floor"
    arm = "fixture"

    def __init__(self, seed: int = 20260907, repeat: int = 0, clause_pool: Optional[Sequence[str]] = None) -> None:
        self.seed = int(seed)
        self.repeat = int(repeat)
        self.clause_pool: List[str] = sorted(set(clause_pool or []))

    def _rng(self, item: Item) -> random.Random:
        return random.Random("%d|random_floor|%d|%s" % (self.seed, self.repeat, item.id))

    def describe(self) -> Dict[str, object]:
        meta = super().describe()
        meta.update({"seed": self.seed, "repeat": self.repeat, "n_clause_pool": len(self.clause_pool)})
        return meta

    def answer(self, item: Item) -> AdapterResponse:
        rng = self._rng(item)
        option = rng.choice(("correct", "adjacent"))
        body = item.correct_answer if option == "correct" else _adjacent_text(item, rng)
        citations: List[str] = []
        suffix = ""
        if self.clause_pool:
            clause = rng.choice(self.clause_pool)
            citations = [clause]
            suffix = " See %s." % clause
        return AdapterResponse(
            item_id=item.id,
            text=body + suffix,
            citations=citations,
            confidence=None,
            metadata={"option": option},
            provenance="random_floor_baseline",
        )
