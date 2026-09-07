"""Oracle ceiling.

Returns the reference answer with the gold citation. It exists to show what a
perfect score looks like under this scoring procedure, so that the gap between
the floor and the ceiling can be read as the room the items leave for a real
system to be discriminated. It reads the answer key by construction and is
labelled ``fixture``.
"""

from __future__ import annotations

from ..schema import AdapterResponse, Item
from .base import Adapter


class OracleAdapter(Adapter):
    name = "oracle"
    arm = "fixture"

    def answer(self, item: Item) -> AdapterResponse:
        clause = item.source.clause
        return AdapterResponse(
            item_id=item.id,
            text="%s See %s." % (item.correct_answer, clause),
            citations=[clause],
            confidence=None,
            provenance="oracle_ceiling",
        )
