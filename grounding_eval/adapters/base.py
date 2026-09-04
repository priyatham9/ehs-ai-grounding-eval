"""The adapter interface.

An adapter is anything that can turn a benchmark item into an
:class:`~grounding_eval.schema.AdapterResponse`. Grounded and ungrounded systems
are scored through the same interface with the same code path, so that a measured
difference between two arms cannot come from a difference in how they were asked
or how their answers were parsed.

Adapters must not see the answer key. ``Adapter.answer`` is handed the whole
:class:`~grounding_eval.schema.Item` for convenience, and the mock adapter does
read the key because it is a fixture rather than a system under test. Any adapter
representing a real system must use only ``item.question`` and, at most,
``item.id``. That rule is enforced socially, not mechanically; see
docs/methodology.md.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Iterable, List, Optional

from ..schema import AdapterResponse, Item


class Adapter(abc.ABC):
    """Base class for systems under test."""

    #: Short identifier used in run filenames and report tables.
    name: str = "adapter"

    #: One of "grounded", "pseudo_grounded", "ungrounded", "fixture", "unknown".
    #: Purely descriptive; the harness does not act on it.
    arm: str = "unknown"

    @abc.abstractmethod
    def answer(self, item: Item) -> AdapterResponse:
        """Return this system's answer to one item."""

    def describe(self) -> Dict[str, Any]:
        """Metadata recorded in the run file so a run can be traced to its source."""
        return {"name": self.name, "arm": self.arm, "class": type(self).__name__}

    def answer_all(self, items: Iterable[Item]) -> List[AdapterResponse]:
        return [self.answer(item) for item in items]

    def close(self) -> None:  # pragma: no cover - default is a no-op
        """Release any resources. Called by the runner in a finally block."""
        return None


class StaticAdapter(Adapter):
    """Returns a fixed answer for every item. Used in tests and as a floor."""

    name = "static"
    arm = "fixture"

    def __init__(self, text: str, name: Optional[str] = None, confidence: Optional[float] = None) -> None:
        self._text = text
        self._confidence = confidence
        if name:
            self.name = name

    def answer(self, item: Item) -> AdapterResponse:
        return AdapterResponse(
            item_id=item.id,
            text=self._text,
            confidence=self._confidence,
            provenance="static_fixture",
        )
