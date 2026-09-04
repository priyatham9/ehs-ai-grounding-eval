"""The run loop: put a corpus through an adapter, score it, write a run file.

Design commitments
------------------
**Every arm takes the same code path.** The grounded and ungrounded systems are
asked in the same order, parsed by the same parser and scored by the same scorer.
A measured difference between two arms therefore cannot come from a difference in
how they were prompted or read. That is the whole point of the adapter interface.

**Single runs are not reported.** ``run_repeats`` executes the same adapter over
the same corpus several times and records each repeat separately, so that
run-to-run variance is visible rather than averaged away. In a safety setting an
answer that is correct only sometimes is not a correct answer, and a benchmark
that hides that is misleading.

**Run files are self-describing.** Each one records the corpus digest, the
adapter's own description of itself, the scoring configuration, the seed, the
repeat index and the library version. A run file that cannot be traced back to
the corpus that produced it is not evidence, so the digest is checked on load.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import MOCK_PROVENANCE, __version__
from .adapters.base import Adapter
from .corpus import Corpus, load_corpus
from .schema import AdapterResponse, Item, ItemScore
from .scoring import ScoringConfig, score_item

__all__ = ["RunResult", "corpus_digest", "run_once", "run_repeats", "write_run", "load_run"]


def corpus_digest(corpus: Corpus) -> str:
    """Stable SHA-256 over the identity-bearing fields of every item.

    Covers the question, the reference answer, the answer key, the adjacency
    signature and the source clause. Editing any of those changes the digest, so
    a run file cannot be silently re-attributed to a corpus it was not produced
    from. Cosmetic fields such as notes are excluded on purpose: annotating an
    item should not invalidate a run.
    """
    hasher = hashlib.sha256()
    for item in sorted(corpus, key=lambda i: i.id):
        parts = [
            item.id,
            item.domain,
            item.item_type.value,
            item.family,
            item.question,
            item.correct_answer,
            repr(item.answer_key.required_concepts),
            repr(item.answer_key.forbidden_concepts),
            repr([(f.label, f.value, f.unit, f.tolerance_abs) for f in item.answer_key.numeric_facts]),
            repr(item.adjacent_wrong.signature_terms),
            item.source.clause,
            str(item.pair_id),
            str(item.pair_role),
        ]
        hasher.update("\x1f".join(parts).encode("utf-8"))
        hasher.update(b"\x1e")
    return hasher.hexdigest()


@dataclass
class RunResult:
    """One pass of one adapter over one corpus, with its scores."""

    adapter: Dict[str, Any]
    corpus_digest: str
    n_items: int
    repeat: int
    seed: Optional[int]
    scoring_config: Dict[str, Any]
    responses: List[AdapterResponse]
    scores: List[ItemScore]
    started_utc: str
    duration_s: float
    environment: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_mock(self) -> bool:
        """True when any response came from the demonstration fixture.

        Reporting code keys the mock banner off this. It checks the responses
        themselves rather than the adapter name, so renaming the adapter cannot
        strip the label off a fixture run.
        """
        return any(r.provenance == MOCK_PROVENANCE for r in self.responses)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "grounding_eval.run/1",
            "library_version": __version__,
            "adapter": self.adapter,
            "corpus_digest": self.corpus_digest,
            "n_items": self.n_items,
            "repeat": self.repeat,
            "seed": self.seed,
            "scoring_config": self.scoring_config,
            "started_utc": self.started_utc,
            "duration_s": self.duration_s,
            "environment": self.environment,
            "is_mock_demonstration": self.is_mock,
            "mock_banner": MOCK_PROVENANCE if self.is_mock else None,
            "responses": [r.to_dict() for r in self.responses],
            "scores": [s.to_dict() for s in self.scores],
        }


def _environment() -> Dict[str, Any]:
    """Recorded so a run can be reproduced, or its non-reproduction explained."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
    }


def run_once(
    adapter: Adapter,
    corpus: Optional[Corpus] = None,
    config: Optional[ScoringConfig] = None,
    repeat: int = 0,
    seed: Optional[int] = None,
    progress: bool = False,
) -> RunResult:
    """Ask every item once, score every response, return the run.

    An adapter that raises on an item does not abort the run: the exception is
    captured into the response's ``error`` field and the item is scored
    UNSCORABLE. An infrastructure failure and a confabulation are different
    events, and collapsing them would corrupt both rates.
    """
    corpus = corpus if corpus is not None else load_corpus()
    cfg = config or ScoringConfig()
    started = time.gmtime()
    t0 = time.time()

    responses: List[AdapterResponse] = []
    try:
        for index, item in enumerate(corpus, start=1):
            if progress:
                sys.stderr.write("\r[%s] %d/%d %s" % (adapter.name, index, len(corpus), item.id.ljust(12)))
                sys.stderr.flush()
            item_t0 = time.time()
            try:
                response = adapter.answer(item)
            except Exception as exc:  # noqa: BLE001 - deliberately broad; see docstring
                response = AdapterResponse(
                    item_id=item.id,
                    text="",
                    error="%s: %s" % (type(exc).__name__, exc),
                    provenance="adapter_exception",
                )
            if response.latency_s is None:
                response.latency_s = time.time() - item_t0
            if response.item_id != item.id:
                raise ValueError(
                    "adapter %r returned a response for %r while answering %r"
                    % (adapter.name, response.item_id, item.id)
                )
            responses.append(response)
    finally:
        if progress:
            sys.stderr.write("\n")
        adapter.close()

    scores = [score_item(item, response, cfg) for item, response in zip(corpus, responses)]

    return RunResult(
        adapter=adapter.describe(),
        corpus_digest=corpus_digest(corpus),
        n_items=len(corpus),
        repeat=repeat,
        seed=seed,
        scoring_config=cfg.describe(),
        responses=responses,
        scores=scores,
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", started),
        duration_s=time.time() - t0,
        environment=_environment(),
    )


def run_repeats(
    adapter_factory,
    n_runs: int = 3,
    corpus: Optional[Corpus] = None,
    config: Optional[ScoringConfig] = None,
    progress: bool = False,
) -> List[RunResult]:
    """Run the same adapter configuration ``n_runs`` times.

    ``adapter_factory`` is called with the repeat index and must return a fresh
    adapter. It is a factory rather than an instance so that a stateful adapter
    (an open HTTP session, a cached index) starts each repeat clean, and so that
    a seeded adapter can vary its seed by repeat.

    Repeats are returned separately and are never averaged here. Aggregation is
    the reporting layer's job, and it reports the spread alongside the centre.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1; got %d" % n_runs)
    corpus = corpus if corpus is not None else load_corpus()
    out: List[RunResult] = []
    for repeat in range(n_runs):
        adapter = adapter_factory(repeat)
        seed = getattr(adapter, "seed", None)
        out.append(
            run_once(
                adapter,
                corpus=corpus,
                config=config,
                repeat=repeat,
                seed=seed,
                progress=progress,
            )
        )
    return out


_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_stem(label: str) -> str:
    """Reduce an adapter name to characters that are legal in a filename everywhere.

    Adapter names may legitimately contain colons and spaces (``mock:grounded``),
    which are illegal on Windows and awkward in shell pipelines. The adapter's
    real name is preserved inside the run file; only the filename is sanitised.
    """
    cleaned = _UNSAFE_FILENAME_CHARS.sub("-", label).strip("-")
    return cleaned or "adapter"


def write_run(run: RunResult, directory: str, label: Optional[str] = None) -> str:
    """Write one run to ``directory`` as JSON and return the path.

    Mock runs are written with a ``mock__`` filename prefix in addition to the
    in-file banner, so that a directory listing alone shows what is a fixture.
    """
    os.makedirs(directory, exist_ok=True)
    stem = _safe_stem(label or run.adapter.get("name", "adapter"))
    prefix = "mock__" if run.is_mock else ""
    filename = "%s%s__repeat%02d.json" % (prefix, stem, run.repeat)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(run.to_dict(), handle, indent=2)
        handle.write("\n")
    return path


def load_run(path: str, corpus: Optional[Corpus] = None) -> Dict[str, Any]:
    """Read a run file, optionally checking it against the current corpus.

    When a corpus is supplied and the digests differ, this raises. A run scored
    against a different version of the corpus is not comparable with one scored
    against this version, and quietly reporting them side by side would be a
    silent error of exactly the kind this repository exists to argue against.
    """
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if corpus is not None:
        expected = corpus_digest(corpus)
        found = payload.get("corpus_digest")
        if found != expected:
            raise ValueError(
                "run file %s was produced against corpus digest %s but the current "
                "corpus digest is %s; these results are not comparable"
                % (os.path.basename(path), found, expected)
            )
    return payload
