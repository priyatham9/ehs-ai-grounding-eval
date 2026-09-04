"""Loading and validating the question corpus.

The loader reads every ``*.json`` file under ``corpus/items/`` and returns items
in a stable, deterministic order (sorted by id). It never reads
``corpus/quarantine/``; items whose provenance could not be verified are excluded
by construction rather than by a runtime flag.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .schema import (
    SCHEMA_VERSION,
    CorpusValidationError,
    Item,
    ItemType,
    item_from_dict,
)

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_PACKAGE_DIR)
DEFAULT_ITEMS_DIR = os.path.join(REPO_ROOT, "corpus", "items")
QUARANTINE_PATH = os.path.join(REPO_ROOT, "corpus", "quarantine", "unverified.json")


@dataclass(frozen=True)
class Corpus:
    """An immutable, validated collection of benchmark items."""

    items: Tuple[Item, ...]
    source_files: Tuple[str, ...]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Item]:
        return iter(self.items)

    def by_id(self, item_id: str) -> Item:
        for item in self.items:
            if item.id == item_id:
                return item
        raise KeyError(item_id)

    def ids(self) -> Tuple[str, ...]:
        return tuple(item.id for item in self.items)

    def domains(self) -> Tuple[str, ...]:
        return tuple(sorted({item.domain for item in self.items}))

    def families(self) -> Tuple[str, ...]:
        return tuple(sorted({item.family for item in self.items}))

    def factual(self) -> Tuple[Item, ...]:
        return tuple(i for i in self.items if i.item_type is ItemType.FACTUAL)

    def category_errors(self) -> Tuple[Item, ...]:
        return tuple(i for i in self.items if i.item_type is ItemType.CATEGORY_ERROR)

    def pairs(self) -> Tuple[Tuple[Item, Item], ...]:
        """Return complete minimal pairs as (role 'a' item, role 'b' item)."""
        buckets: Dict[str, Dict[str, Item]] = {}
        for item in self.items:
            if item.pair_id is None or item.pair_role is None:
                continue
            buckets.setdefault(item.pair_id, {})[item.pair_role] = item
        out: List[Tuple[Item, Item]] = []
        for pair_id in sorted(buckets):
            roles = buckets[pair_id]
            if "a" in roles and "b" in roles:
                out.append((roles["a"], roles["b"]))
        return tuple(out)

    def filter(
        self,
        domains: Optional[Sequence[str]] = None,
        item_ids: Optional[Sequence[str]] = None,
        item_type: Optional[ItemType] = None,
    ) -> "Corpus":
        selected = self.items
        if domains:
            wanted = set(domains)
            selected = tuple(i for i in selected if i.domain in wanted)
        if item_ids:
            wanted_ids = set(item_ids)
            selected = tuple(i for i in selected if i.id in wanted_ids)
        if item_type is not None:
            selected = tuple(i for i in selected if i.item_type is item_type)
        return Corpus(items=selected, source_files=self.source_files)


def load_corpus(items_dir: str = DEFAULT_ITEMS_DIR) -> Corpus:
    """Read, validate and return the corpus.

    Raises :class:`CorpusValidationError` on a duplicate id, an unverified item,
    a malformed answer key, a source clause that does not parse into a citation,
    or a dangling minimal pair.
    """
    if not os.path.isdir(items_dir):
        raise CorpusValidationError("corpus item directory not found: %s" % items_dir)

    paths = sorted(
        os.path.join(items_dir, name)
        for name in os.listdir(items_dir)
        if name.endswith(".json")
    )
    if not paths:
        raise CorpusValidationError("no corpus item files found in %s" % items_dir)

    items: List[Item] = []
    seen_ids: Dict[str, str] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        version = str(payload.get("schema_version", ""))
        if version != SCHEMA_VERSION:
            raise CorpusValidationError(
                "%s: schema_version %r does not match expected %r"
                % (os.path.basename(path), version, SCHEMA_VERSION)
            )
        domain = str(payload.get("domain") or "").strip()
        if not domain:
            raise CorpusValidationError("%s: missing 'domain'" % os.path.basename(path))
        raw_items = payload.get("items") or []
        if not raw_items:
            raise CorpusValidationError("%s: contains no items" % os.path.basename(path))
        for raw in raw_items:
            item = item_from_dict(raw, domain=domain)
            if item.id in seen_ids:
                raise CorpusValidationError(
                    "duplicate item id %r in %s (already defined in %s)"
                    % (item.id, os.path.basename(path), seen_ids[item.id])
                )
            seen_ids[item.id] = os.path.basename(path)
            items.append(item)

    items.sort(key=lambda i: i.id)
    corpus = Corpus(items=tuple(items), source_files=tuple(paths))
    _check_pairs(corpus)
    return corpus


def _check_pairs(corpus: Corpus) -> None:
    buckets: Dict[str, List[Item]] = {}
    for item in corpus:
        if item.pair_id is None:
            continue
        buckets.setdefault(item.pair_id, []).append(item)
    for pair_id, members in sorted(buckets.items()):
        if len(members) != 2:
            raise CorpusValidationError(
                "pair %r has %d members (%s); minimal pairs must have exactly two"
                % (pair_id, len(members), ", ".join(sorted(m.id for m in members)))
            )
        roles = sorted(m.pair_role or "" for m in members)
        if roles != ["a", "b"]:
            raise CorpusValidationError(
                "pair %r has roles %s; expected exactly one 'a' and one 'b'" % (pair_id, roles)
            )


def load_quarantine(path: str = QUARANTINE_PATH) -> dict:
    """Read the quarantine file. Exposed for reporting and tests, never for scoring."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") != "EXCLUDED_FROM_SCORING":
        raise CorpusValidationError(
            "quarantine file must declare status EXCLUDED_FROM_SCORING; got %r"
            % payload.get("status")
        )
    return payload


def summarize(corpus: Corpus) -> Dict[str, object]:
    """A small dictionary describing corpus composition, for the CLI and README."""
    by_domain: Dict[str, int] = {}
    by_tier: Dict[str, int] = {}
    for item in corpus:
        by_domain[item.domain] = by_domain.get(item.domain, 0) + 1
        by_tier[item.risk_tier.value] = by_tier.get(item.risk_tier.value, 0) + 1
    return {
        "n_items": len(corpus),
        "n_factual": len(corpus.factual()),
        "n_category_error": len(corpus.category_errors()),
        "n_minimal_pairs": len(corpus.pairs()),
        "n_domains": len(corpus.domains()),
        "n_families": len(corpus.families()),
        "by_domain": dict(sorted(by_domain.items())),
        "by_risk_tier": dict(sorted(by_tier.items())),
    }
