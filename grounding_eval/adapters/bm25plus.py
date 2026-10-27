"""BM25+ retrieval baseline and a TF-IDF/BM25 rank-fusion ensemble.

Two further non-LLM retrievers over the same documents the TF-IDF and BM25
baselines index (``corpus_documents``: standard, clause, section title and the
verified source excerpt of each item). Like those baselines they answer with the
best-matching excerpt, cite its clause, and never read the answer key or the
adjacent-wrong entry.

* ``retrieval_bm25plus``: BM25+ (Lv and Zhai, "Lower-bounding term frequency
  normalization", CIKM 2011). Each matched term contributes
  ``idf * ((k1 + 1) f / (f + k1 (1 - b + b dl/avgdl)) + delta)`` with
  ``idf = ln((N + 1) / df)``, which is never negative. The plain Okapi IDF used by
  ``retrieval_bm25``, ``ln((N - df + 0.5) / (df + 0.5))``, goes negative for any
  term in more than half of a 68-document index, so matching a common term can
  lower a document's score; BM25+ removes that and lower-bounds the reward for
  any match in a long document.
* ``retrieval_rrf``: reciprocal rank fusion (Cormack, Clarke and Buettcher,
  SIGIR 2009) of the TF-IDF and Okapi BM25 rankings, score ``sum 1 / (k + rank)``
  with ranks counted from 1. Ties are broken by corpus order.

Parameters were fixed in this module, at the values the cited papers recommend,
before either adapter was run on the corpus, and were not changed afterwards:
k1 = 1.2, b = 0.75, delta = 1.0 for BM25+, and k = 60 for RRF. Exactly one
configuration of each was run; nothing was tuned on the 68 items.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..corpus import Corpus, load_corpus
from ..schema import AdapterResponse, Item
from .base import Adapter
from .bm25 import BM25Index
from .retrieval import TfidfIndex, corpus_documents, tokenize

BM25PLUS_K1 = 1.2
BM25PLUS_B = 0.75
BM25PLUS_DELTA = 1.0
RRF_K = 60


class BM25PlusIndex:
    """BM25+ ranking over a list of (doc_id, text) pairs."""

    def __init__(
        self,
        docs: Sequence[Tuple[str, str]],
        k1: float = BM25PLUS_K1,
        b: float = BM25PLUS_B,
        delta: float = BM25PLUS_DELTA,
    ) -> None:
        self.ids = [d[0] for d in docs]
        self.k1, self.b, self.delta = k1, b, delta
        tokenized = [tokenize(d[1]) for d in docs]
        vocab: Dict[str, int] = {}
        for toks in tokenized:
            for t in toks:
                vocab.setdefault(t, len(vocab))
        self.vocab = vocab
        self.n_docs = len(docs)
        tf = np.zeros((self.n_docs, len(vocab)), dtype=float)
        for i, toks in enumerate(tokenized):
            for t in toks:
                tf[i, vocab[t]] += 1.0
        self.tf = tf
        df = (tf > 0).sum(axis=0)
        self.idf = np.log((self.n_docs + 1.0) / np.maximum(df, 1)) if self.n_docs else np.zeros(0)
        self.doc_lengths = tf.sum(axis=1)
        self.avgdl = float(self.doc_lengths.mean()) if self.n_docs else 0.0

    def query(self, text: str) -> List[Tuple[str, float]]:
        scores = np.zeros(self.n_docs)
        if self.n_docs and self.avgdl > 0:
            norm = self.k1 * (1.0 - self.b + self.b * self.doc_lengths / self.avgdl)
            for t in tokenize(text):
                j = self.vocab.get(t)
                if j is None:
                    continue
                f = self.tf[:, j]
                contrib = self.idf[j] * ((self.k1 + 1.0) * f / (f + norm) + self.delta)
                scores += np.where(f > 0, contrib, 0.0)
        order = np.argsort(-scores, kind="stable")
        return [(self.ids[i], float(scores[i])) for i in order]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> List[Tuple[str, float]]:
    """Fuse ranked id lists; ties keep the order of first appearance in ``rankings[0]``."""
    scores: Dict[str, float] = {}
    first_seen: Dict[str, int] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(doc_id, len(first_seen))
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


class _ExcerptRetriever(Adapter):
    """Shared answer format: best-matching excerpt, cited by its clause."""

    arm = "grounded"
    provenance = ""

    def __init__(self, corpus: Optional[Corpus]) -> None:
        corpus = corpus if corpus is not None else load_corpus()
        self._by_id = {item.id: item for item in corpus}
        self._docs = corpus_documents(corpus)

    def rank(self, question: str) -> List[Tuple[str, float]]:  # pragma: no cover - abstract
        raise NotImplementedError

    def answer(self, item: Item) -> AdapterResponse:
        best_id, score = self.rank(item.question)[0]
        src = self._by_id[best_id].source
        text = "%s (%s, %s) states: %s See %s." % (
            src.standard, src.clause, src.section_title, src.anchor_text, src.clause
        )
        return AdapterResponse(
            item_id=item.id,
            text=text,
            citations=[src.clause],
            confidence=round(score, 4) if not math.isnan(score) else None,
            metadata={"matched_item": best_id, "similarity": score, "self_match": best_id == item.id},
            provenance=self.provenance,
        )


class BM25PlusAdapter(_ExcerptRetriever):
    name = "retrieval_bm25plus"
    provenance = "retrieval_bm25plus_baseline"

    def __init__(self, corpus: Optional[Corpus] = None) -> None:
        super().__init__(corpus)
        self.index = BM25PlusIndex(self._docs)

    def describe(self) -> Dict[str, object]:
        meta = super().describe()
        meta.update({
            "method": "bm25plus", "k1": self.index.k1, "b": self.index.b, "delta": self.index.delta,
            "n_documents": len(self.index.ids), "vocab_size": len(self.index.vocab),
        })
        return meta

    def rank(self, question: str) -> List[Tuple[str, float]]:
        return self.index.query(question)


class RRFAdapter(_ExcerptRetriever):
    name = "retrieval_rrf"
    provenance = "retrieval_rrf_baseline"

    def __init__(self, corpus: Optional[Corpus] = None, k: int = RRF_K) -> None:
        super().__init__(corpus)
        self.k = k
        self.tfidf = TfidfIndex(self._docs)
        self.bm25 = BM25Index(self._docs)

    def describe(self) -> Dict[str, object]:
        meta = super().describe()
        meta.update({"method": "rrf(tfidf_cosine, bm25)", "rrf_k": self.k, "n_documents": len(self._docs)})
        return meta

    def rank(self, question: str) -> List[Tuple[str, float]]:
        runs = [[d for d, _ in self.tfidf.query(question)], [d for d, _ in self.bm25.query(question)]]
        return reciprocal_rank_fusion(runs, k=self.k)
