"""Lexical retrieval baseline.

Builds a TF-IDF index (numpy, cosine similarity) over the authoritative text the
corpus items carry: each item's source excerpt (``anchor_text``), section title,
standard and clause. The adapter answers a question with the best-matching
excerpt and cites the clause it came from. It never reads the answer key or the
adjacent-wrong entry, so it is an honest non-LLM system under test: retrieval
with no generation on top.

The index is built over the corpus at construction. Documents belonging to the
item being answered are not excluded, since a deployed retriever would have the
governing clause in its store; what is measured is whether lexical overlap with
the question selects it, and whether the excerpt alone satisfies the answer key.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..corpus import Corpus, load_corpus
from ..schema import AdapterResponse, Item
from .base import Adapter

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class TfidfIndex:
    """Minimal TF-IDF over a list of (doc_id, text) pairs."""

    def __init__(self, docs: Sequence[Tuple[str, str]]) -> None:
        self.ids = [d[0] for d in docs]
        tokenized = [tokenize(d[1]) for d in docs]
        vocab: Dict[str, int] = {}
        for toks in tokenized:
            for t in toks:
                vocab.setdefault(t, len(vocab))
        self.vocab = vocab
        n_docs = len(docs)
        df = np.zeros(len(vocab))
        for toks in tokenized:
            for t in set(toks):
                df[vocab[t]] += 1
        self.idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        self.matrix = np.vstack([self._vector(toks) for toks in tokenized]) if docs else np.zeros((0, 0))

    def _vector(self, toks: Iterable[str]) -> np.ndarray:
        vec = np.zeros(len(self.vocab))
        for t in toks:
            j = self.vocab.get(t)
            if j is not None:
                vec[j] += 1.0
        vec = np.where(vec > 0, 1.0 + np.log(np.maximum(vec, 1.0)), 0.0)
        vec = vec * self.idf
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def query(self, text: str) -> List[Tuple[str, float]]:
        q = self._vector(tokenize(text))
        # Elementwise product rather than BLAS matmul: the Accelerate BLAS shipped
        # with numpy on macOS emits spurious floating-point warnings on tiny matrices.
        scores = (self.matrix * q).sum(axis=1)
        order = np.argsort(-scores, kind="stable")
        return [(self.ids[i], float(scores[i])) for i in order]


def corpus_documents(corpus: Corpus) -> List[Tuple[str, str]]:
    """One document per item, built only from source metadata and excerpt."""
    docs: List[Tuple[str, str]] = []
    for item in corpus:
        src = item.source
        text = " ".join([src.standard, src.clause, src.section_title, src.anchor_text])
        docs.append((item.id, text))
    return docs


class RetrievalAdapter(Adapter):
    name = "retrieval_tfidf"
    arm = "grounded"

    def __init__(self, corpus: Optional[Corpus] = None) -> None:
        corpus = corpus if corpus is not None else load_corpus()
        self._by_id = {item.id: item for item in corpus}
        self.index = TfidfIndex(corpus_documents(corpus))

    def describe(self) -> Dict[str, object]:
        meta = super().describe()
        meta.update({"method": "tfidf_cosine", "n_documents": len(self.index.ids), "vocab_size": len(self.index.vocab)})
        return meta

    def answer(self, item: Item) -> AdapterResponse:
        ranked = self.index.query(item.question)
        best_id, score = ranked[0]
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
            provenance="retrieval_tfidf_baseline",
        )
