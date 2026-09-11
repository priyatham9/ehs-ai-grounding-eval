"""BM25 retrieval baseline.

Builds an Okapi BM25 index over the authoritative text the corpus items carry:
each item's source excerpt (``anchor_text``), section title, standard and clause.
The adapter answers a question with the best-matching excerpt and cites the clause
it came from. It never reads the answer key or the adjacent-wrong entry, so it is
an honest non-LLM system under test: retrieval with no generation on top.

The index is built over the corpus at construction. Documents belonging to the
item being answered are not excluded, since a deployed retriever would have the
governing clause in its store; what is measured is whether lexical overlap with
the question selects it, and whether the excerpt alone satisfies the answer key.

BM25 parameters: k1=1.2, b=0.75 (Okapi BM25 defaults).
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..corpus import Corpus, load_corpus
from ..schema import AdapterResponse, Item
from .base import Adapter
from .retrieval import corpus_documents, tokenize

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")


class BM25Index:
    """Okapi BM25 ranking over a list of (doc_id, text) pairs."""

    def __init__(self, docs: Sequence[Tuple[str, str]], k1: float = 1.2, b: float = 0.75) -> None:
        self.ids = [d[0] for d in docs]
        self.k1 = k1
        self.b = b
        tokenized = [tokenize(d[1]) for d in docs]
        vocab: Dict[str, int] = {}
        for toks in tokenized:
            for t in toks:
                vocab.setdefault(t, len(vocab))
        self.vocab = vocab
        n_docs = len(docs)
        self.n_docs = n_docs

        # Document frequencies
        df = np.zeros(len(vocab), dtype=int)
        for toks in tokenized:
            for t in set(toks):
                df[vocab[t]] += 1

        # Compute IDF: log((N - df + 0.5) / (df + 0.5))
        self.idf = np.log((n_docs - df + 0.5) / (df + 0.5))

        # Term frequencies in each document
        self.tf_matrix = []
        doc_lengths = []
        for toks in tokenized:
            tf = np.zeros(len(vocab), dtype=int)
            for t in toks:
                j = vocab.get(t)
                if j is not None:
                    tf[j] += 1
            self.tf_matrix.append(tf)
            doc_lengths.append(sum(tf))

        self.doc_lengths = np.array(doc_lengths, dtype=float)
        self.avgdl = np.mean(self.doc_lengths) if len(docs) > 0 else 0.0

    def query(self, text: str) -> List[Tuple[str, float]]:
        """Rank documents by BM25 score."""
        toks = tokenize(text)
        scores = np.zeros(self.n_docs)

        for t in toks:
            j = self.vocab.get(t)
            if j is None:
                continue
            idf = self.idf[j]
            for i, doc_tf in enumerate(self.tf_matrix):
                f = doc_tf[j]
                if f > 0:
                    dl = self.doc_lengths[i]
                    score = idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * (dl / self.avgdl)))
                    scores[i] += score

        order = np.argsort(-scores, kind="stable")
        return [(self.ids[i], float(scores[i])) for i in order]


class BM25Adapter(Adapter):
    name = "retrieval_bm25"
    arm = "grounded"

    def __init__(self, corpus: Optional[Corpus] = None) -> None:
        corpus = corpus if corpus is not None else load_corpus()
        self._by_id = {item.id: item for item in corpus}
        self.index = BM25Index(corpus_documents(corpus))

    def describe(self) -> Dict[str, object]:
        meta = super().describe()
        meta.update({"method": "bm25", "n_documents": len(self.index.ids), "vocab_size": len(self.index.vocab)})
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
            provenance="retrieval_bm25_baseline",
        )
