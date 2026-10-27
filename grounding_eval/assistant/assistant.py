"""The assistant: retrieve, decide, compose, cite, log.

Abstain rule (documented in docs/assistant.md)
----------------------------------------------
Let Q be the distinct tokens of the query. Each token t has weight
``idf(t) = ln((N + 1) / df(t))`` (the BM25+ IDF over the N corpus passages; a token
absent from the corpus has df treated as 1, the maximum weight). The *evidence
coverage* of a passage is the weight of the query tokens it contains divided by
the weight of all of Q. The assistant answers only when the top-ranked passage has
coverage >= ``ABSTAIN_THRESHOLD`` (0.30). Otherwise it abstains and cites nothing.

The 0.30 default was fixed after inspecting two example queries (an on-topic
question with a matching passage scored 0.44; an off-topic one scored 0.10) and
before the benchmark was run through the assistant. It was not changed afterwards.
scripts/run_assistant_eval.py reports a sensitivity sweep over other values,
labelled as such, and does not select a threshold from it.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional

from ..adapters.base import Adapter
from ..adapters.bm25 import BM25Index
from ..adapters.bm25plus import BM25PlusIndex
from ..adapters.retrieval import corpus_documents, tokenize
from ..corpus import Corpus, load_corpus
from ..harness import corpus_digest
from ..schema import AdapterResponse, Item
from .model import AnswerModel, MockModel, Passage

ABSTAIN_THRESHOLD = 0.30
TOP_K = 3
ABSTAIN_TEXT = (
    "I cannot answer this from the governed corpus: the retrieved evidence is insufficient "
    "(evidence coverage %.2f < threshold %.2f). No source is cited."
)


def coverage_of(query_tokens: List[str], doc_tokens: set, idf: Dict[str, float], unknown_weight: float) -> float:
    """IDF-weighted share of distinct query tokens found in the passage."""
    uniq = list(dict.fromkeys(query_tokens))
    total = sum(idf.get(t, unknown_weight) for t in uniq)
    if total <= 0:
        return 0.0
    return sum(idf.get(t, unknown_weight) for t in uniq if t in doc_tokens) / total


class AssistantAnswer:
    """Result of one question: decision, text, citations and the audit record."""

    def __init__(self, decision: str, text: str, citations: List[str], audit: Dict[str, Any]) -> None:
        self.decision = decision
        self.text = text
        self.citations = citations
        self.audit = audit

    @property
    def abstained(self) -> bool:
        return self.decision == "abstain"


class Assistant:
    def __init__(
        self,
        corpus: Optional[Corpus] = None,
        retriever: str = "bm25plus",
        threshold: float = ABSTAIN_THRESHOLD,
        model: Optional[AnswerModel] = None,
    ) -> None:
        if retriever not in ("bm25plus", "bm25"):
            raise ValueError("retriever must be 'bm25plus' or 'bm25'")
        self.corpus = corpus if corpus is not None else load_corpus()
        self.retriever = retriever
        self.threshold = threshold
        self.model: AnswerModel = model or MockModel()
        docs = corpus_documents(self.corpus)
        self.index = BM25PlusIndex(docs) if retriever == "bm25plus" else BM25Index(docs)
        self._by_id = {i.id: i for i in self.corpus}
        self._tokens = {doc_id: set(tokenize(text)) for doc_id, text in docs}
        n = len(docs)
        df: Dict[str, int] = {}
        for toks in self._tokens.values():
            for t in toks:
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log((n + 1.0) / c) for t, c in df.items()}
        self._unknown = math.log(n + 1.0)
        self.digest = corpus_digest(self.corpus)

    def ask(self, query: str) -> AssistantAnswer:
        ranked = self.index.query(query)[:TOP_K]
        qtoks = tokenize(query)
        retrieved = []
        passages: List[Passage] = []
        for doc_id, score in ranked:
            src = self._by_id[doc_id].source
            cov = coverage_of(qtoks, self._tokens[doc_id], self._idf, self._unknown)
            retrieved.append({"id": doc_id, "cite": src.clause, "score": round(score, 6), "coverage": round(cov, 6)})
            passages.append(Passage(doc_id, src.standard, src.clause, src.section_title, src.anchor_text, score))
        top_cov = retrieved[0]["coverage"] if retrieved else 0.0
        if retrieved and top_cov >= self.threshold:
            decision, text, cites = "answer", self.model.compose(query, passages), [passages[0].cite]
        else:
            decision, text, cites = "abstain", ABSTAIN_TEXT % (top_cov, self.threshold), []
        audit = {
            "query": query,
            "retriever": self.retriever,
            "model": self.model.name,
            "llm_used": not isinstance(self.model, MockModel),
            "corpus_digest": self.digest,
            "retrieved": retrieved,
            "threshold": self.threshold,
            "top_coverage": top_cov,
            "decision": decision,
            "citations": cites,
            "answer_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        return AssistantAnswer(decision, text, cites, audit)


class AssistantAdapter(Adapter):
    """Expose the assistant through the benchmark's adapter interface (question only)."""

    arm = "grounded"

    def __init__(self, assistant: Assistant) -> None:
        self.assistant = assistant
        self.name = "assistant_%s_t%.2f" % (assistant.retriever, assistant.threshold)
        self.audit: List[Dict[str, Any]] = []

    def describe(self) -> Dict[str, Any]:
        meta = super().describe()
        meta.update({"retriever": self.assistant.retriever, "threshold": self.assistant.threshold,
                     "model": self.assistant.model.name})
        return meta

    def answer(self, item: Item) -> AdapterResponse:
        res = self.assistant.ask(item.question)
        self.audit.append(dict(res.audit, item_id=item.id))
        return AdapterResponse(
            item_id=item.id, text=res.text, citations=list(res.citations),
            confidence=res.audit["top_coverage"], metadata={"decision": res.decision},
            provenance="grounded_assistant_mock_extractive",
        )
