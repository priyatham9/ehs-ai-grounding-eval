"""Adapter for the real Anthropic Messages API.

This is the adapter that lets the benchmark run against an actual language
model instead of a fixture. Everything else in ``adapters/`` is either a
demonstration (``mock``), a hand-built baseline (``random_floor``,
``retrieval``), or a ceiling (``oracle``); this one calls out to
``api.anthropic.com`` and scores whatever comes back.

Only the standard library is used (``urllib.request``), matching the rest of
this repository's no-third-party-dependencies policy.

Grounding rule
--------------
Per ``adapters/base.py`` and ``docs/methodology.md`` section 7, an adapter
representing a real system must use only ``item.question`` and, at most,
``item.id``. This adapter does exactly that: it never reads
``item.correct_answer``, ``item.answer_key``, or ``item.adjacent_wrong``. In
``grounded`` mode it also consults a retrieval index over the corpus's own
source metadata (the same one ``RetrievalAdapter`` builds), which is
authoritative source text, not the answer key.

Two arms
--------
``ungrounded`` (default)
    The question is asked with no supporting context. This is the arm the
    benchmark exists to be suspicious of: an answer that is right here is
    right because the model's parametric memory happens to be right, not
    because anything was checked.

``grounded``
    The best-matching excerpt from ``RetrievalAdapter``'s TF-IDF index is
    placed in the prompt, and the model is instructed to answer only from it
    and to say it does not know if the excerpt does not answer the question.
    This measures the model's behaviour when given a source, not the
    retriever's own accuracy (that is what ``retrieval_tfidf`` measures on its
    own).

Response format
----------------
The system prompt asks for two labelled lines so the scorer's phrase-based
abstention detector (``scoring/abstention.py``) and regex-based citation
parser (``scoring/citations.py``) both have clean text to work with:

    ANSWER: <answer, or "I don't know">
    CITATION: <clause citation, or "none">

Both scorers work on free text regardless, but the fixed format keeps the
model from burying the citation inside a longer paragraph where the citation
regex might still find it, but by accident.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..corpus import Corpus
from ..schema import AdapterResponse, Item
from .base import Adapter
from .retrieval import RetrievalAdapter

#: Current best generally available Claude model, per the claude-api skill
#: (cached 2026-06-24). Overridable via the ``model`` constructor argument or
#: the CLI's ``--model`` flag.
DEFAULT_MODEL = "claude-opus-5"

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

MAX_RETRIES = 5
INITIAL_BACKOFF_S = 1.0
MAX_BACKOFF_S = 32.0
REQUEST_TIMEOUT_S = 60.0

SYSTEM_PROMPT = (
    "You are being asked a single safety-critical regulatory or technical "
    "compliance question. Answer only if you are confident; if you are not "
    "confident, say you do not know rather than guessing.\n\n"
    "Respond in exactly this two-line format and nothing else:\n"
    "ANSWER: <your answer in one or two sentences, or \"I don't know\">\n"
    "CITATION: <the specific clause or section you are relying on (e.g. "
    "\"29 CFR 1910.147(c)(6)\"), or \"none\" if you are not citing one>"
)

GROUNDED_SYSTEM_PROMPT = (
    "You are being asked a single safety-critical regulatory or technical "
    "compliance question. You are given an excerpt from an authoritative "
    "source. Answer using only the information in that excerpt. If the "
    "excerpt does not answer the question, say you do not know rather than "
    "guessing or using outside knowledge.\n\n"
    "Respond in exactly this two-line format and nothing else:\n"
    "ANSWER: <your answer in one or two sentences, or \"I don't know\">\n"
    "CITATION: <the specific clause or section from the excerpt you relied "
    "on, or \"none\" if the excerpt did not answer the question>"
)

_ANSWER_RE = re.compile(r"answer\s*:\s*(.*?)(?:\n|$)", re.IGNORECASE | re.DOTALL)
_CITATION_RE = re.compile(r"citation\s*:\s*(.*?)(?:\n|$)", re.IGNORECASE | re.DOTALL)


class AnthropicAPIError(RuntimeError):
    """Raised when the Anthropic API cannot be reached or fails after retries."""


def _missing_key_error() -> AnthropicAPIError:
    return AnthropicAPIError(
        "ANTHROPIC_API_KEY is not set and no api_key was provided. Set the "
        "environment variable or pass api_key= explicitly to AnthropicAdapter."
    )


def _prompt_hash(system: str) -> str:
    """A short, non-reversible fingerprint of the fixed system prompt.

    Recorded in describe() so a run file can be checked against the prompt
    that produced it without embedding the prompt text (which would bloat
    every response's metadata) or the API key (which must never be logged).
    """
    return hashlib.sha256(system.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class _ParsedAnswer:
    text: str
    citation: Optional[str]
    abstained: bool


def _parse_response_text(raw_text: str) -> _ParsedAnswer:
    """Pull the ANSWER and CITATION lines out of the model's reply.

    Falls back to treating the whole reply as the answer if the model did not
    follow the requested format - the phrase-based abstention detector and
    the citation regex both still work on unstructured text, so a malformed
    reply degrades gracefully rather than becoming unscorable.
    """
    raw_text = raw_text or ""
    answer_match = _ANSWER_RE.search(raw_text)
    citation_match = _CITATION_RE.search(raw_text)

    answer = answer_match.group(1).strip() if answer_match else raw_text.strip()
    citation = citation_match.group(1).strip() if citation_match else None
    if citation and citation.lower() in ("none", "n/a", "none.", "n/a."):
        citation = None

    abstained = bool(answer) and answer.strip().lower().rstrip(".") in (
        "i don't know",
        "i do not know",
        "unknown",
    )
    return _ParsedAnswer(text=answer, citation=citation, abstained=abstained)


class AnthropicAdapter(Adapter):
    """Calls the real Anthropic Messages API for each item.

    Parameters
    ----------
    model:
        Anthropic model id. Defaults to :data:`DEFAULT_MODEL`.
    api_key:
        Explicit API key. Defaults to ``os.environ["ANTHROPIC_API_KEY"]``,
        read lazily on first use (not at construction time), so an adapter
        can be built in a process that only sets the key later, and so that
        ``--dry-run`` never needs a key at all.
    arm:
        ``"ungrounded"`` (default) or ``"grounded"``.
    corpus:
        Only used when ``arm == "grounded"``. Supplies the retrieval index;
        reuses :class:`~grounding_eval.adapters.retrieval.RetrievalAdapter`
        rather than re-implementing TF-IDF retrieval.
    temperature, max_tokens:
        Defaults are 0 (the benchmark scores single most-likely answers, not
        sampled variety) and a modest 512 (answers are one or two sentences
        plus a citation). Claude Opus 5, Sonnet 5 and the 4.7 and later
        families reject sampling parameters with a 400, so ``temperature`` is
        only sent to models that still accept it (Haiku 4.5, the 4.6 and 4.5
        families, and Claude 3.x). ``describe()`` records whether it was sent.
    """

    name = "anthropic_api"

    @staticmethod
    def supports_sampling(model: str) -> bool:
        """Whether the Messages API accepts ``temperature`` for this model."""
        m = model.lower()
        return (
            "haiku" in m
            or "-4-6" in m
            or "-4-5" in m
            or m.startswith("claude-3")
        )

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        arm: str = "ungrounded",
        corpus: Optional[Corpus] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        if arm not in ("ungrounded", "grounded"):
            raise ValueError("arm must be 'ungrounded' or 'grounded', got %r" % (arm,))
        self.model = model
        self._api_key = api_key
        self.arm = arm
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.name = "anthropic_api:%s" % arm

        self._retriever: Optional[RetrievalAdapter] = None
        if arm == "grounded":
            self._retriever = RetrievalAdapter(corpus=corpus)

        self._system_prompt = GROUNDED_SYSTEM_PROMPT if arm == "grounded" else SYSTEM_PROMPT

    # -- credentials -----------------------------------------------------

    def _resolve_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise _missing_key_error()
        return key

    # -- prompt construction ----------------------------------------------

    def _retrieved_excerpt(self, item: Item) -> Optional[str]:
        assert self._retriever is not None
        ranked = self._retriever.index.query(item.question)
        if not ranked:
            return None
        best_id, _score = ranked[0]
        src = self._retriever._by_id[best_id].source  # authoritative source text, not the answer key
        return "%s, %s (%s): %s" % (src.standard, src.clause, src.section_title, src.anchor_text)

    def build_prompt(self, item: Item) -> str:
        """Build the user-turn text for one item. Uses only item.question (and,
        in grounded mode, retrieved source excerpts) - never the answer key."""
        if self.arm == "grounded":
            excerpt = self._retrieved_excerpt(item)
            if excerpt:
                return (
                    "Authoritative source excerpt:\n%s\n\nQuestion: %s"
                    % (excerpt, item.question)
                )
        return "Question: %s" % item.question

    # -- describe ----------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        meta = super().describe()
        meta.update(
            {
                "model": self.model,
                "arm": self.arm,
                "temperature": self.temperature,
                "temperature_sent": self.supports_sampling(self.model),
                "max_tokens": self.max_tokens,
                "system_prompt_sha256_16": _prompt_hash(self._system_prompt),
                "anthropic_version_header": ANTHROPIC_VERSION,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        # Deliberately never include the API key here or anywhere else.
        return meta

    # -- request/response ----------------------------------------------------

    def _request_body(self, item: Item) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self._system_prompt,
            "messages": [{"role": "user", "content": self.build_prompt(item)}],
        }
        if self.supports_sampling(self.model):
            body["temperature"] = self.temperature
        return body

    def _headers(self, api_key: str) -> Dict[str, str]:
        return {
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }

    def _call_api(self, item: Item) -> Dict[str, Any]:
        api_key = self._resolve_api_key()
        body = json.dumps(self._request_body(item)).encode("utf-8")

        last_error: Optional[BaseException] = None
        for attempt in range(MAX_RETRIES):
            request = urllib.request.Request(
                ANTHROPIC_API_URL, data=body, headers=self._headers(api_key), method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                payload = exc.read()
                if exc.code == 429 or exc.code >= 500:
                    last_error = exc
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    delay = _backoff_delay(attempt, retry_after)
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(delay)
                        continue
                    raise AnthropicAPIError(
                        "Anthropic API failed after %d attempts (HTTP %d): %s"
                        % (MAX_RETRIES, exc.code, payload[:500])
                    ) from exc
                raise AnthropicAPIError(
                    "Anthropic API returned HTTP %d: %s" % (exc.code, payload[:500])
                ) from exc
            except urllib.error.URLError as exc:
                last_error = exc
                delay = _backoff_delay(attempt, None)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(delay)
                    continue
                raise AnthropicAPIError(
                    "Anthropic API unreachable after %d attempts: %s" % (MAX_RETRIES, exc)
                ) from exc
        # Unreachable in practice; satisfies static analysis.
        raise AnthropicAPIError("Anthropic API failed: %s" % last_error)

    def answer(self, item: Item) -> AdapterResponse:
        payload = self._call_api(item)
        content_blocks = payload.get("content") or []
        raw_text = "".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        )
        parsed = _parse_response_text(raw_text)

        citations = [parsed.citation] if parsed.citation else []
        usage = payload.get("usage") or {}

        return AdapterResponse(
            item_id=item.id,
            text=parsed.text,
            citations=citations,
            provenance="anthropic_messages_api",
            metadata={
                "model": payload.get("model", self.model),
                "stop_reason": payload.get("stop_reason"),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "raw_text": raw_text,
                "parsed_abstain": parsed.abstained,
            },
        )


def _backoff_delay(attempt: int, retry_after_header: Optional[str]) -> float:
    """Bounded exponential backoff, honouring Retry-After when the server sends one."""
    if retry_after_header:
        try:
            return min(float(retry_after_header), MAX_BACKOFF_S)
        except ValueError:
            pass
    return min(INITIAL_BACKOFF_S * (2 ** attempt), MAX_BACKOFF_S)


def estimate_input_tokens(text: str) -> int:
    """Crude chars/4 estimate, used only by --dry-run. Never used for billing."""
    return max(1, len(text) // 4)
