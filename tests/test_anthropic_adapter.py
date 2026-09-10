"""Tests for the Anthropic Messages API adapter.

All network access is mocked (``urllib.request.urlopen``). Nothing here ever
calls the real API. The tests cover: request shape, response parsing,
abstain parsing, retry-then-success on 429, failure after exhausting
retries, the missing-API-key error, that the grounded prompt carries
retrieved source text, that the adapter never touches the answer key, and
that --dry-run makes no network call.
"""

from __future__ import annotations

import io
import json
import unittest
import urllib.error
from unittest import mock

from grounding_eval.adapters.anthropic_api import (
    AnthropicAdapter,
    AnthropicAPIError,
    _backoff_delay,
    _parse_response_text,
    estimate_input_tokens,
)
from grounding_eval.corpus import load_corpus
from grounding_eval.schema import Item


def _message_response(text: str, model: str = "claude-opus-5") -> bytes:
    payload = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 42, "output_tokens": 7},
    }
    return json.dumps(payload).encode("utf-8")


class _FakeHTTPResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _ok_response(text: str):
    return _FakeHTTPResponse(_message_response(text))


class TestRequestShape(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = load_corpus()
        self.item = next(iter(self.corpus))

    def test_body_and_headers(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _ok_response("ANSWER: foo\nCITATION: none")

        adapter = AnthropicAdapter(api_key="sk-test-key", model="claude-opus-5")
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            adapter.answer(self.item)

        self.assertEqual(captured["url"], "https://api.anthropic.com/v1/messages")
        self.assertEqual(captured["headers"]["x-api-key"], "sk-test-key")
        self.assertEqual(captured["headers"]["anthropic-version"], "2023-06-01")
        self.assertNotIn("sk-test-key", json.dumps(captured["headers"]).replace("sk-test-key", ""))

        body = captured["body"]
        self.assertEqual(body["model"], "claude-opus-5")
        self.assertEqual(body["temperature"], 0)
        self.assertIn("system", body)
        self.assertEqual(len(body["messages"]), 1)
        self.assertEqual(body["messages"][0]["role"], "user")
        self.assertIn(self.item.question, body["messages"][0]["content"])

    def test_api_key_never_logged_in_describe(self) -> None:
        adapter = AnthropicAdapter(api_key="sk-super-secret")
        described = json.dumps(adapter.describe())
        self.assertNotIn("sk-super-secret", described)

    def test_describe_fields(self) -> None:
        adapter = AnthropicAdapter(api_key="sk-test-key", model="claude-opus-5", arm="ungrounded")
        meta = adapter.describe()
        self.assertEqual(meta["model"], "claude-opus-5")
        self.assertEqual(meta["arm"], "ungrounded")
        self.assertEqual(meta["temperature"], 0.0)
        self.assertIn("system_prompt_sha256_16", meta)
        self.assertIn("anthropic_version_header", meta)
        self.assertIn("timestamp_utc", meta)


class TestResponseParsing(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = load_corpus()
        self.item = next(iter(self.corpus))

    def _answer_with(self, text: str):
        adapter = AnthropicAdapter(api_key="sk-test-key")
        with mock.patch("urllib.request.urlopen", return_value=_ok_response(text)):
            return adapter.answer(self.item)

    def test_parses_answer_and_citation(self) -> None:
        response = self._answer_with(
            "ANSWER: The employer must monitor atmosphere continuously.\n"
            "CITATION: 29 CFR 1910.146(c)(5)(ii)"
        )
        self.assertEqual(response.item_id, self.item.id)
        self.assertIn("monitor atmosphere continuously", response.text)
        self.assertEqual(response.citations, ["29 CFR 1910.146(c)(5)(ii)"])
        self.assertEqual(response.provenance, "anthropic_messages_api")
        self.assertIsNone(response.error)

    def test_citation_none_yields_no_citations(self) -> None:
        response = self._answer_with("ANSWER: Something.\nCITATION: none")
        self.assertEqual(response.citations, [])

    def test_abstain_parsing(self) -> None:
        response = self._answer_with("ANSWER: I don't know\nCITATION: none")
        self.assertEqual(response.metadata["parsed_abstain"], True)
        self.assertIn("i don't know", response.text.lower())

    def test_malformed_reply_falls_back_to_raw_text(self) -> None:
        response = self._answer_with("This is not in the requested format at all.")
        self.assertIn("not in the requested format", response.text)

    def test_metadata_records_usage(self) -> None:
        response = self._answer_with("ANSWER: x\nCITATION: none")
        self.assertEqual(response.metadata["input_tokens"], 42)
        self.assertEqual(response.metadata["output_tokens"], 7)


class TestRetryBehaviour(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = load_corpus()
        self.item = next(iter(self.corpus))

    def test_retries_on_429_then_succeeds(self) -> None:
        calls = {"n": 0}

        def fake_urlopen(request, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.HTTPError(
                    "url", 429, "rate limited", {"Retry-After": "0"}, io.BytesIO(b"{}")
                )
            return _ok_response("ANSWER: ok\nCITATION: none")

        adapter = AnthropicAdapter(api_key="sk-test-key")
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen), mock.patch(
            "time.sleep", return_value=None
        ):
            response = adapter.answer(self.item)

        self.assertEqual(calls["n"], 2)
        self.assertIn("ok", response.text)

    def test_fails_after_max_retries(self) -> None:
        def fake_urlopen(request, timeout=None):
            raise urllib.error.HTTPError("url", 503, "unavailable", {}, io.BytesIO(b"{}"))

        adapter = AnthropicAdapter(api_key="sk-test-key")
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen), mock.patch(
            "time.sleep", return_value=None
        ):
            with self.assertRaises(AnthropicAPIError):
                adapter._call_api(self.item)

    def test_400_is_not_retried(self) -> None:
        calls = {"n": 0}

        def fake_urlopen(request, timeout=None):
            calls["n"] += 1
            raise urllib.error.HTTPError("url", 400, "bad request", {}, io.BytesIO(b"{}"))

        adapter = AnthropicAdapter(api_key="sk-test-key")
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(AnthropicAPIError):
                adapter._call_api(self.item)
        self.assertEqual(calls["n"], 1)

    def test_backoff_honours_retry_after(self) -> None:
        self.assertEqual(_backoff_delay(0, "5"), 5.0)

    def test_backoff_is_bounded_and_exponential(self) -> None:
        self.assertLess(_backoff_delay(10, None), 40.0)
        self.assertGreaterEqual(_backoff_delay(1, None), _backoff_delay(0, None))


class TestMissingApiKey(unittest.TestCase):
    def test_missing_key_raises_clear_error_only_at_use_time(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            # Construction must not raise.
            adapter = AnthropicAdapter()
            with self.assertRaises(AnthropicAPIError):
                adapter._resolve_api_key()

    def test_explicit_key_overrides_environment(self) -> None:
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            adapter = AnthropicAdapter(api_key="explicit-key")
            self.assertEqual(adapter._resolve_api_key(), "explicit-key")

    def test_environment_key_used_when_no_explicit_key(self) -> None:
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            adapter = AnthropicAdapter()
            self.assertEqual(adapter._resolve_api_key(), "env-key")


class TestGroundedArm(unittest.TestCase):
    def test_grounded_prompt_contains_retrieved_source_text(self) -> None:
        corpus = load_corpus()
        adapter = AnthropicAdapter(api_key="sk-test-key", arm="grounded", corpus=corpus)
        item = next(iter(corpus))
        prompt = adapter.build_prompt(item)
        self.assertIn(item.source.anchor_text, prompt)
        self.assertIn(item.source.clause, prompt)
        self.assertIn(item.question, prompt)

    def test_ungrounded_prompt_has_no_source_excerpt(self) -> None:
        corpus = load_corpus()
        adapter = AnthropicAdapter(api_key="sk-test-key", arm="ungrounded")
        item = next(iter(corpus))
        prompt = adapter.build_prompt(item)
        self.assertNotIn(item.source.anchor_text, prompt)

    def test_invalid_arm_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AnthropicAdapter(api_key="sk-test-key", arm="sideways")


class _KeyAccessSentinel:
    """An Item stand-in whose answer-key-shaped attributes raise if touched.

    Mirrors the sentinel pattern used to enforce the "adapter must not see the
    answer key" rule described in adapters/base.py: the real Item dataclass is
    wrapped so that reading anything beyond question/id/source explodes.
    """

    def __init__(self, real_item: Item) -> None:
        object.__setattr__(self, "_real", real_item)

    def __getattr__(self, attr_name: str):
        if attr_name in ("correct_answer", "answer_key", "adjacent_wrong"):
            raise AssertionError(
                "adapter accessed forbidden field %r on the item" % attr_name
            )
        return getattr(object.__getattribute__(self, "_real"), attr_name)


class TestNeverReadsAnswerKey(unittest.TestCase):
    def test_ungrounded_answer_never_touches_answer_key_fields(self) -> None:
        corpus = load_corpus()
        real_item = next(iter(corpus))
        sentinel_item = _KeyAccessSentinel(real_item)

        adapter = AnthropicAdapter(api_key="sk-test-key", arm="ungrounded")
        with mock.patch(
            "urllib.request.urlopen", return_value=_ok_response("ANSWER: x\nCITATION: none")
        ):
            response = adapter.answer(sentinel_item)  # type: ignore[arg-type]
        self.assertEqual(response.item_id, real_item.id)

    def test_grounded_answer_never_touches_answer_key_fields(self) -> None:
        corpus = load_corpus()
        real_item = next(iter(corpus))
        sentinel_item = _KeyAccessSentinel(real_item)

        adapter = AnthropicAdapter(api_key="sk-test-key", arm="grounded", corpus=corpus)
        with mock.patch(
            "urllib.request.urlopen", return_value=_ok_response("ANSWER: x\nCITATION: none")
        ):
            response = adapter.answer(sentinel_item)  # type: ignore[arg-type]
        self.assertEqual(response.item_id, real_item.id)

    def test_build_prompt_never_touches_answer_key_fields(self) -> None:
        corpus = load_corpus()
        real_item = next(iter(corpus))
        sentinel_item = _KeyAccessSentinel(real_item)
        adapter = AnthropicAdapter(api_key="sk-test-key", arm="grounded", corpus=corpus)
        adapter.build_prompt(sentinel_item)  # type: ignore[arg-type]


class TestParsingHelper(unittest.TestCase):
    def test_parse_response_text_handles_missing_citation_line(self) -> None:
        parsed = _parse_response_text("ANSWER: something\n")
        self.assertEqual(parsed.text, "something")
        self.assertIsNone(parsed.citation)

    def test_estimate_input_tokens_is_chars_over_four(self) -> None:
        self.assertEqual(estimate_input_tokens("abcd"), 1)
        self.assertEqual(estimate_input_tokens("a" * 40), 10)


class TestDryRunMakesNoNetworkCall(unittest.TestCase):
    def test_dry_run_cli_makes_no_network_calls(self) -> None:
        from grounding_eval import cli

        with mock.patch("urllib.request.urlopen") as fake_urlopen:
            exit_code = cli.main(["run", "--adapter", "anthropic", "--dry-run"])
        self.assertEqual(exit_code, 0)
        fake_urlopen.assert_not_called()

    def test_dry_run_grounded_cli_makes_no_network_calls(self) -> None:
        from grounding_eval import cli

        with mock.patch("urllib.request.urlopen") as fake_urlopen:
            exit_code = cli.main(["run", "--adapter", "anthropic", "--grounded", "--dry-run"])
        self.assertEqual(exit_code, 0)
        fake_urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
