# Changelog

All notable changes to this project are documented here. The format follows
Keep a Changelog and this project uses semantic versioning.

## [Unreleased]

### Added
- Anthropic Messages API adapter (`grounding_eval/adapters/anthropic_api.py`), using only `urllib.request`, with ungrounded and grounded arms, bounded retry with backoff, and a `--dry-run` mode
- `python3 -m grounding_eval.cli run --adapter anthropic` for running a real system under test; run files are not labelled as mock demonstrations
- Tests for the adapter (`tests/test_anthropic_adapter.py`), all network access mocked

## [0.1.0] - 2026-09-09

### Added
- Benchmark with 68 verified items (63 factual, 5 category-error) across 8 regulatory domains measuring whether AI answers are grounded in authoritative sources
- Complete minimal pairs (23) where items differ only in the entity asked about, to expose generic topical answers that fail the distinction
- Source verification framework verifying all 68 items against primary federal sources (eCFR) with anchor-text and paragraph-location checks
- Scoring harness with deterministic lexical scoring: UNSCORABLE, ABSTAINED, CORRECT, ADJACENT_SUBSTITUTION, OTHER_INCORRECT outcomes
- Three non-LLM baselines (random floor, TF-IDF retrieval, oracle ceiling) showing items discriminate and establishing baseline performance
- **No language model has been evaluated on this benchmark.** Baseline results are non-LLM only. Mock adapter provides labelled demonstration fixture.
