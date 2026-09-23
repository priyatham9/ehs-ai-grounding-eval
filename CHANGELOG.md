# Changelog

All notable changes to this project are documented here. The format follows
Keep a Changelog and this project uses semantic versioning.

## [Unreleased]

### Added
- Anthropic Messages API adapter (`grounding_eval/adapters/anthropic_api.py`), using only `urllib.request`, with ungrounded and grounded arms, bounded retry with backoff, and a `--dry-run` mode
- `python3 -m grounding_eval.cli run --adapter anthropic` for running a real system under test; run files are not labelled as mock demonstrations
- Tests for the adapter (`tests/test_anthropic_adapter.py`), all network access mocked
- `docs/try.html`: pair verdict after answering both twins of a minimal pair, a running "pairs both right" count (the README's primary outcome), an answered-progress meter and a reset control
- `docs/story.html`: the "You go first" card now asks the reader to pick one of the two answers before the reveal

### Fixed
- `docs/try.html`: BM25 was missing from the comparison table; replayed wrong answers now show which option was picked; navigating items no longer floods browser history; back/forward between items works; keyboard shortcuts no longer fire with modifier keys or inside the site menu
- `docs/baselines.svg`: TF-IDF and BM25 labels printed on top of each other; `tools/plot_baselines.py` now stacks labels that would collide

## [0.1.0] - 2026-09-09

### Added
- Benchmark with 68 verified items (63 factual, 5 category-error) across 8 regulatory domains measuring whether AI answers are grounded in authoritative sources
- Complete minimal pairs (23) where items differ only in the entity asked about, to expose generic topical answers that fail the distinction
- Source verification framework verifying all 68 items against primary federal sources (eCFR) with anchor-text and paragraph-location checks
- Scoring harness with deterministic lexical scoring: UNSCORABLE, ABSTAINED, CORRECT, ADJACENT_SUBSTITUTION, OTHER_INCORRECT outcomes
- Three non-LLM baselines (random floor, TF-IDF retrieval, oracle ceiling) showing items discriminate and establishing baseline performance
- **No language model has been evaluated on this benchmark.** Baseline results are non-LLM only. Mock adapter provides labelled demonstration fixture.
