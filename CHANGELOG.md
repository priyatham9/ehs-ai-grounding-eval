# Changelog

All notable changes to this project are documented here. The format follows
Keep a Changelog and this project uses semantic versioning.

## [Unreleased]

### Added
- Anthropic Messages API adapter (`grounding_eval/adapters/anthropic_api.py`), using only `urllib.request`, with ungrounded and grounded arms, bounded retry with backoff, and a `--dry-run` mode
- `python3 -m grounding_eval.cli run --adapter anthropic` for running a real system under test; run files are not labelled as mock demonstrations
- Tests for the adapter (`tests/test_anthropic_adapter.py`), all network access mocked
- `docs/story.html`: the "You go first" card now asks the reader to pick one of the two answers before the reveal

- `docs/story.html`: motion layer on story engine 2.0 (GSAP 3.15 from cdnjs): cinematic hero, word-mask headline reveals, the 60-second version as a pinned five-beat arc (problem, evidence, method, finding, so what) with scrubbed counters, pinned "what this does not establish" and "five questions" chapters, spring hover and press on buttons, Flip on the ranking re-order; in-page links to a pinned chapter land at its end so every number is on screen
- `docs/try.html`: GSAP motion layer: masked headline reveal, clause and trap slide in from opposite sides, verdict and round summary spring in, the marked question's dot pops, the round settings open with a Flip layout transition, and a pinned "What the baselines show" chapter where the scatter wipes in while three beats land (47.3%, 0.0%, 0 models, from results/baselines_summary.csv)
- Reduced motion (OS setting or `?reduced=1`), print and missing GSAP all show the final static state; phones get one-shot reveals instead of pins

## [0.1.0] - 2026-09-09

### Added
- Benchmark with 68 verified items (63 factual, 5 category-error) across 8 regulatory domains measuring whether AI answers are grounded in authoritative sources
- Complete minimal pairs (23) where items differ only in the entity asked about, to expose generic topical answers that fail the distinction
- Source verification framework verifying all 68 items against primary federal sources (eCFR) with anchor-text and paragraph-location checks
- Scoring harness with deterministic lexical scoring: UNSCORABLE, ABSTAINED, CORRECT, ADJACENT_SUBSTITUTION, OTHER_INCORRECT outcomes
- Three non-LLM baselines (random floor, TF-IDF retrieval, oracle ceiling) showing items discriminate and establishing baseline performance
- **No language model has been evaluated on this benchmark.** Baseline results are non-LLM only. Mock adapter provides labelled demonstration fixture.
