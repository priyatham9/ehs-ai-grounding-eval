# Design: blind paraphrase audit of answer keys

Status: draft v2, revised after two fresh-context (goldfish) reviews; see "Review record". Method: Elephant-Goldfish Model (EGM),
Apache-2.0, https://github.com/google/elephant-goldfish-model (workflow adapted
by hand; the EGM plugin itself is not vendored here).

## Problem

`docs/limitations.md` section 1 states that the scorer is lexical, so a correct
answer that paraphrases around every listed alternative is scored as a miss, and
that "the size of the gap is unknown". The existing invariant (every reference
answer satisfies its own key) cannot detect keys that are merely narrow, because
the reference answer was written by the same author as the key.

## Goals

1. Produce a measured estimate of the scorer's false-negative rate on correct
   answers, per item and overall, with a confidence interval.
2. Make the audit reproducible: a script regenerates every number from committed
   files.
3. Flag narrow keys (items where correct paraphrases fail) for author review.

## Non-goals

- No language-model system is evaluated; this does not produce benchmark scores.
- No change to the scorer in this design. Widening keys is a separate change,
  made only after the audit reports, and must not use embeddings (see
  limitations section 1, "Why it was accepted").
- Not a claim of correctness for any answer; the audit tests only scorer recall.

## Design

**Scope:** the 63 factual items. The 5 `category_error` items expect premise
rejection (a separate abstention scoring path) and are out of scope. Unit is the
item, using `Item.source.anchor_text` (present on all 68 items) and `Item.question`;
`cross_reference` entries and per-clause splitting are out of scope.

**Inputs to the paraphraser:** a generated file holding only `item_id`, question
stem and `anchor_text`. The paraphraser has no repo access, so keys and reference
answers are withheld (the goldfish condition). The scorer result is never shown to
the paraphraser or the checker.

**Procedure:**
1. Paraphraser (human or fresh-context agent) writes K=3 correct answers of 1-3
   sentences, faithful to the clause, avoiding its wording, and keeping every
   number and unit. The exact prompt is committed verbatim (`corpus/audit/PROMPT.md`).
   A second arm ("verbose") allows restating the clause freely, to bracket the rate
   because real system outputs are longer.
2. A fresh-context checker, seeing only clause and paraphrase, labels it `correct`
   or `incorrect` (numeric drift is `incorrect`). Rejected rows are kept with a
   reason and the rejection rate is reported per item. Regeneration is capped at
   one retry and is never conditioned on the scorer outcome.
3. Each accepted paraphrase is scored with `score_item` and default `ScoringConfig()`
   (recorded in the report via `describe()`). **hit = `Outcome.CORRECT`**; every
   other outcome (including ABSTAINED and UNSCORABLE) is a miss and is counted.
4. Each miss records which gate failed: `concept_coverage`, `numeric_coverage` or
   `forbidden_asserted`, plus the uncovered required concept groups. Only a
   concept-coverage miss implies "widen the key"; numeric and forbidden-misfire
   misses are reported separately and not conflated.
5. Headline: overall miss rate with `cluster_bootstrap_ci` (cluster = item, default
   seed). Wilson on paraphrases is shown only as an orientation figure because the
   K paraphrases per item are not independent. Per item: the count misses/K with no
   interval. Stratify by risk tier, family and numeric-vs-qualitative key.
6. Flag rule (a review prompt for authors, not a verdict): an item is flagged if at
   least 1 of its accepted paraphrases is a concept-coverage miss; the report lists
   the failing gate and uncovered groups.

**Files (modelled on `scripts/item_analysis.py`):** `corpus/audit/paraphrases.jsonl`
(`item_id, k_index, arm, text, producer_kind, producer_id, model_id, prompt_sha256,
GENERATED_BY_AGENT, checker_label, checker_id, rejected_reason, run_date`),
`scripts/key_narrowness_audit.py` (`analyse()`, `render_csv()`, `render_md()`,
`--check`), `results/key_narrowness_audit.{md,csv}`, `tests/test_key_narrowness_audit.py`
with an inline fixture. Add the script to the `stdlib_ok` allowlist in
`tests/test_integrity.py` if needed; report text avoids em dashes (existing check);
run `test_integrity.py` and `test_corpus.py` after adding files.

**Reproducibility:** the script is deterministic over the committed paraphrases.
Generating the paraphrases is not reproducible (it needs an LLM or a human); model
id, prompt hash and date are recorded per row. Agent-written paraphrases are never
presented as human answers or as any vendor system's output.

## Risks

- Same model family filling both roles shares blind spots: report producer kind;
  human adjudication of a fixed sample and of high-rejection items.
- A low miss rate does not show the scorer is valid on real system outputs; the
  limitations pointer must say the figure applies to clause paraphrases only and
  that the gap varies by system.
- The forbidden/contrast heuristic (limitations section 2) can cause misses on
  correct text; separated by gate (step 4).

## Implementation plan

1. Script, fixture and tests (offline, no model calls).
2. Commit prompt and generate paraphrases with provenance.
3. Generate results; add a caveated pointer in `docs/limitations.md` section 1.

## Review record

Two fresh-context reviewers (EGM `design_critic` and `implementation_readiness`
personas, no author context) returned 24 findings. Accepted and folded into this
version: undefined hit/miss (3 blockers across both), clustered interval instead of
Wilson, scope to 63 factual items, gate-separated miss reasons, numeric-preservation
in the prompt, verbose arm, enforced blindness, rejection and retry rules, flag rule,
stratification, schema and test-suite constraints, reproducibility wording, closed
the multi-clause question. Left for implementation: K=3 versus K=5 for high-risk
tiers (decide after pilot variance is seen).
