# The excerpt ceiling: measure the answer space before blaming the ranker

**No language model was evaluated.** This is an exhaustive analysis of the committed lexical retrieval baselines and their finite answer space.

## Finding

Only **27/63 factual questions (42.9%)** have any scorer-passing answer among the 68 indexed single-excerpt responses. We tested all **4,284 question/excerpt combinations** under the unchanged scoring rule. Supplying each question's designated excerpt already reaches this ceiling.

| Retriever | Passes | Recoverable by reranking | No indexed excerpt passes | Retrieves own excerpt but fails | Both members pass |
|---|---:|---:|---:|---:|---:|
| retrieval_tfidf | 17/63 | 10 | 36 | 31 | 3/23 |
| retrieval_bm25 | 18/63 | 9 | 36 | 31 | 3/23 |
| retrieval_bm25plus | 17/63 | 10 | 36 | 30 | 3/23 |
| retrieval_rrf | 17/63 | 10 | 36 | 30 | 3/23 |

Reranking can recover the middle column, but cannot make the final failure class pass without changing the answer space or scorer. These are exhaustive counts for this corpus; no population inference or significance claim is needed.

## The pair constraint has its own ceiling

Even an answer-key-informed selector can make both members pass for only **7/23 pairs**. An observed 3/23 joint score therefore mixes ranking failures with an answer-space limitation. It cannot on its own establish entity confusion or a language-model failure.

## The measuring instrument matters

Removing source titles and clause metadata reduces the attainable score from 27/63 to **24/63**. The designated-excerpt passes that depend on metadata are: cs-004, prd-006, psm-007.

For `cs-004`, the anchor contains 19.5% and 23.5%, while the clause metadata supplies the phrases 'oxygen deficient' and 'oxygen enriched'. For `psm-007`, the clause identifier supplies a third required group, reaching the 75% coverage threshold while the requested contrast with incident-report retention remains absent. Metadata is part of the evaluated answer, so this is format sensitivity, not automatically an error.

A sharper counterexample is `hc-005`: its numeric-only key asks for eight pictograms, but excerpts `hn-001` (noise exposure duration) and `rk-001` (fatality reporting interval) also pass. Both contain eight. The lexical scorer does not establish what the number refers to. The ceiling is consequently a **score ceiling**, not a ceiling on factually correct answers.

## Scores do not certify an answer

The JSON contains every achievable score threshold, with counts of answered questions, score passes, complete pairs, fractured pairs, and both-pass pairs. Equal scores enter together. A pair with only one member answered is not a demonstrated distinction.

| Retriever | First threshold reaching 25% coverage | Passes / answered | Complete pairs | Both-pass pairs |
|---|---:|---:|---:|---:|
| retrieval_tfidf | 25.4% | 8/16 | 1 | 1 |
| retrieval_bm25 | 25.4% | 9/16 | 2 | 1 |
| retrieval_bm25plus | 25.4% | 7/16 | 3 | 1 |
| retrieval_rrf | 88.9% | 17/56 | 17 | 3 |

Cosine similarity may rank answers without estimating their probability of correctness. Being bounded between zero and one does not establish probability semantics. Published Brier/ECE values for raw cosine should be read only as a hypothetical score-as-probability diagnostic.

## Reproduce and inspect

```sh
python3 -m grounding_eval.answerability
python3 -m grounding_eval.answerability --check
python3 -m unittest tests.test_answerability -v
```

`results/answerability.json` contains the per-item passing excerpt IDs, designated-excerpt diagnostics, retrieved IDs, failure decomposition, pair counts, threshold curves, scoring configuration, corpus digest and SHA-256 hashes of input artifacts and scoring code. The generator writes an identical public copy to `docs/evidence-audit-data.json`; `--check` verifies both JSON files and this report. Runs are rejected if mock, duplicated, stale, incompatible with the answer format, or inconsistent with rescoring. The complete corpus is indexed; only the 63 factual items enter question-level denominators. The five category-error questions remain excluded from factual accuracy.

## Limits

- A score pass is not proof of semantic correctness; numeric-only keys can accept unrelated excerpts.
- This is an exhaustive in-sample score ceiling, not a bound for full documents, multi-excerpt synthesis, or language models.
- Lexical omissions can be genuine missing information or unrecognized wording; this analysis does not adjudicate them.
- Score thresholds are descriptive and chosen on these same items, not validated deployment policies.
- Cosine similarity is not a correctness probability merely because it is bounded to [0,1].
- The random_floor baseline reads the answer key and selects between prepared correct and adjacent-wrong answers; it is not an information-matched comparator for excerpt retrieval.
