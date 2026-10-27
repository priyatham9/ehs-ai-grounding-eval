# Grounded assistant: design note

A citation-first safety Q&A demo built on this repository's corpus, retrievers and scoring, in the pattern of governed-retrieval assistants: retrieve from a controlled corpus, answer only with cited passages, abstain when the evidence is weak, and gate release on a benchmark.

**No language model is used anywhere.** The answer composer, `MockModel`, is extractive and deterministic: it quotes the top retrieved passage verbatim and attaches its clause. It needs no keys or network. It demonstrates the control flow and the evaluation gate, not the quality of any generative system.

## Components

| piece | where |
|---|---|
| Retrieval (BM25+ default, Okapi BM25 selectable; same indexes as the baselines) | `grounding_eval/assistant/assistant.py` |
| Answer composer (`MockModel`) and the `AnswerModel` seam | `grounding_eval/assistant/model.py` |
| Benchmark gate and metrics | `scripts/run_assistant_eval.py` -> `results/assistant_eval.{csv,md}` |
| Per-answer audit log for the benchmark run | `results/assistant_audit.jsonl` |
| In-browser demo (BM25+ in JavaScript, corpus embedded, no external requests) | `docs/assistant.html`, rebuilt by `scripts/build_assistant_page.py` |

Try it: `python3 -c "from grounding_eval.assistant import Assistant; r = Assistant().ask('...'); print(r.text, r.audit)"`.

## Behaviour

1. Rank the 68 corpus passages (standard, clause, section title, verified excerpt) with BM25+ (k1 1.2, b 0.75, delta 1.0).
2. Compute **evidence coverage** of the top passage: the IDF-weighted share of the question's distinct tokens that the passage contains (IDF = ln((N+1)/df); a token absent from the corpus counts at the maximum weight, so off-corpus questions score low).
3. If coverage is at least the **threshold, 0.30**, answer: quote the passage, cite its clause. Otherwise abstain with a fixed message and cite nothing.
4. Emit an audit record: query, retriever, model, `llm_used`, corpus digest, top-3 retrieved ids with clause, BM25+ score and coverage, threshold, decision, citations and the SHA-256 of the answer text. No timestamps, so records are reproducible.

The 0.30 threshold was fixed after looking at two example queries (an on-topic question whose own passage scored 0.44, an off-topic one that scored 0.10) and before the benchmark was run through the assistant; it was not changed afterwards. The evaluation reports a sweep over other thresholds for transparency and does not pick from it.

## Evaluation and gate

`results/assistant_eval.md` gives, at the default threshold: accuracy 22.2% on 63 factual items (abstention counts as not correct), factual abstention 34.9%, all 5 category-error items declined, accuracy when answered 34.1% (41 answered), citation precision on answered items 78.0% (correct clause), 0 adjacent substitutions. The ungated BM25+ baseline is 27.0% accuracy and 74.6% citation precision.

The release gate (zero adjacent substitutions, zero category-error items answered, citation precision on answered at least 0.80, fixed before the first run) **fails** on the third criterion at 0.780. The artefacts are committed as they came out; `python3 scripts/run_assistant_eval.py --gate` exits non-zero. Lowering the abstention rate or raising precision is possible by moving the threshold (see the sweep) but at 63 items such a move would be tuning on the test set, so the default stays.

Metrics are computed by the repository's own scorer (`grounding_eval.scoring`) through the standard adapter interface, so the assistant is scored by the same code path as every other system.

## Governance: what it will not answer, and why

Implemented, via the abstain rule:

* **Questions the corpus does not support** (off-topic, false-premise, or about standards not indexed). A fluent answer without evidence is the failure this benchmark exists to measure. All 5 false-premise items were declined.
* **Questions with weak lexical evidence.** Declining is cheaper than a confident wrong answer, and the repository's safety utility policy encodes that ordering.

Not implemented, and therefore not claimed (a deployment would need them):

* No policy filter for questions that ask for a compliance determination, a legal opinion or a go/no-go for a specific plant. The abstain rule only reacts to evidence, so an on-topic question of that kind would be answered with a quote.
* No user authentication, access control per document, or retention policy for the audit log.
* No handling of superseded or amended regulations beyond the corpus edition date; the corpus is a snapshot (see `docs/CORPUS_CARD.md`).

Every answer cites exactly one clause (the quoted passage) and the audit record lets a reviewer reconstruct why it was given or withheld.

## Limitations

* The corpus is 68 short excerpts. The governing passage of each benchmark question is in the index (as in the repository's baselines), so the benchmark checks selection of a known passage, not retrieval from an unseen corpus; it overstates performance on genuinely new questions.
* Accuracy is low (22.2% gated, 27.0% ungated) because the quoted excerpt often does not contain all the concepts in the answer key, and lexical retrieval often picks a neighbouring passage. Abstention helps precision when answered (34.1% vs 27.0%), not recall.
* Coverage is lexical. A question with the right words in the wrong sense (the rupture disk / relief valve "disc" problem described in the README) can still pass the threshold. The 0 adjacent substitutions here is a count on 68 items, not a guarantee.
* Citation precision is measured against this repository's gold clause per item; it says the cited clause is the right one, not that the quote answers the question.
* The browser demo shares the algorithm with the Python package; scores agreed to 1e-6 on all 68 benchmark questions when checked by hand, but there is no automated JS test.

## Plugging in a real LLM

`AnswerModel` is a one-method protocol, `compose(query, passages) -> str`. A real adapter would receive the top-k `Passage` objects (id, standard, clause, title, text, score), prompt the model to answer using only those passages and to cite their clauses, and return the text. The pieces that must stay outside the model, so the guarantees do not depend on it:

1. The abstain decision stays in `Assistant.ask`, made from retrieval evidence before the model is called.
2. The citation list is taken from the retrieved passages, not parsed from model output; a post-check should reject any answer that names a clause not in the retrieved set, or that contains numbers absent from the passages, and fall back to abstention.
3. The audit record already carries `model` and `llm_used`; add the model id, prompt hash and sampling settings.
4. Re-run `scripts/run_assistant_eval.py` with the adapter; the same gate applies. The repository's `anthropic_api` adapter shows the keyed, networked path; it should stay opt-in and out of CI.

Nothing in this repository has evaluated a language model, and swapping one in would change every number reported here.
