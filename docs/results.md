# Results: non-LLM baselines

**No language model has been evaluated on this benchmark yet.** No LLM API was
available on the machine these runs were made on, and no LLM arm is reported
anywhere in this repository. The six arms below are non-LLM baselines. They
exist to show that the 68 items discriminate, that is, that the scoring
procedure separates a system that knows nothing from one that knows everything,
and to show where a plain lexical retriever lands between them.

All numbers are produced by `python3 -m grounding_eval.run_baselines` and
written to `results/`. The table below is `results/baselines_table.md`
verbatim; the full per-item responses and scores are in
`results/baselines_<adapter>.json` and the summary row set is
`results/baselines_summary.csv`.

## Arms

* **random_floor** (fixture). For each item, picks uniformly between the two
  answer options the item carries (the reference answer and a sentence built
  from the adjacent-wrong entity's signature terms), and cites a clause drawn
  uniformly from the pool of clauses cited across the corpus. Run with 5 seeds.
  Its accuracy is about one half by construction; what it calibrates is the
  adjacent-substitution rate, the citation rate a guesser achieves, and the
  policy utility of coin-flipping.
* **retrieval_tfidf** (grounded, no generation). TF-IDF cosine retrieval,
  implemented in numpy, over the authoritative text each item carries: standard,
  clause, section title and the verified source excerpt. The question is the
  query; the answer is the best-matching excerpt, cited by its clause. It never
  reads the answer key or the adjacent-wrong entry. Deterministic, run once.
* **retrieval_bm25** (grounded, no generation). Okapi BM25 over the same
  documents, same answer format. k1 = 1.2, b = 0.75 (the textbook defaults;
  not tuned on this corpus). IDF is the Robertson-Sparck Jones form
  ln((N - df + 0.5) / (df + 0.5)), which is negative for a term found in more
  than half of the 68 documents.
* **retrieval_bm25plus** (grounded, no generation). BM25+ (Lv and Zhai, CIKM
  2011): k1 = 1.2, b = 0.75, delta = 1.0, IDF ln((N + 1) / df), which is never
  negative. Same documents and answer format.
* **retrieval_rrf** (grounded, no generation). Reciprocal rank fusion
  (Cormack, Clarke and Buettcher, SIGIR 2009) of the TF-IDF and BM25 rankings,
  k = 60, ties broken by corpus order.
* **oracle** (fixture). Reference answer plus gold citation. The ceiling.

Confidence intervals are 95 percent percentile cluster bootstraps resampling
item families (2000 resamples, seed recorded in the run files), so minimal
pairs are resampled together. Accuracy and adjacent substitution are over the
63 factual items; citation correctness and safety utility are over all 68.
For random_floor the five seed replicates of each item are pooled and travel
with their family, so the interval measures uncertainty from which item
families are in the corpus, not the seed-to-seed spread of the coin flips;
five seeds are enough to average the coin flips, not to widen the interval.
The deterministic arms are run once, and the oracle's degenerate [100%, 100%]
intervals say only that every family scores identically, which is true by
construction rather than a finding about precision.
"Safety utility" is the mean of the named `safety` policy in
`grounding_eval/scoring/score.py` (correct 1.0, acceptable abstention 0.9,
other abstention 0.25, other incorrect -0.5, adjacent substitution -1.0). It is
a stated policy, not a measurement.

## Table

| adapter | arm | repeats | accuracy (factual, n=63) [95% CI] | adjacent substitution [95% CI] | citation correct clause [95% CI] | category-error declined | safety utility [95% CI] |
|---|---|---|---|---|---|---|---|
| random_floor | fixture | 5 | 52.1% [45.9%, 58.9%] | 47.3% [40.4%, 53.5%] | 5.0% [2.1%, 8.2%] | 12.0% | 0.03 [-0.09, 0.14] |
| retrieval_tfidf | grounded | 1 | 27.0% [15.4%, 40.0%] | 0.0% [0.0%, 0.0%] | 73.5% [61.2%, 85.1%] | 0.0% | -0.12 [-0.29, 0.06] |
| retrieval_bm25 | grounded | 1 | 28.6% [17.2%, 40.9%] | 0.0% [0.0%, 0.0%] | 75.0% [63.8%, 85.3%] | 0.0% | -0.10 [-0.27, 0.07] |
| retrieval_bm25plus | grounded | 1 | 27.0% [15.4%, 40.0%] | 0.0% [0.0%, 0.0%] | 70.6% [57.6%, 81.8%] | 0.0% | -0.12 [-0.29, 0.06] |
| retrieval_rrf | grounded | 1 | 27.0% [15.4%, 40.0%] | 0.0% [0.0%, 0.0%] | 73.5% [61.3%, 84.6%] | 0.0% | -0.12 [-0.29, 0.06] |
| oracle | fixture | 1 | 100.0% [100.0%, 100.0%] | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 20.0% | 1.00 [1.00, 1.00] |

The parameters of retrieval_bm25plus and retrieval_rrf were written into
`grounding_eval/adapters/bm25plus.py`, at the values the cited papers
recommend, before either was run on the corpus, and one configuration of each
was run. Nothing was tuned on the 68 items, so these rows are not an upper
bound on what lexical retrieval can do here; they are what standard settings do.

The runner also reports how often each retriever's top document was the item's
own source clause: TF-IDF 48 of 68, BM25 49, BM25+ 47, RRF 47.

## By domain and on minimal pairs

`results/baselines_breakdown.md` verbatim (full rows in
`results/baselines_breakdown.csv`). Factual accuracy with 95 percent cluster
bootstrap CIs (same seed and resample count as the table above); `n` is the
number of factual items, or of pairs for the last row. `minimal_pair_items` are
the 46 factual items that belong to one of the 23 complete minimal pairs;
`pairs_both_correct` is the share of pairs where both members are correct
(one value per pair and repeat, resampled by pair). Per-domain intervals rest
on 5 to 10 items and are very wide; they are reported so nothing is hidden,
not because any domain-level difference is established.

| subset | n | random_floor | retrieval_tfidf | retrieval_bm25 | retrieval_bm25plus | retrieval_rrf | oracle |
|---|---|---|---|---|---|---|---|
| all_factual | 63 | 52.1% [45.9, 58.9] | 27.0% [15.4, 40.0] | 28.6% [17.2, 40.9] | 27.0% [15.4, 40.0] | 27.0% [15.4, 40.0] | 100.0% [100.0, 100.0] |
| confined_space | 8 | 50.0% [26.7, 72.5] | 25.0% [0.0, 66.7] | 25.0% [0.0, 66.7] | 25.0% [0.0, 66.7] | 25.0% [0.0, 66.7] | 100.0% [100.0, 100.0] |
| hazard_communication | 5 | 48.0% [40.0, 55.0] | 20.0% [0.0, 75.0] | 20.0% [0.0, 75.0] | 20.0% [0.0, 75.0] | 20.0% [0.0, 75.0] | 100.0% [100.0, 100.0] |
| injury_recordkeeping | 9 | 48.9% [37.8, 62.9] | 44.4% [14.3, 72.7] | 44.4% [14.3, 72.7] | 44.4% [14.3, 72.7] | 44.4% [14.3, 72.7] | 100.0% [100.0, 100.0] |
| lockout_tagout | 9 | 57.8% [46.0, 71.1] | 44.4% [20.0, 71.4] | 44.4% [20.0, 71.4] | 44.4% [20.0, 71.4] | 44.4% [20.0, 71.4] | 100.0% [100.0, 100.0] |
| machine_electrical_fire | 7 | 60.0% [40.0, 73.3] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] |
| pressure_relief_devices | 10 | 48.0% [32.7, 75.0] | 40.0% [7.7, 80.0] | 50.0% [22.2, 81.8] | 40.0% [7.7, 80.0] | 40.0% [7.7, 80.0] | 100.0% [100.0, 100.0] |
| process_safety_management | 8 | 47.5% [42.5, 53.3] | 12.5% [0.0, 37.5] | 12.5% [0.0, 37.5] | 12.5% [0.0, 37.5] | 12.5% [0.0, 37.5] | 100.0% [100.0, 100.0] |
| respiratory_protection_and_noise | 7 | 57.1% [40.0, 80.0] | 14.3% [0.0, 37.5] | 14.3% [0.0, 37.5] | 14.3% [0.0, 37.5] | 14.3% [0.0, 37.5] | 100.0% [100.0, 100.0] |
| minimal_pair_items | 46 | 52.2% [45.3, 59.6] | 28.3% [14.9, 41.7] | 30.4% [17.0, 43.8] | 28.3% [14.9, 41.7] | 28.3% [14.9, 41.7] | 100.0% [100.0, 100.0] |
| unpaired_items | 17 | 51.8% [40.0, 63.5] | 23.5% [5.9, 47.1] | 23.5% [5.9, 47.1] | 23.5% [5.9, 47.1] | 23.5% [5.9, 47.1] | 100.0% [100.0, 100.0] |
| pairs_both_correct | 23 | 28.7% [20.0, 38.3] | 13.0% [0.0, 26.1] | 13.0% [0.0, 26.1] | 13.0% [0.0, 26.1] | 13.0% [0.0, 26.1] | 100.0% [100.0, 100.0] |

The four retrievers are within one item of each other everywhere: BM25 answers
18 of 63 factual items correctly, the other three 17, and the one extra item is
in pressure_relief_devices. On minimal pairs all four get both members right
for 3 of 23 pairs (13.0 percent), below the random floor's 28.7 percent.
None of them answers a machine_electrical_fire item correctly.

## Answer or abstain: risk-coverage

Method and references: `docs/methodology.md` section 5a. Each retriever's
confidence is its top retrieval score; the system answers the items scoring at
or above a threshold and abstains on the rest. Computed on the 63 factual items
by `grounding_eval/selective.py`; `results/selective_summary.md` verbatim below
(full curves in `results/selective_curves.csv`). AURC is mean selective risk, so
lower is better; "uninformative" is the overall error rate and "best ordering"
puts every correct answer first. 95 percent cluster bootstrap CIs. Parentheses
give the coverage actually reached, because tied scores are answered together.

| adapter | accuracy at full coverage | AURC [95% CI] | AURC, uninformative score | AURC, best ordering | E-AURC | selective accuracy at 25% coverage | at 50% | at 75% |
|---|---|---|---|---|---|---|---|---|
| random_floor | 52.1% | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| retrieval_tfidf | 27.0% | 0.564 [0.385, 0.751] | 0.730 | 0.382 | 0.182 | 50.0% [22.2, 75.0] (answers 25%) | 40.6% [20.6, 59.4] (answers 51%) | 33.3% [18.4, 50.0] (answers 76%) |
| retrieval_bm25 | 28.6% | 0.571 [0.393, 0.765] | 0.714 | 0.362 | 0.209 | 56.2% [26.7, 81.2] (answers 25%) | 40.6% [21.9, 61.3] (answers 51%) | 31.2% [18.0, 46.0] (answers 76%) |
| retrieval_bm25plus | 27.0% | 0.602 [0.401, 0.807] | 0.730 | 0.382 | 0.219 | 43.8% [18.8, 76.5] (answers 25%) | 40.6% [20.0, 61.3] (answers 51%) | 29.2% [14.9, 44.2] (answers 76%) |
| retrieval_rrf | 27.0% | 0.699 [0.555, 0.827] | 0.730 | 0.382 | 0.316 | 30.4% [17.5, 45.0] (answers 89%) | 30.4% [17.5, 45.0] (answers 89%) | 30.4% [17.5, 45.0] (answers 89%) |
| oracle | 100.0% | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

What the table supports, and no more:

* **The retrieval score is a weak abstention signal.** For TF-IDF, BM25 and BM25+
  the AURC is below the uninformative value (0.564 vs 0.730, 0.571 vs 0.714, 0.602
  vs 0.730), so answering only the highest-scoring items does raise accuracy. But
  the AURC intervals (for example TF-IDF [0.385, 0.751]) include the uninformative
  value, and the gap to the best possible ordering (E-AURC 0.18 to 0.22) is large.
* **Fused-rank scores carry almost no signal.** Reciprocal rank fusion scores take
  few distinct values, so most items tie; its AURC (0.699) is close to the
  uninformative 0.730, and at 25 and 50 percent targets it must already answer 89
  percent of items.
* **Abstaining on the lower-scoring half does not make a retriever safe.** At
  about 25 percent coverage the point estimates of selective accuracy are 43.8 to
  56.2 percent for the three scored-ranking retrievers, with intervals spanning
  roughly 19 to 81 percent, from 16 answered items; no claim of a difference
  between retrievers is supported.
* **random_floor and oracle report no confidence**, so no curve exists for them;
  an LLM arm that reports none likewise has nothing to threshold.

## Calibration: Brier score, ECE, reliability bins

Method and references: `docs/methodology.md` section 5b. Computed on the 63
factual items by `grounding_eval/calibration.py`; `results/calibration_summary.md`
verbatim below (also `results/calibration_summary.csv` and
`results/calibration_reliability.csv`). Brier and ECE treat the confidence as a
probability, so they are computed only where the score is bounded to [0, 1] by
construction: TF-IDF cosine similarity. BM25, BM25+ and RRF scores are not on that
scale and random_floor and oracle report no confidence, so those rows are n/a, not
imputed. 95 percent cluster bootstrap CIs.

| adapter | accuracy | mean confidence | Brier [95% CI] | Brier, constant at accuracy | ECE (10 bins) [95% CI] |
|---|---|---|---|---|---|
| random_floor | 52.1% | n/a | n/a (no confidence reported) | n/a | n/a |
| retrieval_tfidf | 27.0% | 0.447 | 0.202 [0.182, 0.221] | 0.197 | 0.192 [0.128, 0.307] |
| retrieval_bm25 | 28.6% | n/a | n/a (not on a [0, 1] scale) | n/a | n/a |
| retrieval_bm25plus | 27.0% | n/a | n/a (not on a [0, 1] scale) | n/a | n/a |
| retrieval_rrf | 27.0% | n/a | n/a (not on a [0, 1] scale) | n/a | n/a |
| oracle | 100.0% | n/a | n/a (no confidence reported) | n/a | n/a |

Reliability bins, retrieval_tfidf:

| bin | items | mean confidence | accuracy |
|---|---|---|---|
| [0.2, 0.3) | 8 | 0.272 | 12.5% |
| [0.3, 0.4) | 15 | 0.364 | 0.0% |
| [0.4, 0.5) | 21 | 0.452 | 33.3% |
| [0.5, 0.6) | 14 | 0.540 | 42.9% |
| [0.6, 0.7) | 3 | 0.657 | 33.3% |
| [0.7, 0.8) | 2 | 0.766 | 100.0% |

What the table supports, and no more:

* **TF-IDF's score is over-confident as a probability.** Mean confidence is 0.447
  against 27.0 percent accuracy. Its Brier score, 0.202, is above the 0.197 of
  reporting the 27.0 percent base rate for every item, so the score is no better
  than a constant as a probability forecast (the intervals overlap; the
  difference is 0.005). ECE is 0.192 [0.128, 0.307].
* **The bins are sparse.** Two bins hold 3 and 2 items, so the 33.3 percent and
  100 percent accuracies there rest on those counts. The two lowest-scoring
  occupied bins, [0.2, 0.3) and [0.3, 0.4), are 12.5 percent (1 of 8) and 0.0
  percent (0 of 15) correct; the reliability curve is not monotone.
* **Not computable for the other systems.** Without a probability-scale
  confidence, calibration cannot be assessed for BM25, BM25+, RRF, random_floor or
  oracle, and an LLM arm that reports no confidence would likewise be n/a.

## What this shows

* **The items discriminate.** Floor and ceiling are separated on every axis the
  benchmark cares about: adjacent substitution goes from 47.3 percent to 0,
  clause-level citation from 5.0 percent to 100, safety utility from 0.03 to
  1.00 (the oracle's unrounded utility is 0.9985, because one category-error
  reference answer scores as an acceptable abstention at 0.9). The CIs of the floor and the ceiling do not overlap on any column. A
  system that guesses is not confusable with a system that knows.
* **Citing the right clause is not the same as answering.** The retriever finds
  the governing clause for most items (73.5 percent clause-level citation) but
  its excerpt satisfies the answer key for only 27.0 percent of factual items
  (BM25 28.6, BM25+ 27.0, RRF 27.0; the CIs of all four overlap almost
  entirely),
  because a one-sentence source excerpt rarely carries every required concept.
  Retrieval without generation therefore sits below the random floor on accuracy
  and on safety utility, and above it on citation. Any grounded LLM arm should
  be expected to beat this on accuracy while at least matching it on citation;
  one that does not has failed to use what it retrieved.
* **A better lexical ranker does not fix this.** BM25, BM25+ and rank fusion
  move clause-level citation between 70.6 and 75.0 percent and accuracy by at
  most one item (28.6 percent for BM25, 27.0 for the others). The ceiling on
  excerpt-only answers is set by how much of the answer key a single source
  sentence carries, not by which lexical scoring function picks it.
* **Adjacent substitution is a generation failure, not a retrieval failure.**
  The retriever never substitutes the adjacent entity because it never
  paraphrases. The 47.3 percent rate of the floor is what a system that flips
  between the two entities produces, and it is the axis on which an ungrounded
  LLM arm would be expected to differ most from a grounded one.
* **Category-error items are not yet well served by any arm.** The oracle's
  reference answers for the 5 category-error items are scored CORRECT on
  concept coverage but only 1 of 5 trips the abstention detector, because the
  reference text rejects the premise rather than declining in the detector's
  vocabulary. That is a property of the abstention cue list, not of the items,
  and it should be read as a limitation of the "declined" column.

## What this does not show

Nothing here is evidence about any language model, any vendor product, or the
grounded-versus-ungrounded question the benchmark was built to ask. The
preregistration in `docs/preregistration.md` stands; these baselines fix the
scale on which the preregistered comparison will be read.

## Reproducing

    python3 -m grounding_eval.run_baselines

writes `results/` from the committed corpus with fixed seeds. The runner
takes under a minute on a laptop with numpy and pandas installed.
