# Results: non-LLM baselines

**No language model has been evaluated on this benchmark yet.** No LLM API was
available on the machine these runs were made on, and no LLM arm is reported
anywhere in this repository. The six arms below are non-LLM baselines. They
exercise the scoring procedure and report lexical retrieval over a finite source
index. The random and oracle fixtures read the answer key; their separation is
not validation that the benchmark distinguishes knowledgeable systems from
unknowledgeable ones.

**Updated interpretation:** [the exhaustive excerpt analysis](../results/answerability.md)
tests all 4,284 factual-question/indexed-excerpt combinations. Only 27/63 questions
have any scorer-passing answer in this fixed format. BM25 passes 18, leaving nine
recoverable ranking misses and 36 questions for which no indexed excerpt passes.
The corresponding joint-pair ceiling is 7/23. These findings replace the earlier
inference that the baseline scores by themselves identified a generation failure
or established a general retrieval ceiling. [Inspect the experiment](evidence-audit.html).

The original baseline numbers are produced by `python3 -m grounding_eval.run_baselines` and
written to `results/`. The table below is `results/baselines_table.md`
verbatim; the full per-item responses and scores are in
`results/baselines_<adapter>.json` and the summary row set is
`results/baselines_summary.csv`.

## Arms

* **random_floor** (fixture). For each item, picks uniformly between the two
  answer options the item carries (the reference answer and a sentence built
  from the adjacent-wrong entity's signature terms), and cites a clause drawn
  uniformly from the pool of clauses cited across the corpus. Run with 5 seeds.
  Its accuracy is about one half by construction. It reads the reference answer
  and adjacent-wrong vocabulary, unlike the retrievers. It illustrates the
  scorer's response to prepared alternatives; it is not an information-matched
  chance floor for retrieving excerpts.
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
for 3 of 23 pairs (13.0 percent). The answer-key-assisted random fixture reaches
28.7 percent from a different answer space, so this is not a fair chance comparison.
The fixed excerpt format permits at most 7 of 23 both-pass pairs under the scorer.
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

## Paired significance between arms

Method and references: `docs/methodology.md` section 5c. Every arm answers the
same 63 factual items, so arms are compared on the items they disagree about:
exact McNemar test on the discordant items, a paired cluster-bootstrap 95 percent
interval for the accuracy difference (families resampled, 2000 resamples, seed
20260904), and Holm adjustment across all 10 comparisons (the 6 pairs among the
four retrievers plus each retriever against random_floor). Computed by
`grounding_eval/significance.py`; `results/significance_pairwise.md` verbatim below
(also `results/significance_pairwise.csv`, which adds the sensitivity columns).
random_floor is compared at repeat 0 (35 of 63 correct, 55.6 percent; the five
repeats give 30 to 36 correct, mean 52.1 percent). The oracle is not compared.

| A | B | accuracy A | accuracy B | difference A-B [95% paired cluster CI] | A only right | B only right | McNemar exact p | Holm-adjusted p | significant at 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| retrieval_tfidf | retrieval_bm25 | 27.0% | 28.6% | -1.6 pts [-4.8, +0.0] | 0 | 1 | 1.000 | 1.000 | no |
| retrieval_tfidf | retrieval_bm25plus | 27.0% | 27.0% | +0.0 pts [+0.0, +0.0] | 0 | 0 | 1.000 | 1.000 | no |
| retrieval_tfidf | retrieval_rrf | 27.0% | 27.0% | +0.0 pts [+0.0, +0.0] | 0 | 0 | 1.000 | 1.000 | no |
| retrieval_bm25 | retrieval_bm25plus | 28.6% | 27.0% | +1.6 pts [+0.0, +4.8] | 1 | 0 | 1.000 | 1.000 | no |
| retrieval_bm25 | retrieval_rrf | 28.6% | 27.0% | +1.6 pts [+0.0, +4.8] | 1 | 0 | 1.000 | 1.000 | no |
| retrieval_bm25plus | retrieval_rrf | 27.0% | 27.0% | +0.0 pts [+0.0, +0.0] | 0 | 0 | 1.000 | 1.000 | no |
| retrieval_tfidf | random_floor | 27.0% | 55.6% | -28.6 pts [-47.6, -9.7] | 7 | 25 | 0.002 | 0.021 | yes |
| retrieval_bm25 | random_floor | 28.6% | 55.6% | -27.0 pts [-46.0, -8.2] | 8 | 25 | 0.005 | 0.032 | yes |
| retrieval_bm25plus | random_floor | 27.0% | 55.6% | -28.6 pts [-47.6, -9.7] | 7 | 25 | 0.002 | 0.021 | yes |
| retrieval_rrf | random_floor | 27.0% | 55.6% | -28.6 pts [-47.6, -9.7] | 7 | 25 | 0.002 | 0.021 | yes |

What the table supports, and no more:

* **The four retrievers are not distinguishable from each other.** All six
  retriever-retriever differences are not significant (raw and Holm-adjusted p =
  1.000). Across the 63 factual items the retrievers disagree on at most one item
  per pair (BM25 answers one item that each of the other three misses; TF-IDF,
  BM25+ and RRF agree on every item). With 0 or 1 discordant items the test has no
  power: it cannot show the retrievers are equivalent, only that this corpus
  cannot separate them. The BM25 versus TF-IDF point estimate of 1.6 points is
  one item.
* **Each retriever scores below the answer-key-assisted fixture at repeat 0.** Holm-adjusted
  p = 0.021 (TF-IDF, BM25+, RRF) and 0.032 (BM25), differences of -27.0 to -28.6
  points with paired intervals that exclude zero. This is a statement about
  retrievers scoring below a coin flip between the two answer options on this
  item set. The fixture has access to full reference answers and the retrievers
  do not; statistical significance cannot remove this information mismatch.
* **That comparison depends on which random_floor repeat is used.** Raw McNemar p
  across the five repeats ranges from 0.0005 to 0.035 for TF-IDF, BM25+ and RRF,
  and from 0.0014 to 0.058 for BM25, so for BM25 the weakest repeat is not
  significant even before Holm adjustment. The conclusion that retrievers are
  below the fixture is consistent in direction on every repeat but not uniformly
  significant at 0.05 once the correction is applied.
* **The test treats items as independent.** Families (minimal pairs) cluster, so
  the McNemar p-values are optimistic; the cluster-bootstrap intervals are the
  cluster-aware check, and for the random-floor contrasts they exclude zero too.

## Hypothetical score-as-probability diagnostic

Method and references: `docs/methodology.md` section 5b. Computed on the 63
factual items by `grounding_eval/calibration.py`; `results/calibration_summary.md`
verbatim below (also `results/calibration_summary.csv` and
`results/calibration_reliability.csv`). None of the retrievers declares a
probability of correctness. The retained TF-IDF calculation asks what happens
if its raw cosine score is interpreted as one; being bounded to [0, 1] does not
justify that interpretation. RRF scores are also within [0, 1], so their omission
is a diagnostic scope choice, not a mathematical range restriction. BM25 and
BM25+ have unbounded weight scores, and the fixtures report no confidence.
Intervals are 95 percent cluster bootstraps.

Hypothetical raw-score diagnostic: TF-IDF cosine is treated as a probability solely for this calculation. No retriever declares a correctness probability; bounded scores do not establish probabilistic calibration.

| adapter | accuracy | mean retrieval score | Brier [95% CI] | Brier, constant at accuracy | ECE (10 bins) [95% CI] |
|---|---|---|---|---|---|
| random_floor | 52.1% | n/a | n/a (no confidence reported) | n/a | n/a |
| retrieval_tfidf | 27.0% | 0.447 | 0.202 [0.182, 0.221] | 0.197 | 0.192 [0.128, 0.307] |
| retrieval_bm25 | 28.6% | n/a | n/a (raw-score diagnostic not computed) | n/a | n/a |
| retrieval_bm25plus | 27.0% | n/a | n/a (raw-score diagnostic not computed) | n/a | n/a |
| retrieval_rrf | 27.0% | n/a | n/a (raw-score diagnostic not computed) | n/a | n/a |
| oracle | 100.0% | n/a | n/a (no confidence reported) | n/a | n/a |

Raw-score bins, retrieval_tfidf:

| bin | items | mean retrieval score | accuracy |
|---|---|---|---|
| [0.2, 0.3) | 8 | 0.272 | 12.5% |
| [0.3, 0.4) | 15 | 0.364 | 0.0% |
| [0.4, 0.5) | 21 | 0.452 | 33.3% |
| [0.5, 0.6) | 14 | 0.540 | 42.9% |
| [0.6, 0.7) | 3 | 0.657 | 33.3% |
| [0.7, 0.8) | 2 | 0.766 | 100.0% |

What the table supports, and no more:

* **Raw cosine is a poor substitute for a correctness probability in this diagnostic.**
  Mean score is 0.447 against 27.0 percent scorer accuracy. Treating the score as
  a probability gives Brier 0.202, compared with 0.197 for the constant observed
  base rate, and ECE 0.192 [0.128, 0.307]. These are descriptive in-sample values;
  they neither measure calibration of a declared probability forecast nor show a
  statistically established difference from the constant predictor.
* **The bins are sparse.** Two bins hold 3 and 2 items, so the 33.3 percent and
  100 percent accuracies there rest on those counts. The two lowest-scoring
  occupied bins, [0.2, 0.3) and [0.3, 0.4), are 12.5 percent (1 of 8) and 0.0
  percent (0 of 15) correct; the reliability curve is not monotone.
* **No system here reports a correctness probability.** Probabilistic calibration
  would require such a forecast or a separately fitted and held-out calibration
  procedure. Selective ranking can still be evaluated using raw retrieval scores.

## What this shows

* **The fixed answer space limits the attainable score.** Exhausting all 68
  indexed excerpts for each of 63 factual questions yields 27 possible passes.
  For BM25, 18 pass now, nine can be recovered by selecting another indexed
  excerpt, and 36 cannot pass through reranking alone. This is conditional on
  the source index, response template and lexical rubric; it is not a semantic
  answerability bound or a result about full-document retrieval or synthesis.
* **Retrieving the designated source does not guarantee a scorer pass.** TF-IDF
  retrieves its own item source for 47/63 factual questions; 31 of those still
  fail. Missing lexical groups can reflect incomplete evidence or an inadequate
  match rule. The analysis records those groups and does not adjudicate which.
* **The scorer itself admits counterexamples.** The pictogram question `hc-005`
  accepts unrelated eight-hour excerpts `hn-001` and `rk-001`. Its numeric-only
  key does not check the number's referent. Removing clause/title metadata
  changes the attainable score from 27/63 to 24/63; that format sensitivity is
  not automatically an error, because headings can contribute relevant content.
* **The absence of adjacent-substitution labels does not locate a cause.** These
  four excerpt retrievers receive zero such labels on this corpus. No generative
  system was tested, and returning a wrong-entity passage is possible without
  paraphrasing. The random fixture's 47.3 percent rate comes from deliberately
  prepared adjacent answers; it is not an observed language-model failure rate.
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
