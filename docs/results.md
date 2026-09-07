# Results: non-LLM baselines

**No language model has been evaluated on this benchmark yet.** No LLM API was
available on the machine these runs were made on, and no LLM arm is reported
anywhere in this repository. The three arms below are non-LLM baselines. They
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
| oracle | fixture | 1 | 100.0% [100.0%, 100.0%] | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 20.0% | 1.00 [1.00, 1.00] |

The runner also reports that the retriever's top document was the item's own
source clause for 48 of 68 items.

## What this shows

* **The items discriminate.** Floor and ceiling are separated on every axis the
  benchmark cares about: adjacent substitution goes from 47.3 percent to 0,
  clause-level citation from 5.0 percent to 100, safety utility from 0.03 to
  1.00 (the oracle's unrounded utility is 0.9985, because one category-error
  reference answer scores as an acceptable abstention at 0.9). The CIs of the floor and the ceiling do not overlap on any column. A
  system that guesses is not confusable with a system that knows.
* **Citing the right clause is not the same as answering.** The retriever finds
  the governing clause for most items (73.5 percent clause-level citation) but
  its excerpt satisfies the answer key for only 27.0 percent of factual items,
  because a one-sentence source excerpt rarely carries every required concept.
  Retrieval without generation therefore sits below the random floor on accuracy
  and on safety utility, and above it on citation. Any grounded LLM arm should
  be expected to beat this on accuracy while at least matching it on citation;
  one that does not has failed to use what it retrieved.
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
