# Co-hyponym substitution in safety-critical question answering: a source-verified benchmark and non-LLM baselines

**Status: unsubmitted draft. Not submitted, not posted, not peer reviewed.** Preprint-format working manuscript, prepared for the author's own review. No venue has been approached.

**Author:** Priyatham Chimmani (independent researcher; GitHub: priyatham9)
**Draft date:** 2026-10-27
**Code, data and results:** https://github.com/priyatham9/ehs-ai-grounding-eval (every figure below is read from a committed file; see Appendix A)

## Abstract

**Background.** Systems that answer safety-critical technical questions can fail by answering about a semantically adjacent but categorically inapplicable entity, for example giving relief-valve parameters for a rupture disk. We call this co-hyponym substitution. **Methods.** We built a benchmark of 68 questions across 8 occupational safety and health domains, each grounded in a public US federal regulation with a verbatim anchor span checked against the eCFR, with 10 further drafts quarantined and excluded from scoring because their sources could not be verified. We scored five retrieval or guessing arms and an oracle with a deterministic lexical scorer, and report accuracy with cluster-bootstrap, Wilson and exact (Clopper-Pearson) intervals, answer-or-abstain risk-coverage, calibration, and paired exact tests with Holm correction. **Results.** TF-IDF, BM25, BM25+ and reciprocal-rank-fusion retrieval answered 27.0%, 28.6%, 27.0% and 27.0% of the 63 factual items correctly, against 52.1% for a random choice between the two answer options; no retriever differed significantly from another (every Holm-adjusted p = 1.000), and each was significantly below random-floor repeat 0 (Holm-adjusted p 0.021 to 0.032). Retrieval score was a weak abstention signal and TF-IDF's cosine score was over-confident as a probability (ECE 0.192). **Conclusions.** The items separate a floor from a ceiling, and a plain lexical retriever lands below the floor on accuracy while citing the governing clause for most items. **No language model was evaluated**; the benchmark is released to make that comparison possible.

## 1. Introduction

An engineer asks an assistant how to size overpressure protection for a reactor protected by a rupture disk, and receives a fluent answer about a spring-loaded relief valve: set pressure, blowdown, reseating. A rupture disk is a one-shot non-reclosing device and has none of those parameters. The two are siblings under "pressure relief device", share a definitions section in the same standards, and a retrieval index built over those standards will often return the valve passage as the nearest neighbour of the disk passage. The answer survives casual review because it is well formed. We call this error class *co-hyponym substitution* and argue that it is a predictable interaction between retrieval over a lexically adversarial corpus and the terminology of that corpus, not a random defect.

This paper describes a benchmark built around that error, the non-LLM baselines that fix its scale, and the checks that make its numbers auditable. Its contribution is the instrument and the baseline measurements. It is not a model comparison: no language model has been run.

## 2. Methods

### 2.1 Corpus

The corpus has 68 items: 63 factual questions and 5 category-error questions whose premise is false (for example asking for the blowdown of a rupture disk), where rejecting the premise is the correct answer. Items fall in 8 domains and 44 question families; 23 complete minimal pairs (46 items) differ only in which of two confusable entities or conditions the stem asks about, and the other 17 factual items are unpaired. Factual items per domain range from 5 (hazard communication) to 10 (pressure relief devices). Every item cites a paragraph of a public federal regulation (29 CFR, 40 CFR, 46 CFR), carries a verbatim anchor span from it, a reference answer written from the clause, the specific adjacent wrong answer with its signature vocabulary, and a statement of why the confusion is dangerous. Copyrighted consensus standards (ASME, API, NFPA, ISO) are not quoted or redistributed and no answer key depends on their text. As a result the pressure-relief items rest on 46 CFR subpart 54.15, marine equipment regulation, which is a proxy for general chemical-plant practice and not the same thing.

### 2.2 Task and scoring

For each item a system returns an answer and, optionally, citations. Scoring is lexical and deterministic: concept coverage against explicit alternation lists, forbidden concepts, and all declared numeric facts present. The decision procedure runs in fixed order: empty or error output is unscorable; an explicit decline is an abstention; sufficient coverage with no forbidden concept asserted and all numbers present is CORRECT; an incorrect answer that asserts adjacency signature terms is an ADJACENT_SUBSTITUTION; anything else is OTHER_INCORRECT. A term counts as asserted only outside a contrast construction, so "a rupture disk has no blowdown" is not a substitution. Accuracy and adjacent substitution are computed on the factual items; citation correctness and the policy-weighted safety utility over all items. Scoring parameters were fixed in a preregistration before any system was run. The scorer's cost, that a correct paraphrase avoiding every listed alternative is a miss, makes accuracy a lower bound on substantive correctness.

### 2.3 Arms

Six arms were run through one interface. *random_floor* picks uniformly between the two answer options an item carries and cites a clause drawn uniformly from the clauses cited in the corpus; it was run with 5 seeds. *retrieval_tfidf* is TF-IDF cosine retrieval over the authoritative text each item carries (standard, clause, section title and verified excerpt), returning the best-matching excerpt cited by its clause; it never reads the answer key. *retrieval_bm25* is Okapi BM25 (k1 = 1.2, b = 0.75). *retrieval_bm25plus* is BM25+ [R8] with delta = 1.0 and a non-negative IDF. *retrieval_rrf* is reciprocal rank fusion [R7] of the TF-IDF and BM25 rankings with k = 60. *oracle* returns the reference answer and gold citation and is the ceiling. All parameters are standard published defaults written into code before the arms were run; one configuration of each was run and nothing was tuned on the corpus. These retrievers return an excerpt and do not generate text, so they bound what extraction alone achieves and are not proxies for a language model.

### 2.4 Statistics

Confidence intervals are 95% percentile bootstraps that resample whole question families, so a minimal pair enters a resample whole or not at all (2000 resamples, seed 20260904). For accuracy we also report the Wilson interval and the exact Clopper-Pearson interval [R6] computed from the number correct out of the number of factual items. Both treat items as independent, so they are narrower than the cluster interval where clustering matters and are shown as sensitivity checks; the cluster-bootstrap interval is the primary one. Clopper-Pearson is given for the deterministic arms and for random-floor repeat 0, not for pooled random-floor repeats, because pooled seed replicates of one item are not independent.

Answer-or-abstain behaviour uses the risk-coverage framework of El-Yaniv and Wiener [R9] and Geifman and El-Yaniv [R10]. Each retriever's confidence is its top retrieval score; the system answers items at or above a threshold. We report the area under the risk-coverage curve (AURC, lower is better), its excess over the best possible ordering (E-AURC) [R11], and selective accuracy at target coverages. Tied scores are answered together, so achieved coverage can exceed the target and is reported. An uninformative score has an AURC equal to the overall error rate.

Calibration uses the Brier score [R1] and expected calibration error (ECE) with equal-width bins [R2, R3] over 10 bins on [0, 1]. These treat the confidence as a probability, so they are computed only for TF-IDF, whose cosine similarity is bounded to [0, 1] by construction. BM25 and BM25+ scores are unbounded weight sums and fused-rank scores are rank-based; rescaling them would be a calibration procedure chosen after seeing the data, so they are reported as not applicable. random_floor and the oracle report no confidence.

Arms are compared on the items they disagree about with McNemar's exact test [R4] (the two-sided binomial test on discordant counts), with a paired cluster-bootstrap interval for the accuracy difference. Ten comparisons are made on the same data (6 among the four retrievers and 4 retrievers against random_floor), so p-values are adjusted by the Holm step-down procedure [R5] and significance is declared at an adjusted p below 0.05. random_floor is stochastic, so repeat 0 enters the family and the range of raw p over all repeats is reported as a sensitivity check. The oracle is excluded as the ceiling by construction.

### 2.5 Citation integrity and quarantine

Two separate checks guard citations. The first concerns the corpus: `tools/verify_sources.py` fetches each cited section from the eCFR versioner API at a stated edition date and checks that the section resolves, that the item's verbatim anchor text appears in it after whitespace, quote and dash normalisation, that the cited paragraph markers appear with case preserved, and that the anchor lies inside the cited paragraph and not merely elsewhere in the section. The locality check was added after an independent audit found an item that cited one paragraph while its anchor belonged to another, a defect the earlier checks passed. The report is committed and re-checked offline by the test suite. It confirms provenance only: not that the anchor is the most relevant span, that the answer key is complete, or that a clause was not later superseded.

The second concerns system responses. Each cited clause is parsed and graded as none, wrong, correct section or correct clause against the item's source, and citations of copyrighted standards that the item does not list are counted separately because a fabricated paragraph number of a paywalled standard cannot be checked for free. The grading is structural: it shows a system named the right clause, not that it retrieved or used it.

An item enters the scored corpus only if its clause identifier and the substance of its answer key can be checked against a public primary source the reader can also check. Drafts that fail are *quarantined* in `corpus/quarantine/unverified.json`: the loader never returns them, no scoring path reads the file, and their text is stored under a field name that cannot be mistaken for a question. Two kinds are held there. Drafts that depend on the text of paywalled standards whose clause numbers or values are edition-dependent, and drafts about regulatory currency, where the answer changed with a dated amendment: for the latter the amending Federal Register document (91 FR 1695, 89 FR 81829 and 88 FR 47254) was fetched and its quotation recorded, but the current eCFR text could not be fetched, so they stay excluded until the verifier can run.

## 3. Results

### 3.1 Accuracy

Table: factual accuracy over the 63 factual items. CI is the primary cluster-bootstrap interval; Wilson and exact Clopper-Pearson (CP) intervals treat items as independent.

| arm | accuracy | cluster-bootstrap 95% CI | Wilson 95% CI | exact CP 95% CI |
|---|---|---|---|---|
| random_floor (5 repeats, pooled) | 52.1% | [45.9%, 58.9%] | [46.6%, 57.5%] | not computed (pooled replicates) |
| random_floor (repeat 0) | 55.6% | not computed | not computed | [42.5%, 68.1%] |
| retrieval_tfidf | 27.0% | [15.4%, 40.0%] | [17.6%, 39.0%] | [16.6%, 39.7%] |
| retrieval_bm25 | 28.6% | [17.2%, 40.9%] | [18.9%, 40.7%] | [17.9%, 41.3%] |
| retrieval_bm25plus | 27.0% | [15.4%, 40.0%] | [17.6%, 39.0%] | [16.6%, 39.7%] |
| retrieval_rrf | 27.0% | [15.4%, 40.0%] | [17.6%, 39.0%] | [16.6%, 39.7%] |
| oracle | 100.0% | [100.0%, 100.0%] | [94.3%, 100.0%] | [94.3%, 100.0%] |

The four retrievers answered 17 (TF-IDF, BM25+, RRF) or 18 (BM25) of 63 items correctly. The oracle's degenerate cluster interval reflects that every family scores identically by construction and says nothing about precision; the Wilson and exact intervals are the informative ones for it. Adjacent substitution was 47.3% [40.4%, 53.5%] for random_floor and 0.0% for every retriever and the oracle, as expected for systems that never paraphrase. The retrievers cited the governing clause for 73.5% (TF-IDF, interval [61.2%, 85.1%]), 75.0% (BM25), 70.6% (BM25+) and 73.5% (RRF) of the 68 items, against 5.0% for random_floor and 100.0% for the oracle. None of the retrievers declined a category-error item (0.0%); the oracle did so for 20.0% only because its reference text rejects the premise in wording the abstention detector largely does not recognise, a limitation of that detector. On minimal pairs, all four retrievers answered both members of 13.0% of the pairs correctly, below random_floor's 28.7%.

### 3.2 Selective accuracy and risk-coverage

AURC is mean selective risk, so lower is better; the uninformative value equals the overall error rate and the best ordering ranks every correct answer first.

| arm | AURC [95% CI] | uninformative | best ordering | E-AURC | selective accuracy at 25% coverage |
|---|---|---|---|---|---|
| retrieval_tfidf | 0.564 [0.385, 0.751] | 0.730 | 0.382 | 0.182 | 50.0% [22.2%, 75.0%] (reached coverage 25.4%) |
| retrieval_bm25 | 0.571 [0.393, 0.765] | 0.714 | 0.362 | 0.209 | 56.2% [26.7%, 81.2%] (25.4%) |
| retrieval_bm25plus | 0.602 [0.401, 0.807] | 0.730 | 0.382 | 0.219 | 43.8% [18.8%, 76.5%] (25.4%) |
| retrieval_rrf | 0.699 [0.555, 0.827] | 0.730 | 0.382 | 0.316 | 30.4% [17.5%, 45.0%] (88.9%) |

For TF-IDF, BM25 and BM25+ the AURC is below the uninformative value, so answering only the highest-scoring items does raise accuracy, but each AURC interval includes the uninformative value and the gap to the best ordering is large. Fused-rank scores take few distinct values, so most items tie; RRF must already answer 88.9% of items to reach the 25% and 50% targets and its AURC is close to uninformative. For TF-IDF, selective accuracy is 40.6% [20.6%, 59.4%] at 50% coverage (reached 50.8%) and 33.3% [18.4%, 50.0%] at 75% coverage (reached 76.2%). The point estimates at 25% coverage rest on 16 answered items and the intervals overlap heavily, so no difference between retrievers is supported. random_floor and the oracle report no confidence and have no curve.

### 3.3 Calibration

TF-IDF is the only arm whose score is a probability-scale quantity. Mean confidence is 0.447 against 27.0% accuracy. The Brier score is 0.202 [0.182, 0.221], against 0.197 for a constant forecast of the 27.0% base rate, so the score is no better than the base rate as a probability forecast (the intervals overlap and the difference is 0.005). ECE over 10 bins is 0.192 [0.128, 0.307], which is expected: a cosine similarity is not a probability of correctness. Reliability bins are sparse and not monotone:

| confidence bin | items | mean confidence | accuracy |
|---|---|---|---|
| [0.2, 0.3) | 8 | 0.272 | 12.5% |
| [0.3, 0.4) | 15 | 0.364 | 0.0% |
| [0.4, 0.5) | 21 | 0.452 | 33.3% |
| [0.5, 0.6) | 14 | 0.540 | 42.9% |
| [0.6, 0.7) | 3 | 0.657 | 33.3% |
| [0.7, 0.8) | 2 | 0.766 | 100.0% |

Only 6 of the 10 bins are occupied and the top two hold 3 and 2 items, so ECE is a coarse summary. Calibration cannot be assessed for any other arm.

### 3.4 Paired significance

Differences are first-named arm minus second-named arm, in percentage points, with the paired cluster-bootstrap interval; "A only" and "B only" count items only the first or second arm answered correctly.

| A | B | difference [95% CI] | A only | B only | McNemar exact p | Holm-adjusted p |
|---|---|---|---|---|---|---|
| retrieval_tfidf | retrieval_bm25 | -1.6 pts [-4.8, +0.0] | 0 | 1 | 1.000 | 1.000 |
| retrieval_tfidf | retrieval_bm25plus | +0.0 pts [+0.0, +0.0] | 0 | 0 | 1.000 | 1.000 |
| retrieval_tfidf | retrieval_rrf | +0.0 pts [+0.0, +0.0] | 0 | 0 | 1.000 | 1.000 |
| retrieval_bm25 | retrieval_bm25plus | +1.6 pts [+0.0, +4.8] | 1 | 0 | 1.000 | 1.000 |
| retrieval_bm25 | retrieval_rrf | +1.6 pts [+0.0, +4.8] | 1 | 0 | 1.000 | 1.000 |
| retrieval_bm25plus | retrieval_rrf | +0.0 pts [+0.0, +0.0] | 0 | 0 | 1.000 | 1.000 |
| retrieval_tfidf | random_floor (repeat 0) | -28.6 pts [-47.6, -9.7] | 7 | 25 | 0.002 | 0.021 |
| retrieval_bm25 | random_floor (repeat 0) | -27.0 pts [-46.0, -8.2] | 8 | 25 | 0.005 | 0.032 |
| retrieval_bm25plus | random_floor (repeat 0) | -28.6 pts [-47.6, -9.7] | 7 | 25 | 0.002 | 0.021 |
| retrieval_rrf | random_floor (repeat 0) | -28.6 pts [-47.6, -9.7] | 7 | 25 | 0.002 | 0.021 |

The four retrievers cannot be told apart: the pairs disagree on at most one item, so the test has no power, and a non-significant result here means the corpus cannot separate them, not that they are equivalent. Each retriever is significantly below the random floor at repeat 0 after Holm adjustment, with paired intervals that exclude zero. That conclusion depends on the repeat: across the five repeats the raw McNemar p ranges from 0.001 to 0.035 for TF-IDF, BM25+ and RRF and from 0.001 to 0.058 for BM25, so for BM25 the weakest repeat is not significant even before adjustment. The direction is consistent on every repeat; uniform significance at 0.05 after correction is not. The exact test treats items as independent, so its p-values are optimistic where families cluster; the cluster intervals are the cluster-aware check and also exclude zero.

## 4. Discussion

Three observations follow, each limited to what the numbers support. First, the items discriminate: floor and ceiling differ on every axis (adjacent substitution, clause citation, accuracy) with non-overlapping intervals, so a system that guesses is not confusable with one that knows. Second, finding the governing clause is not the same as answering: the retrievers cite the right clause for most items but their one-sentence excerpt satisfies the whole answer key for only about a quarter of factual items, which places retrieval-without-generation below a coin flip on accuracy and above it on citation. Third, a better lexical ranker does not change this: the four retrievers differ by at most one item, so the ceiling on excerpt-only answers is set by how much of an answer key one source sentence carries and not by the scoring function that picks it. Adjacent substitution is a generation failure and not a retrieval failure here, because a system that never paraphrases never substitutes; this is the axis on which a generative arm, grounded or not, would be expected to differ most, and it is untested.

The benchmark's value for later work is the combination of exact, source-checked items and an auditable scoring path. The preregistration fixes the analysis before any language model is run, so the comparison that motivated the benchmark can be read on the scale these baselines establish.

## 5. Limitations

1. **No language model has been evaluated.** No LLM arm was run and no LLM API was available where these runs were made. Nothing here is evidence about any language model, any vendor product, or the grounded-versus-ungrounded question the benchmark was built to ask. An adapter for the Anthropic Messages API is implemented and tested with mocked network access only. The mock adapter's outputs are labelled demonstration fixtures and are not results.
2. **The scorer is lexical.** A correct paraphrase avoiding every listed alternative is scored as a miss, so accuracy is a lower bound whose gap is unknown and may vary by system. The assertion-versus-mention heuristic can misfire in both directions, so any adjacent-substitution rate for a generative system must be checked against a human-adjudicated subsample; the retrievers' 0.0% is structural and is not subject to this.
3. **The corpus is a construct, not a sample.** It has 68 items hand-built by one author with the failure mode in mind; item selection is adversarial by design, so error rates should be expected to exceed those on natural questions, by an unknown amount. There was no external review of the answer keys.
4. **Federal regulation is a proxy for consensus standards.** Results do not transfer to ASME or API practice, and the pressure-relief items rely on marine equipment regulation.
5. **Power is limited.** With 63 factual items in 44 families the intervals are wide, per-domain intervals rest on 5 to 10 items, and the study can resolve large differences only. The exact tests treat items as independent and are optimistic under clustering.
6. **Calibration is computable for one arm only**, and that arm's score is a similarity and not a probability, so its large ECE is a statement about that mismatch.
7. **Verification checks provenance, not editorial quality**, and is pinned to one eCFR edition; it cannot detect later amendments. Four items in the quarantine concern exactly such amendments.
8. **Citation grading is structural.** It cannot tell retrieval from recall of a plausible clause identifier.
9. **Contamination is unmeasured.** Parts of 29 CFR circulate widely and an ungrounded model may have memorised answers.
10. **Utility weights are policy,** not measurement, and are not used in any claim above.

## 6. Reproducibility

All results regenerate from the committed corpus with fixed seeds by `python3 -m grounding_eval.run_baselines`, in under a minute with numpy and pandas. `python3 tools/check_manuscript.py` and `tests/test_manuscript.py` re-derive every number in this manuscript from the files named in Appendix A and fail if one drifts. Code is MIT licensed.

## Acknowledgements and declarations

No funding. No competing interests are declared. The work is independent. This draft was prepared with the assistance of an AI model (Claude); the author is responsible for its content and has not yet reviewed it for submission.

## Appendix A. Source of every number

Every numeric value in the sections above appears in the Printed column of one row below. The Locator is machine-readable and is resolved by `tools/check_manuscript.py`, run by `tests/test_manuscript.py`: it reads the named committed file (or, for `cp`, recomputes the exact Clopper-Pearson bound from the counts of other rows) and fails if the printed value does not equal the source value rounded to the printed precision, or if a number in the prose has no row. Locator forms: `csv::file::column=value&...::column` (one matching row), `csvn::` (product of two columns, an integer count), `json::` and `jsonlen::` (a value or list length, optionally filtered; an empty filter value selects entries lacking the key), `corpus::` (a count from the loaded corpus), `grep::file::literal` (the literal occurs in the file and contains every number printed), and `cp::lo|hi::K::N`. Percentages are the source fraction times 100; "pts" are percentage points. The trivial values 0 and 1 (interval ends, "one item") are exempt from the prose check. Items-per-domain, exact intervals and counts correct (K, N) are derived and are labelled by their locators.

### A.1 Corpus, verification and quarantine

| ID | Printed | Source file | Locator |
|---|---|---|---|
| C1 | 68 | `corpus/items/` | `corpus::n_items` |
| C2 | 63 | `corpus/items/` | `corpus::n_factual` |
| C3 | 5 | `corpus/items/` | `corpus::n_category` |
| C4 | 23 | `corpus/items/` | `corpus::n_pairs` |
| C5 | 44 | `corpus/items/` | `corpus::n_families` |
| C6 | 8 | `corpus/items/` | `corpus::n_domains` |
| C7 | 46 | `results/baselines_breakdown.csv` | `csv::results/baselines_breakdown.csv::adapter=oracle&subset=minimal_pair_items::n_items` |
| C8 | 17 | `results/baselines_breakdown.csv` | `csv::results/baselines_breakdown.csv::adapter=oracle&subset=unpaired_items::n_items` |
| C9 | 5 | `results/baselines_breakdown.csv` | `csv::results/baselines_breakdown.csv::adapter=oracle&subset=domain:hazard_communication::n_items` |
| C10 | 10 | `results/baselines_breakdown.csv` | `csv::results/baselines_breakdown.csv::adapter=oracle&subset=domain:pressure_relief_devices::n_items` |
| C11 | 29, 40, 46 | `README.md` | `grep::README.md::29 CFR, 40 CFR, 46 CFR` |
| V1 | 68 | `corpus/verification/source_verification.json` | `json::corpus/verification/source_verification.json::n_checked` |
| V2 | 68 | `corpus/verification/source_verification.json` | `json::corpus/verification/source_verification.json::n_ok` |
| V3 | 0 | `corpus/verification/source_verification.json` | `json::corpus/verification/source_verification.json::n_failed` |
| V5 | 0 | `corpus/verification/source_verification.json` | `jsonlen::corpus/verification/source_verification.json::anchor_locality_warnings` |
| Q1 | 10 | `corpus/quarantine/unverified.json` | `jsonlen::corpus/quarantine/unverified.json::items` |
| Q2 | 6 | `corpus/quarantine/unverified.json` | `jsonlen::corpus/quarantine/unverified.json::items?item_class=` |
| Q3 | 4 | `corpus/quarantine/unverified.json` | `jsonlen::corpus/quarantine/unverified.json::items?item_class=regulatory_currency` |
| C12 | 46, 54.15 | `docs/limitations.md` | `grep::docs/limitations.md::46 CFR subpart 54.15` |
| FR1 | 91, 1695 | `corpus/quarantine/unverified.json` | `grep::corpus/quarantine/unverified.json::91 FR 1695` |
| FR2 | 89, 81829 | `corpus/quarantine/unverified.json` | `grep::corpus/quarantine/unverified.json::89 FR 81829` |
| FR3 | 88, 47254 | `corpus/quarantine/unverified.json` | `grep::corpus/quarantine/unverified.json::88 FR 47254` |

### A.1 Configuration

| ID | Printed | Source file | Locator |
|---|---|---|---|
| M1 | 5 | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=random_floor::n_repeats` |

### A.1 Accuracy, substitution, citation and pairs

| ID | Printed | Source file | Locator |
|---|---|---|---|
| FA1 | 52.1% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=random_floor::accuracy` |
| FA2 | 45.9% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=random_floor::accuracy_ci_low` |
| FA3 | 58.9% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=random_floor::accuracy_ci_high` |
| FA4 | 46.6% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=random_floor::accuracy_wilson_low` |
| FA5 | 57.5% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=random_floor::accuracy_wilson_high` |
| FJ1 | 47.3% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=random_floor::adjacent_substitution_rate` |
| FJ2 | 40.4% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=random_floor::adjacent_ci_low` |
| FJ3 | 53.5% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=random_floor::adjacent_ci_high` |
| FG1 | 5.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=random_floor::citation_correct_clause_rate` |
| FG3 | 8.2% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=random_floor::citation_ci_high` |
| FW1 | 28.7% | `results/baselines_breakdown.csv` | `csv::results/baselines_breakdown.csv::adapter=random_floor&subset=pairs_both_correct::accuracy` |
| TA1 | 27.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_tfidf::accuracy` |
| TA2 | 15.4% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_tfidf::accuracy_ci_low` |
| TA3 | 40.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_tfidf::accuracy_ci_high` |
| TA4 | 17.6% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_tfidf::accuracy_wilson_low` |
| TA5 | 39.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_tfidf::accuracy_wilson_high` |
| TK1 | 17 | `results/baselines_summary.csv` | `csvn::results/baselines_summary.csv::adapter=retrieval_tfidf::accuracy*n_factual_obs` |
| TN1 | 63 | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_tfidf::n_factual_obs` |
| TA6 | 16.6% | `results/baselines_summary.csv` | `cp::lo::TK1::TN1` |
| TA7 | 39.7% | `results/baselines_summary.csv` | `cp::hi::TK1::TN1` |
| TJ1 | 0.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_tfidf::adjacent_substitution_rate` |
| TJ2 | 0.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_tfidf::adjacent_ci_low` |
| TJ3 | 0.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_tfidf::adjacent_ci_high` |
| TG1 | 73.5% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_tfidf::citation_correct_clause_rate` |
| TG2 | 61.2% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_tfidf::citation_ci_low` |
| TG3 | 85.1% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_tfidf::citation_ci_high` |
| TD1 | 0.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_tfidf::category_error_declined_rate` |
| TW1 | 13.0% | `results/baselines_breakdown.csv` | `csv::results/baselines_breakdown.csv::adapter=retrieval_tfidf&subset=pairs_both_correct::accuracy` |
| BA1 | 28.6% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_bm25::accuracy` |
| BA2 | 17.2% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_bm25::accuracy_ci_low` |
| BA3 | 40.9% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_bm25::accuracy_ci_high` |
| BA4 | 18.9% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_bm25::accuracy_wilson_low` |
| BA5 | 40.7% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_bm25::accuracy_wilson_high` |
| BK1 | 18 | `results/baselines_summary.csv` | `csvn::results/baselines_summary.csv::adapter=retrieval_bm25::accuracy*n_factual_obs` |
| BN1 | 63 | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_bm25::n_factual_obs` |
| BA6 | 17.9% | `results/baselines_summary.csv` | `cp::lo::BK1::BN1` |
| BA7 | 41.3% | `results/baselines_summary.csv` | `cp::hi::BK1::BN1` |
| BJ1 | 0.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_bm25::adjacent_substitution_rate` |
| BJ2 | 0.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_bm25::adjacent_ci_low` |
| BJ3 | 0.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_bm25::adjacent_ci_high` |
| BG1 | 75.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_bm25::citation_correct_clause_rate` |
| BD1 | 0.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_bm25::category_error_declined_rate` |
| BW1 | 13.0% | `results/baselines_breakdown.csv` | `csv::results/baselines_breakdown.csv::adapter=retrieval_bm25&subset=pairs_both_correct::accuracy` |
| PA1 | 27.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_bm25plus::accuracy` |
| PA2 | 15.4% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_bm25plus::accuracy_ci_low` |
| PA3 | 40.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_bm25plus::accuracy_ci_high` |
| PA4 | 17.6% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_bm25plus::accuracy_wilson_low` |
| PA5 | 39.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_bm25plus::accuracy_wilson_high` |
| PK1 | 17 | `results/baselines_summary.csv` | `csvn::results/baselines_summary.csv::adapter=retrieval_bm25plus::accuracy*n_factual_obs` |
| PN1 | 63 | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_bm25plus::n_factual_obs` |
| PA6 | 16.6% | `results/baselines_summary.csv` | `cp::lo::PK1::PN1` |
| PA7 | 39.7% | `results/baselines_summary.csv` | `cp::hi::PK1::PN1` |
| PJ1 | 0.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_bm25plus::adjacent_substitution_rate` |
| PJ2 | 0.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_bm25plus::adjacent_ci_low` |
| PJ3 | 0.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_bm25plus::adjacent_ci_high` |
| PG1 | 70.6% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_bm25plus::citation_correct_clause_rate` |
| PD1 | 0.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_bm25plus::category_error_declined_rate` |
| PW1 | 13.0% | `results/baselines_breakdown.csv` | `csv::results/baselines_breakdown.csv::adapter=retrieval_bm25plus&subset=pairs_both_correct::accuracy` |
| RA1 | 27.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_rrf::accuracy` |
| RA2 | 15.4% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_rrf::accuracy_ci_low` |
| RA3 | 40.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_rrf::accuracy_ci_high` |
| RA4 | 17.6% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_rrf::accuracy_wilson_low` |
| RA5 | 39.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_rrf::accuracy_wilson_high` |
| RK1 | 17 | `results/baselines_summary.csv` | `csvn::results/baselines_summary.csv::adapter=retrieval_rrf::accuracy*n_factual_obs` |
| RN1 | 63 | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_rrf::n_factual_obs` |
| RA6 | 16.6% | `results/baselines_summary.csv` | `cp::lo::RK1::RN1` |
| RA7 | 39.7% | `results/baselines_summary.csv` | `cp::hi::RK1::RN1` |
| RJ1 | 0.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_rrf::adjacent_substitution_rate` |
| RJ2 | 0.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_rrf::adjacent_ci_low` |
| RJ3 | 0.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_rrf::adjacent_ci_high` |
| RG1 | 73.5% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_rrf::citation_correct_clause_rate` |
| RD1 | 0.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=retrieval_rrf::category_error_declined_rate` |
| RW1 | 13.0% | `results/baselines_breakdown.csv` | `csv::results/baselines_breakdown.csv::adapter=retrieval_rrf&subset=pairs_both_correct::accuracy` |
| OA1 | 100.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=oracle::accuracy` |
| OA2 | 100.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=oracle::accuracy_ci_low` |
| OA3 | 100.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=oracle::accuracy_ci_high` |
| OA4 | 94.3% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=oracle::accuracy_wilson_low` |
| OA5 | 100.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=oracle::accuracy_wilson_high` |
| OK1 | 63 | `results/baselines_summary.csv` | `csvn::results/baselines_summary.csv::adapter=oracle::accuracy*n_factual_obs` |
| ON1 | 63 | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=oracle::n_factual_obs` |
| OA6 | 94.3% | `results/baselines_summary.csv` | `cp::lo::OK1::ON1` |
| OA7 | 100.0% | `results/baselines_summary.csv` | `cp::hi::OK1::ON1` |
| OJ1 | 0.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=oracle::adjacent_substitution_rate` |
| OJ2 | 0.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=oracle::adjacent_ci_low` |
| OJ3 | 0.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=oracle::adjacent_ci_high` |
| OG1 | 100.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=oracle::citation_correct_clause_rate` |
| OG2 | 100.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=oracle::citation_ci_low` |
| OG3 | 100.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=oracle::citation_ci_high` |
| OD1 | 20.0% | `results/baselines_summary.csv` | `csv::results/baselines_summary.csv::adapter=oracle::category_error_declined_rate` |
| OW1 | 100.0% | `results/baselines_breakdown.csv` | `csv::results/baselines_breakdown.csv::adapter=oracle&subset=pairs_both_correct::accuracy` |
| FK0 | 35 | `results/significance_pairwise.csv` | `csvn::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=random_floor::acc_b*n_items` |
| FN0 | 63 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=random_floor::n_items` |
| FA0 | 55.6% | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=random_floor::acc_b` |
| FA7 | 42.5% | `results/significance_pairwise.csv` | `cp::lo::FK0::FN0` |
| FA8 | 68.1% | `results/significance_pairwise.csv` | `cp::hi::FK0::FN0` |

### A.1 Selective accuracy and risk-coverage

| ID | Printed | Source file | Locator |
|---|---|---|---|
| TS1 | 0.564 | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_tfidf::aurc` |
| TS2 | 0.730 | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_tfidf::aurc_random` |
| TS3 | 0.382 | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_tfidf::aurc_optimal` |
| TS4 | 0.182 | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_tfidf::e_aurc` |
| TS5 | 0.385 | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_tfidf::aurc_ci_low` |
| TS6 | 0.751 | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_tfidf::aurc_ci_high` |
| TSA25 | 50.0% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_tfidf::sel_acc_at_25` |
| TSC25 | 25.4% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_tfidf::achieved_cov_at_25` |
| TSL25 | 22.2% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_tfidf::sel_acc_at_25_ci_low` |
| TSH25 | 75.0% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_tfidf::sel_acc_at_25_ci_high` |
| TSA50 | 40.6% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_tfidf::sel_acc_at_50` |
| TSC50 | 50.8% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_tfidf::achieved_cov_at_50` |
| TSL50 | 20.6% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_tfidf::sel_acc_at_50_ci_low` |
| TSH50 | 59.4% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_tfidf::sel_acc_at_50_ci_high` |
| TSA75 | 33.3% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_tfidf::sel_acc_at_75` |
| TSC75 | 76.2% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_tfidf::achieved_cov_at_75` |
| TSL75 | 18.4% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_tfidf::sel_acc_at_75_ci_low` |
| TSH75 | 50.0% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_tfidf::sel_acc_at_75_ci_high` |
| BS1 | 0.571 | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25::aurc` |
| BS2 | 0.714 | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25::aurc_random` |
| BS3 | 0.362 | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25::aurc_optimal` |
| BS4 | 0.209 | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25::e_aurc` |
| BS5 | 0.393 | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25::aurc_ci_low` |
| BS6 | 0.765 | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25::aurc_ci_high` |
| BSA25 | 56.2% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25::sel_acc_at_25` |
| BSC25 | 25.4% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25::achieved_cov_at_25` |
| BSL25 | 26.7% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25::sel_acc_at_25_ci_low` |
| BSH25 | 81.2% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25::sel_acc_at_25_ci_high` |
| BSA50 | 40.6% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25::sel_acc_at_50` |
| BSC50 | 50.8% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25::achieved_cov_at_50` |
| BSC75 | 76.2% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25::achieved_cov_at_75` |
| BSH75 | 46.0% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25::sel_acc_at_75_ci_high` |
| PS1 | 0.602 | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25plus::aurc` |
| PS2 | 0.730 | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25plus::aurc_random` |
| PS3 | 0.382 | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25plus::aurc_optimal` |
| PS4 | 0.219 | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25plus::e_aurc` |
| PS5 | 0.401 | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25plus::aurc_ci_low` |
| PS6 | 0.807 | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25plus::aurc_ci_high` |
| PSA25 | 43.8% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25plus::sel_acc_at_25` |
| PSC25 | 25.4% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25plus::achieved_cov_at_25` |
| PSL25 | 18.8% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25plus::sel_acc_at_25_ci_low` |
| PSH25 | 76.5% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25plus::sel_acc_at_25_ci_high` |
| PSA50 | 40.6% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25plus::sel_acc_at_50` |
| PSC50 | 50.8% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25plus::achieved_cov_at_50` |
| PSL50 | 20.0% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25plus::sel_acc_at_50_ci_low` |
| PSC75 | 76.2% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_bm25plus::achieved_cov_at_75` |
| RS1 | 0.699 | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_rrf::aurc` |
| RS2 | 0.730 | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_rrf::aurc_random` |
| RS3 | 0.382 | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_rrf::aurc_optimal` |
| RS4 | 0.316 | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_rrf::e_aurc` |
| RS5 | 0.555 | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_rrf::aurc_ci_low` |
| RS6 | 0.827 | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_rrf::aurc_ci_high` |
| RSA25 | 30.4% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_rrf::sel_acc_at_25` |
| RSC25 | 88.9% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_rrf::achieved_cov_at_25` |
| RSL25 | 17.5% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_rrf::sel_acc_at_25_ci_low` |
| RSH25 | 45.0% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_rrf::sel_acc_at_25_ci_high` |
| RSA50 | 30.4% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_rrf::sel_acc_at_50` |
| RSC50 | 88.9% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_rrf::achieved_cov_at_50` |
| RSL50 | 17.5% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_rrf::sel_acc_at_50_ci_low` |
| RSH50 | 45.0% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_rrf::sel_acc_at_50_ci_high` |
| RSA75 | 30.4% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_rrf::sel_acc_at_75` |
| RSC75 | 88.9% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_rrf::achieved_cov_at_75` |
| RSL75 | 17.5% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_rrf::sel_acc_at_75_ci_low` |
| RSH75 | 45.0% | `results/selective_summary.csv` | `csv::results/selective_summary.csv::adapter=retrieval_rrf::sel_acc_at_75_ci_high` |

### A.1 Calibration (retrieval_tfidf)

| ID | Printed | Source file | Locator |
|---|---|---|---|
| L1 | 0.447 | `results/calibration_summary.csv` | `csv::results/calibration_summary.csv::adapter=retrieval_tfidf::mean_confidence` |
| L2 | 0.202 | `results/calibration_summary.csv` | `csv::results/calibration_summary.csv::adapter=retrieval_tfidf::brier` |
| L3 | 0.182 | `results/calibration_summary.csv` | `csv::results/calibration_summary.csv::adapter=retrieval_tfidf::brier_ci_low` |
| L4 | 0.221 | `results/calibration_summary.csv` | `csv::results/calibration_summary.csv::adapter=retrieval_tfidf::brier_ci_high` |
| L5 | 0.197 | `results/calibration_summary.csv` | `csv::results/calibration_summary.csv::adapter=retrieval_tfidf::brier_constant` |
| L6 | 0.192 | `results/calibration_summary.csv` | `csv::results/calibration_summary.csv::adapter=retrieval_tfidf::ece` |
| L7 | 0.128 | `results/calibration_summary.csv` | `csv::results/calibration_summary.csv::adapter=retrieval_tfidf::ece_ci_low` |
| L8 | 0.307 | `results/calibration_summary.csv` | `csv::results/calibration_summary.csv::adapter=retrieval_tfidf::ece_ci_high` |
| L9 | 10 | `results/calibration_summary.csv` | `csv::results/calibration_summary.csv::adapter=retrieval_tfidf::n_bins` |

### A.1 Paired significance

| ID | Printed | Source file | Locator |
|---|---|---|---|
| YL2 | 0.2 | `results/calibration_reliability.csv` | `csv::results/calibration_reliability.csv::adapter=retrieval_tfidf&bin=2::low` |

### A.1 Calibration (retrieval_tfidf)

| ID | Printed | Source file | Locator |
|---|---|---|---|
| YC2 | 8 | `results/calibration_reliability.csv` | `csv::results/calibration_reliability.csv::adapter=retrieval_tfidf&bin=2::count` |
| YA2 | 12.5% | `results/calibration_reliability.csv` | `csv::results/calibration_reliability.csv::adapter=retrieval_tfidf&bin=2::accuracy` |
| YM2 | 0.272 | `results/calibration_reliability.csv` | `csv::results/calibration_reliability.csv::adapter=retrieval_tfidf&bin=2::mean_confidence` |

### A.1 Paired significance

| ID | Printed | Source file | Locator |
|---|---|---|---|
| YL3 | 0.3 | `results/calibration_reliability.csv` | `csv::results/calibration_reliability.csv::adapter=retrieval_tfidf&bin=3::low` |

### A.1 Calibration (retrieval_tfidf)

| ID | Printed | Source file | Locator |
|---|---|---|---|
| YC3 | 15 | `results/calibration_reliability.csv` | `csv::results/calibration_reliability.csv::adapter=retrieval_tfidf&bin=3::count` |
| YA3 | 0.0% | `results/calibration_reliability.csv` | `csv::results/calibration_reliability.csv::adapter=retrieval_tfidf&bin=3::accuracy` |
| YM3 | 0.364 | `results/calibration_reliability.csv` | `csv::results/calibration_reliability.csv::adapter=retrieval_tfidf&bin=3::mean_confidence` |

### A.1 Paired significance

| ID | Printed | Source file | Locator |
|---|---|---|---|
| YL4 | 0.4 | `results/calibration_reliability.csv` | `csv::results/calibration_reliability.csv::adapter=retrieval_tfidf&bin=4::low` |

### A.1 Calibration (retrieval_tfidf)

| ID | Printed | Source file | Locator |
|---|---|---|---|
| YC4 | 21 | `results/calibration_reliability.csv` | `csv::results/calibration_reliability.csv::adapter=retrieval_tfidf&bin=4::count` |
| YA4 | 33.3% | `results/calibration_reliability.csv` | `csv::results/calibration_reliability.csv::adapter=retrieval_tfidf&bin=4::accuracy` |
| YM4 | 0.452 | `results/calibration_reliability.csv` | `csv::results/calibration_reliability.csv::adapter=retrieval_tfidf&bin=4::mean_confidence` |

### A.1 Paired significance

| ID | Printed | Source file | Locator |
|---|---|---|---|
| YL5 | 0.5 | `results/calibration_reliability.csv` | `csv::results/calibration_reliability.csv::adapter=retrieval_tfidf&bin=5::low` |

### A.1 Calibration (retrieval_tfidf)

| ID | Printed | Source file | Locator |
|---|---|---|---|
| YC5 | 14 | `results/calibration_reliability.csv` | `csv::results/calibration_reliability.csv::adapter=retrieval_tfidf&bin=5::count` |
| YA5 | 42.9% | `results/calibration_reliability.csv` | `csv::results/calibration_reliability.csv::adapter=retrieval_tfidf&bin=5::accuracy` |
| YM5 | 0.540 | `results/calibration_reliability.csv` | `csv::results/calibration_reliability.csv::adapter=retrieval_tfidf&bin=5::mean_confidence` |

### A.1 Paired significance

| ID | Printed | Source file | Locator |
|---|---|---|---|
| YL6 | 0.6 | `results/calibration_reliability.csv` | `csv::results/calibration_reliability.csv::adapter=retrieval_tfidf&bin=6::low` |

### A.1 Calibration (retrieval_tfidf)

| ID | Printed | Source file | Locator |
|---|---|---|---|
| YC6 | 3 | `results/calibration_reliability.csv` | `csv::results/calibration_reliability.csv::adapter=retrieval_tfidf&bin=6::count` |
| YA6 | 33.3% | `results/calibration_reliability.csv` | `csv::results/calibration_reliability.csv::adapter=retrieval_tfidf&bin=6::accuracy` |
| YM6 | 0.657 | `results/calibration_reliability.csv` | `csv::results/calibration_reliability.csv::adapter=retrieval_tfidf&bin=6::mean_confidence` |

### A.1 Paired significance

| ID | Printed | Source file | Locator |
|---|---|---|---|
| YL7 | 0.7 | `results/calibration_reliability.csv` | `csv::results/calibration_reliability.csv::adapter=retrieval_tfidf&bin=7::low` |

### A.1 Calibration (retrieval_tfidf)

| ID | Printed | Source file | Locator |
|---|---|---|---|
| YC7 | 2 | `results/calibration_reliability.csv` | `csv::results/calibration_reliability.csv::adapter=retrieval_tfidf&bin=7::count` |
| YA7 | 100.0% | `results/calibration_reliability.csv` | `csv::results/calibration_reliability.csv::adapter=retrieval_tfidf&bin=7::accuracy` |
| YM7 | 0.766 | `results/calibration_reliability.csv` | `csv::results/calibration_reliability.csv::adapter=retrieval_tfidf&bin=7::mean_confidence` |

### A.1 Paired significance

| ID | Printed | Source file | Locator |
|---|---|---|---|
| P1D | -1.6 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=retrieval_bm25::diff` |
| P1L | -4.8 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=retrieval_bm25::diff_ci_low` |
| P1H | +0.0 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=retrieval_bm25::diff_ci_high` |
| P1A | 0 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=retrieval_bm25::n_a_only` |
| P1B | 1 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=retrieval_bm25::n_b_only` |
| P1M | 1.000 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=retrieval_bm25::p_mcnemar` |
| P1P | 1.000 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=retrieval_bm25::p_holm` |
| P2D | +0.0 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=retrieval_bm25plus::diff` |
| P2L | +0.0 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=retrieval_bm25plus::diff_ci_low` |
| P2H | +0.0 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=retrieval_bm25plus::diff_ci_high` |
| P2A | 0 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=retrieval_bm25plus::n_a_only` |
| P2B | 0 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=retrieval_bm25plus::n_b_only` |
| P2M | 1.000 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=retrieval_bm25plus::p_mcnemar` |
| P2P | 1.000 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=retrieval_bm25plus::p_holm` |
| P3D | +0.0 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=retrieval_rrf::diff` |
| P3L | +0.0 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=retrieval_rrf::diff_ci_low` |
| P3H | +0.0 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=retrieval_rrf::diff_ci_high` |
| P3A | 0 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=retrieval_rrf::n_a_only` |
| P3B | 0 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=retrieval_rrf::n_b_only` |
| P3M | 1.000 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=retrieval_rrf::p_mcnemar` |
| P3P | 1.000 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=retrieval_rrf::p_holm` |
| P4D | +1.6 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25&arm_b=retrieval_bm25plus::diff` |
| P4L | +0.0 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25&arm_b=retrieval_bm25plus::diff_ci_low` |
| P4H | +4.8 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25&arm_b=retrieval_bm25plus::diff_ci_high` |
| P4A | 1 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25&arm_b=retrieval_bm25plus::n_a_only` |
| P4B | 0 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25&arm_b=retrieval_bm25plus::n_b_only` |
| P4M | 1.000 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25&arm_b=retrieval_bm25plus::p_mcnemar` |
| P4P | 1.000 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25&arm_b=retrieval_bm25plus::p_holm` |
| P5D | +1.6 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25&arm_b=retrieval_rrf::diff` |
| P5L | +0.0 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25&arm_b=retrieval_rrf::diff_ci_low` |
| P5H | +4.8 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25&arm_b=retrieval_rrf::diff_ci_high` |
| P5A | 1 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25&arm_b=retrieval_rrf::n_a_only` |
| P5B | 0 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25&arm_b=retrieval_rrf::n_b_only` |
| P5M | 1.000 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25&arm_b=retrieval_rrf::p_mcnemar` |
| P5P | 1.000 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25&arm_b=retrieval_rrf::p_holm` |
| P6D | +0.0 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25plus&arm_b=retrieval_rrf::diff` |
| P6L | +0.0 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25plus&arm_b=retrieval_rrf::diff_ci_low` |
| P6H | +0.0 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25plus&arm_b=retrieval_rrf::diff_ci_high` |
| P6A | 0 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25plus&arm_b=retrieval_rrf::n_a_only` |
| P6B | 0 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25plus&arm_b=retrieval_rrf::n_b_only` |
| P6M | 1.000 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25plus&arm_b=retrieval_rrf::p_mcnemar` |
| P6P | 1.000 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25plus&arm_b=retrieval_rrf::p_holm` |
| P7D | -28.6 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=random_floor::diff` |
| P7L | -47.6 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=random_floor::diff_ci_low` |
| P7H | -9.7 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=random_floor::diff_ci_high` |
| P7A | 7 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=random_floor::n_a_only` |
| P7B | 25 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=random_floor::n_b_only` |
| P7M | 0.002 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=random_floor::p_mcnemar` |
| P7P | 0.021 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=random_floor::p_holm` |
| P8D | -27.0 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25&arm_b=random_floor::diff` |
| P8L | -46.0 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25&arm_b=random_floor::diff_ci_low` |
| P8H | -8.2 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25&arm_b=random_floor::diff_ci_high` |
| P8A | 8 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25&arm_b=random_floor::n_a_only` |
| P8B | 25 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25&arm_b=random_floor::n_b_only` |
| P8M | 0.005 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25&arm_b=random_floor::p_mcnemar` |
| P8P | 0.032 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25&arm_b=random_floor::p_holm` |
| P9D | -28.6 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25plus&arm_b=random_floor::diff` |
| P9L | -47.6 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25plus&arm_b=random_floor::diff_ci_low` |
| P9H | -9.7 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25plus&arm_b=random_floor::diff_ci_high` |
| P9A | 7 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25plus&arm_b=random_floor::n_a_only` |
| P9B | 25 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25plus&arm_b=random_floor::n_b_only` |
| P9M | 0.002 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25plus&arm_b=random_floor::p_mcnemar` |
| P9P | 0.021 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25plus&arm_b=random_floor::p_holm` |
| P10D | -28.6 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_rrf&arm_b=random_floor::diff` |
| P10L | -47.6 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_rrf&arm_b=random_floor::diff_ci_low` |
| P10H | -9.7 pts | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_rrf&arm_b=random_floor::diff_ci_high` |
| P10A | 7 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_rrf&arm_b=random_floor::n_a_only` |
| P10B | 25 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_rrf&arm_b=random_floor::n_b_only` |
| P10M | 0.002 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_rrf&arm_b=random_floor::p_mcnemar` |
| P10P | 0.021 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_rrf&arm_b=random_floor::p_holm` |
| PM1 | 0.001 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=random_floor::p_mcnemar_min_over_repeats` |
| PM2 | 0.035 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_tfidf&arm_b=random_floor::p_mcnemar_max_over_repeats` |
| PM3 | 0.001 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25&arm_b=random_floor::p_mcnemar_min_over_repeats` |
| PM4 | 0.058 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25&arm_b=random_floor::p_mcnemar_max_over_repeats` |
| PM5 | 0.05 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25&arm_b=random_floor::alpha` |
| PM6 | 10 | `results/significance_pairwise.csv` | `csv::results/significance_pairwise.csv::arm_a=retrieval_bm25&arm_b=random_floor::n_comparisons` |

### A.1 Configuration

| ID | Printed | Source file | Locator |
|---|---|---|---|
| H1 | 1.2 | `grounding_eval/adapters/bm25plus.py` | `grep::grounding_eval/adapters/bm25plus.py::BM25PLUS_K1 = 1.2` |
| H2 | 0.75 | `grounding_eval/adapters/bm25plus.py` | `grep::grounding_eval/adapters/bm25plus.py::BM25PLUS_B = 0.75` |
| H3 | 1.0 | `grounding_eval/adapters/bm25plus.py` | `grep::grounding_eval/adapters/bm25plus.py::BM25PLUS_DELTA = 1.0` |
| H4 | 60 | `grounding_eval/adapters/bm25plus.py` | `grep::grounding_eval/adapters/bm25plus.py::RRF_K = 60` |
| H5 | 20260904 | `grounding_eval/run_baselines.py` | `grep::grounding_eval/run_baselines.py::BOOT_SEED = 20260904` |
| H7 | 95 | `grounding_eval/stats.py` | `grep100::grounding_eval/stats.py::alpha: float = 0.05,` |
| H8 | 25, 50, 75 | `grounding_eval/selective.py` | `grep100::grounding_eval/selective.py::COVERAGE_TARGETS = (0.25, 0.5, 0.75)` |
| H9 | 16 | `results/selective_summary.csv` | `csvn::results/selective_summary.csv::adapter=retrieval_tfidf::achieved_cov_at_25*n_items` |

### A.1 Paired significance

| ID | Printed | Source file | Locator |
|---|---|---|---|
| YH1 | 0.8 | `results/calibration_reliability.csv` | `csv::results/calibration_reliability.csv::adapter=retrieval_tfidf&bin=7::high` |

## Appendix B. References and verification status

Each reference was checked in this session against a publisher, proceedings or index page returned by a web search, unless marked otherwise. "Details not verified" means a field could not be confirmed and is omitted or marked. The numbers in this appendix are bibliographic and are outside the numeric check.

* [R1] G. W. Brier. Verification of forecasts expressed in terms of probability. Monthly Weather Review 78(1), 1-3 (1950). Verified: title, journal, volume, issue, pages and DOI 10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2 as listed on the Wikipedia article on Brier (secondary source); the publisher page was not reachable. Earlier text in this repository gave the same citation from memory.
* [R2] M. Pakdaman Naeini, G. Cooper and M. Hauskrecht. Obtaining well calibrated probabilities using Bayesian binning. Proceedings of the AAAI Conference on Artificial Intelligence 29(1) (2015), DOI 10.1609/aaai.v29i1.9602. Verified on the AAAI OJS record (https://ojs.aaai.org/index.php/AAAI/article/view/9602); page range not shown there and not given here. The name was written "Naeini" in earlier repository text; the record lists the author as Mahdi Pakdaman Naeini. The record page does not itself show the ECE definition; Guo et al. [R3] attribute ECE to this paper.
* [R3] C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger. On calibration of modern neural networks. Proceedings of the 34th International Conference on Machine Learning, PMLR 70, 1321-1330 (2017). Verified on https://proceedings.mlr.press/v70/guo17a.html; the arXiv version (1706.04599) defines ECE as the bin-weighted absolute gap between accuracy and confidence over M equal-width bins.
* [R4] Q. McNemar. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika 12, 153-157 (1947). Verified title, author, journal, volume, pages and year on the Springer record (DOI 10.1007/BF02295996); issue not shown.
* [R5] S. Holm. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics 6(2), 65-70 (1979). Verified as listed in the Wikipedia article on the Holm-Bonferroni method (secondary source).
* [R6] C. J. Clopper and E. S. Pearson. The use of confidence or fiducial limits illustrated in the case of the binomial. Biometrika (1934). Authors, title, journal and year verified on an index record; volume and pages not verified and omitted.
* [R7] G. V. Cormack, C. L. A. Clarke and S. Buettcher. Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval, 758-759 (2009). Verified on the SIGIR 2009 anthology record.
* [R8] Y. Lv and C. Zhai. Lower-bounding term frequency normalization. Proceedings of the 20th ACM Conference on Information and Knowledge Management (CIKM 2011), DOI 10.1145/2063576.2063584. Title and DOI confirmed on an index record; authors, venue and year not shown there and taken from the repository's earlier citation (Lv and Zhai, CIKM 2011): details not independently verified.
* [R9] R. El-Yaniv and Y. Wiener. On the foundations of noise-free selective classification. Journal of Machine Learning Research 11, 1605-1641 (2010). Verified on https://jmlr.org/papers/v11/el-yaniv10a.html.
* [R10] Y. Geifman and R. El-Yaniv. Selective classification for deep neural networks. Advances in Neural Information Processing Systems 30 (2017). Verified on the NeurIPS proceedings record.
* [R11] Y. Geifman, G. Uziel and R. El-Yaniv. Bias-reduced uncertainty estimation for deep neural classifiers. ICLR 2019, arXiv:1805.08206. Authors, title and venue verified on the arXiv record; the record's abstract does not state the AURC and E-AURC definitions, so the attribution of those two quantities to this paper follows the repository's methodology note and is not independently verified.

Not cited because not verified: Wilson (1927) for the score interval, which is named without a citation; Efron and Tibshirani (1993) for the bootstrap, which appears in the repository's methodology note and is omitted here.
