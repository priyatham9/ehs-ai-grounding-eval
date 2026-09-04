# Preregistration

This file fixes the analysis plan **before** any real system is evaluated on this
corpus. It exists so that a later result cannot be the product of choices made
after seeing the data.

Status: **no system has been evaluated.** Only the mock demonstration fixture has
been run. Nothing in this repository is a result.

If any item below is changed after a real system has been run, the change must be
recorded in the Amendments section with its date and reason, and any affected
number must be reported as exploratory rather than confirmatory.

---

## 1. Frozen before any evaluation

### Corpus

- 68 items; 63 factual, 5 category-error; 23 complete minimal pairs; 44 families.
- Corpus digest at freeze:
  `a9fd4f4c2222dfb9f6135ee39a94ca9b92168867b14a56a5a319d16cb3980528`
  (recomputed by `python3 -m grounding_eval.cli corpus`; the digest changes if any
  identity-bearing field changes, and a run file recording a different digest is
  not comparable).
- Source verification: eCFR edition date 2026-08-01, 68/68 verified, 0 failures.
- The 6 quarantined items are excluded and will not be added mid-study.

### Scoring configuration

| Parameter | Value |
|---|---|
| `coverage_threshold` | 0.75 |
| `require_full_numeric` | true |
| `adjacent_min_terms` | 2 |
| `adjacent_low_coverage` | 0.5 |
| `policy` | `safety` |
| Contrast window | 70 characters |

Utility weights under the `safety` policy: correct 1.0, abstained 0.25, abstained
on a false-premise item 0.9, other incorrect −0.5, adjacent substitution −1.0.
These are a stated policy, not an estimate.

### Statistical plan

| Choice | Value |
|---|---|
| Primary outcome | Paired accuracy over the 23 minimal pairs |
| Secondary outcomes | Adjacent-substitution rate; per-item accuracy; abstention rate on answerable items; decline rate on category-error items; citation grade distribution |
| Arm comparison test | McNemar's exact test on discordant items |
| Interval method | Percentile cluster bootstrap over families, 2000 resamples, seed 20260904 |
| Alpha | 0.05, two-sided |
| Multiplicity | Only the primary outcome is confirmatory. Every secondary outcome is exploratory and will be labelled as such. |
| Repeats | 3 per arm minimum, reported separately with across-repeat spread |
| Temperature | Fixed and recorded per arm in the run file |

Runs are never pooled across arms scored under different configurations, and a run
file whose corpus digest differs from the current corpus raises rather than being
silently included.

### Arms

1. **Ungrounded** — instruction/persona prompt over parametric memory, no retrieval.
2. **Pseudo-grounded** — retrieval over a plausible but unauthoritative corpus.
3. **Grounded** — retrieval over a version-pinned authoritative corpus with
   device-type metadata filtering.

Primary comparison: grounded versus ungrounded on paired accuracy.
Secondary comparison of interest: pseudo-grounded versus ungrounded, where the
direction is **not predicted** — retrieval over unauthoritative sources may
plausibly perform worse than no retrieval.

---

## 2. Stated in advance so it cannot be claimed later

### Predictions

- Grounded will exceed ungrounded on paired accuracy.
- Adjacent-substitution rate will be highest in the ungrounded arm.
- Paired accuracy will be substantially below per-item accuracy in every arm.

### Outcomes that would count against the benchmark's premise

Recorded now so they cannot be reinterpreted later:

- If grounded and ungrounded do not differ on paired accuracy, that is a real
  result and will be reported as one.
- If the pseudo-grounded arm outperforms the grounded arm, that is a real result.
- If adjacent-substitution rates are near zero in every arm, the corpus failed to
  construct genuinely confusable items, and that will be reported as a limitation
  of the corpus rather than as evidence that systems handle the distinction well.

### Contamination

Portions of these regulations circulate on the open web, so an ungrounded arm may
have memorised some answers, which would compress the measured difference. This
will be probed with edition-specific values and reported. It will not be assumed
away.

### Human adjudication

The automatic adjacency detection is lexical and its contrast heuristic misfires
in both directions. Before any adjacent-substitution rate is published:

- a random sample of at least 20 percent of responses classified as adjacent
  substitutions, plus 20 percent of those classified correct, will be adjudicated
  by a human blind to the arm;
- inter-rater agreement will be reported (Krippendorff's alpha or Cohen's kappa)
  if more than one rater is used;
- the human-adjudicated figures are the **primary** report; automatic scores are a
  screen.

`grounding_eval.adjudication` exports the sample in a blinded, shuffled form.

---

## 3. What will not be claimed

- That grounding eliminates hallucination. Any claim is a measured difference with
  an interval under stated conditions.
- Anything about a population of questions beyond this corpus. There is no
  sampling frame; the item set is a construct.
- Any calibration claim about incident probability. Nothing here estimates one.
- Any generalisation from 46 CFR to ASME or API engineering practice.

---

## 4. Amendments

None. No system has been evaluated.

| Date | Change | Reason | Affected outcomes |
|---|---|---|---|
| — | — | — | — |
