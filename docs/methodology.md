# Methodology

This document states how the benchmark is constructed and scored, in enough
detail that a reader who disagrees with a score can reconstruct it by hand and
say exactly which decision they would have made differently.

## 1. What is being measured

The object of measurement is whether a system's answer to a safety-critical
technical question is **attributable to an authoritative source**, and in
particular whether the system distinguishes between entities that are taxonomic
siblings in the governing standard.

The specific error class is **co-hyponym substitution**: answering about the
semantically adjacent wrong entity. A rupture disk and a spring-loaded pressure
relief valve are both "pressure relief devices." They share sentences, tables and
regulatory verbs. Their parameter spaces do not intersect. An answer that
substitutes one for the other is fluent, well-formed, and categorically
inapplicable.

This is not the same as measuring hallucination in general, and the benchmark
should not be described as if it were.

## 2. Item construction

Each item was built by the following procedure.

1. Identify a pair of entities or conditions that a governing regulation treats
   separately and that share vocabulary - rupture disk and relief valve; the
   non-fire and fire-case accumulation limits; lockout and tagout periodic
   inspection requirements; reporting and recording obligations.
2. Locate a public federal clause that states a substantive requirement for each
   member.
3. Write two question stems that differ only in which member they ask about.
4. Write the reference answer from the clause text.
5. Write the adjacent wrong answer, with the signature vocabulary that would
   appear if a system answered about the wrong member.
6. State why the confusion is dangerous, in operational terms.
7. Record the clause identifier, the URL, and a verbatim anchor span.

Items are grouped into **families** (the cluster unit for the bootstrap) and, where
the two stems form a contrast, into **minimal pairs** with roles `a` and `b`.

### Structural invariants, enforced by tests

- Every item's own reference answer must satisfy its own answer key. If it does
  not, the key is wrong, and every system measured against it is measured against
  a broken target.
- No adjacency signature term may appear in the item's own reference answer.
  Otherwise a correct response would be scored as an adjacent substitution, which
  would bias the headline statistic in the direction that flatters the benchmark.
- Every minimal pair has exactly one `a` and one `b`, and their questions differ.
- Every item is marked verified, and appears in the source verification report
  with a matching clause string.

The first two invariants were violated by four items during construction. Each was
found by these tests and corrected: `prd-004` restated the paired item's figure in
its own reference answer; `psm-008` required a number its own question stem
supplied; `rk-008` had answer-key phrasings too narrow to match natural wording;
`hc-004` spelled a number as a word where the scorer only recognised digits (which
was fixed in the scorer, not the item). Recording these here is part of the point:
the invariants are load-bearing, not decorative.

## 3. Source verification

Verification is provenance verification. `tools/verify_sources.py` fetches each
cited section from the eCFR versioner API and checks:

- the section resolves at a stated edition date;
- the item's verbatim `anchor_text` appears in the section text after
  whitespace, quote and dash normalisation;
- the cited paragraph markers appear in the section, with case preserved (CFR
  hierarchy distinguishes `(a)` from `(A)`);
- the anchor is located inside the paragraph the item cites, not merely somewhere
  in the same section (`anchor_in_cited_paragraph`).

The last check was added after the preceding three passed an item that cited
`1910.157(d)` but anchored on text belonging to `(e)(2)`. Marker presence alone
is too weak to catch that: `(d)` occurs somewhere in almost every section that
has a paragraph (d). Locality is reported as a tri-state and is `null` where the
question does not apply - a citation into an alphabetical definitions block has
no numbered paragraph to be inside. Any item whose locality is `false` is listed
in `anchor_locality_warnings`, and the test suite fails when that list is
non-empty.

It does **not** check that the answer key is a good answer, that the anchor is the
most relevant span, or that the reference answer is complete. Those are editorial
judgements a script cannot make.

The report is committed with its edition date and re-checked offline by the test
suite on every run, so an item whose clause is edited after verification fails
immediately.

### Table-derived values

Two items (`rp-001`, `rp-002`) take numeric values from Table 1 of 29 CFR
1910.134, which the eCFR XML renders as a flattened run of cells whose row and
column alignment is not machine-recoverable. For these, the automated check
confirms the mandatory-use sentence and the paragraph path; the specific assigned
protection factors were read from the table by a human. Their verification method
is recorded as `ecfr_api+human_table_read` rather than presented as equivalent to
the fully automated checks.

### Copyrighted standards

ASME BPVC, API 520/521, NFPA 70E and ISO 4126 are not fetched, quoted or
redistributed. No item's answer key depends on their text. Paragraph identifiers
appear only as cross-references corroborated from a public federal source that
names them.

This has a cost, stated plainly: the corpus is grounded in federal regulation
rather than in the consensus standards a practising engineer reaches for first,
and 46 CFR subpart 54.15 is marine equipment regulation rather than general
chemical plant context. It was chosen because it contains substantive,
device-specific requirements for both device types and is public. It is a proxy.

## 4. Scoring

### Determinism

Scoring is lexical and deterministic. Given the same corpus and the same response
text, the same score comes out. There are no embeddings and no learned components
in the scoring path.

The cost is that a paraphrase avoiding every listed alternative is scored as a
miss. The reason for accepting it: the benchmark's subject is whether a system
distinguishes near-synonymous technical entities, and scoring that with an
embedding model would use as the instrument the very mechanism under study.

### The decision procedure

In fixed order:

1. Adapter error or empty response → `UNSCORABLE`. Never counted as wrong.
2. Explicit decline with no substantive commitment → `ABSTAINED`. Hedging then
   answering is answering.
3. Concept coverage ≥ threshold, no forbidden concept asserted, all declared
   numeric facts present → `CORRECT`.
4. Not correct, and asserts ≥ `adjacent_min_terms` adjacency signature terms (or
   ≥ 1 when coverage is below `adjacent_low_coverage`) → `ADJACENT_SUBSTITUTION`.
5. Otherwise → `OTHER_INCORRECT`.

### Assertion versus mention

A forbidden or adjacency term counts only when **asserted**, meaning it appears
outside a contrast construction within a 70-character left window. "A rupture disk
has no blowdown and does not reseat" is a correct answer naming what it rules out,
and is not scored as a substitution.

This heuristic is the single weakest link in automatic scoring. It will misfire in
both directions. Any reported adjacent-substitution rate should be checked against
a human-adjudicated subsample; see `grounding_eval.adjudication` and
`docs/limitations.md`.

### Numeric facts

When a key declares numeric facts, all of them must be present. A wrong number in
a safety answer is not a partial-credit situation. Both digits and spelled-out
integers are recognised, so "twenty-four hours" and "24 hours" are the same claim.

## 5. Statistics

The design is paired (every arm answers every item; most items belong to a
minimal pair) and clustered (pair members share a stem, a source and an author).
The analysis matches both facts.

- **Arm comparison:** McNemar's exact test on discordant items. Exact rather than
  chi-square because discordant counts are often small at this corpus size. An
  independent-samples test would discard the pairing and is refused by the code,
  which raises if two arms' item sets differ rather than silently comparing an
  intersection.
- **Intervals:** percentile bootstrap resampling whole **families** with
  replacement, so a minimal pair enters a replicate whole or not at all. A Wilson
  interval ignoring clustering is reported alongside so the difference is visible.
- **Primary discrimination statistic:** paired accuracy, the fraction of minimal
  pairs where both members are correct.
- **Selective prediction:** risk-coverage curves rather than raw accuracy.
- **Repeats:** runs are never averaged silently. Spread across repeats is
  reported next to the centre, because in a safety setting an answer that is
  correct only sometimes is not a correct answer.

## 6. Policies, not estimates

Two sets of numbers in this repository are **stated policies**, not measured or
estimated quantities, and must be reported with the policy name attached:

- **Utility weights** (`safety` policy): correct 1.0, abstained on an answerable
  item 0.25, abstained on a false-premise item 0.9, other incorrect −0.5,
  adjacent substitution −1.0. These encode a judgement that a confident wrong
  answer about the adjacent device is worse than a plainly wrong answer, and both
  are worse than declining.
- **Risk tiers** on items: an author judgement about consequence, used only for an
  optional weighted summary and never for headline outcome counts.

Scoring thresholds (`coverage_threshold` 0.75, `adjacent_min_terms` 2,
`adjacent_low_coverage` 0.5, `require_full_numeric` true) are configuration. They
were chosen before any system was run against the corpus and are recorded in
`docs/preregistration.md`. Every reported number should name the configuration
that produced it.

## 7. Adapters and arms

An adapter turns an item into a response. Grounded and ungrounded systems are
scored through the same interface on the same code path, so a measured difference
cannot come from a difference in prompting or parsing.

An adapter representing a real system must use only `item.question` and at most
`item.id`. It must not read the answer key, the reference answer, or the
adjacent-wrong description. The mock adapter does read them because it is a
fixture rather than a system under test. **This rule is enforced socially, not
mechanically.** A reviewer evaluating a reported result should inspect the adapter
source.

The intended three-arm design is ungrounded / pseudo-grounded / grounded, where
the pseudo-grounded arm retrieves over a plausible but unauthoritative corpus.
That arm is the one worth running: it is where most deployments sit, it is the
least studied, and it is where retrieval may perform worse than none.

## 8. Reporting rules

- Outcome composition, never a single accuracy number. Adjacent substitutions and
  plainly wrong answers are not summed.
- Category-error items reported separately, never folded into overall accuracy.
- Every point estimate accompanied by an interval and by the across-repeat spread.
- Mock runs always display the provenance banner. Suppressing it raises.
