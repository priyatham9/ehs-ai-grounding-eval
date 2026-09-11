# Dataset Card: ehs-ai-grounding-eval

## Summary

A benchmark of 68 safety-critical technical questions across eight occupational health and safety regulatory domains. Each item tests whether an AI system answers a question about the correct entity (a rupture disk) versus a semantically adjacent but categorically inapplicable sibling (a pressure relief valve). The corpus measures **co-hyponym substitution** errors: fluent, well-formed answers about the wrong regulated entity.

**Source:** README.md, lines 72-153

## Composition

- **Total items:** 68 (63 factual, 5 category-error)
- **Complete minimal pairs:** 23 pairs, or 46 items
- **Regulatory domains:** 8
- **Question families:** 44
- **Primary sources:** 29 CFR, 40 CFR, 46 CFR (all public)

**Domains:**
- Pressure relief devices (12 items, 3 pairs)
- Lockout/tagout (10 items, 4 pairs)
- Confined space (8 items, 3 pairs)
- Process safety management (8 items, 3 pairs)
- Machine guarding / electrical / flammable liquids (8 items, 3 pairs)
- Injury recordkeeping (10 items, 3 pairs)
- Respiratory protection and noise (7 items, 3 pairs)
- Hazard communication (5 items, 1 pair)

**Source:** README.md, lines 74-88 and corpus_stats.py output

## Construction

Each item is built from a regulatory minimal pair: two entities treated separately by law that share vocabulary and regulatory context. The construction procedure is:

1. Identify a pair of entities or conditions (rupture disk and relief valve; 10% vs 20% accumulation limit; etc.)
2. Locate a public federal clause stating a substantive requirement for each member
3. Write two question stems differing only in which member they ask about
4. Write the reference answer from the clause text
5. Write the adjacent wrong answer with signature vocabulary from the wrong member
6. State why the confusion is dangerous in operational terms
7. Record the clause identifier, its URL, and a verbatim anchor span

Structural invariants enforced by tests:
- Every item's reference answer satisfies its own answer key
- No adjacency signature term appears in the reference answer
- Every minimal pair has exactly one 'a' and one 'b' member
- Every item is marked verified and appears in the source verification report

Five items have false premises and are scored separately: asking for the blowdown of a rupture disk or the set pressure of a bursting disc. Rejecting the premise is the correct response.

**Source:** docs/methodology.md, sections 2 and 3

## Verification

Every item asserts that a specific paragraph of a specific federal regulation says a specific thing. All such assertions are re-checked against primary sources. The verification script (`tools/verify_sources.py`) checks:

- The section resolves at a stated edition date
- The item's verbatim anchor text appears in the section text (after whitespace, quote and dash normalisation)
- Cited paragraph markers appear in the section with case preserved
- The anchor is located inside the paragraph cited, not merely somewhere in the same section

All 68 items are verified. Six drafted items were excluded from the corpus because their provenance could not be verified (corpus/quarantine/unverified.json).

Verification status is committed in corpus/verification/source_verification.json with a dated eCFR report.

**Source:** docs/methodology.md section 3, corpus/verification/, README.md line 84

## Intended use

The benchmark measures whether a language model or retrieval system provides answers attributable to authoritative source text and whether it distinguishes between semantically adjacent entities regulated separately. It tests:

- **Factual accuracy:** Does the answer to the question match the reference answer? (63 factual items scored independently)
- **Adjacent substitution rate:** How often does a system substitute the wrong entity? (Factual items only)
- **Citation correctness:** Does the cited source clause actually support the answer? (All 68 items)
- **Paired accuracy:** Does the system answer both members of a minimal pair correctly? (Stricter test for discrimination)
- **Category-error handling:** How does the system respond to false premises? (5 items scored separately, never folded into overall accuracy)

The scoring procedure is deterministic and reproducible. No language model or vendor product has been evaluated on this benchmark; the published results contain only non-LLM baselines (random floor, TF-IDF retrieval, oracle ceiling).

**Source:** README.md sections "What the benchmark is", "The scoring harness", "Provenance"; docs/methodology.md section 1; docs/results.md

## Limitations

- **Scope is narrow but intentional.** The benchmark measures co-hyponym substitution specifically, not hallucination in general. It does not measure factual grounding across the full range of possible errors.
- **Sample size is small.** 68 items across 8 domains limits statistical power per domain.
- **Sources are US federal regulations.** The corpus does not cover state, local, or international occupational safety standards.
- **Category-error items are not well served.** The oracle's reference answers score as correct on concept coverage but only 1 of 5 trips the abstention detector, because the reference text rejects the premise rather than declining in the detector's vocabulary. This is a limitation of the abstention cue list, not of the items.
- **Adjacent wrong answers are written by domain experts.** The signature terms and rationale reflect expert judgment about what confusion is dangerous and plausible. They are not derived from observed LLM errors.
- **Regulatory text changes.** The corpus is pinned to specific eCFR editions. Regulatory updates may alter the cited paragraphs or their numbering.

**Source:** docs/limitations.md; docs/results.md section "What this does not show"; README.md line 73

## License

MIT License. Copyright (c) 2026 Priyatham Chimmani. See LICENSE file in repository.

**Source:** LICENSE file, CITATION.cff

---

**Repository:** https://github.com/priyatham9/ehs-ai-grounding-eval

**Published site:** https://priyatham9.github.io/ehs-ai-grounding-eval/

**Version:** 0.1.0 (2026-09-07)
