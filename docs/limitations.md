# Limitations

Written to be read by someone deciding whether to trust a number produced with
this apparatus. Ordered roughly by how likely each is to change a conclusion.

## 1. The scorer is lexical

Concept coverage is substring and prefix matching over explicit alternation lists.
There are no embeddings anywhere in the scoring path.

**Consequence:** a correct answer that paraphrases around every listed alternative
is scored as a miss. Measured accuracy is therefore a **lower bound** on
substantive correctness, and the size of the gap is unknown and varies by system -
a verbose system that restates regulatory phrasing will be scored more generously
than a terse one that paraphrases, independent of correctness.

**Why it was accepted:** the benchmark's subject is whether a system distinguishes
near-synonymous technical entities. Using an embedding model as the scorer would
make the measuring instrument the same mechanism whose failure is under study, and
a scoring disagreement would be uninterpretable.

**Mitigation:** answer keys use alternation groups with several surface forms, and
the invariant that every reference answer must satisfy its own key catches the
worst cases. It does not catch keys that are merely narrow.

## 2. The contrast heuristic is the weakest link

Adjacency and forbidden-concept detection distinguish assertion from mention by
looking for a contrast cue in a 70-character left window.

**It misfires in both directions.** "This is not a relief valve, so blowdown does
not apply, and the blowdown would be 7 percent" would be under-penalised. A
correct contrast phrased at longer range - cue and term separated by more than 70
characters - would be over-penalised.

**Consequence:** the adjacent-substitution rate, which is the benchmark's headline
error class, is the number most exposed to scorer error.

**Required mitigation:** any published adjacent-substitution rate must be checked
against a human-adjudicated subsample, with the human figure reported as primary.
See `docs/preregistration.md` §2.

## 3. Federal regulation is a proxy for consensus standards

ASME BPVC and API 520/521 are copyrighted and are not used. The corpus is grounded
in 29 CFR, 40 CFR and 46 CFR.

**Consequence:** 46 CFR subpart 54.15 is US Coast Guard marine equipment
regulation. It contains real, substantive, device-specific requirements for both
rupture disks and relief valves, which is why it was chosen, and it names the ASME
paragraphs it modifies, which allows those identifiers to be corroborated from a
public source. But it is not the general chemical plant context the motivating
incident came from, and a system tuned to ASME and API practice may answer
correctly in substance while missing the 46 CFR framing.

Results do not transfer to ASME or API practice without further work.

## 4. The corpus is a construct, not a sample

68 items, hand-built by one author, with the failure mode in mind.

**Consequence:** there is no sampling frame and no basis for generalising to "the
questions a safety engineer asks." Item selection is adversarial by design - items
were chosen because a confusable neighbour exists - so error rates here should be
expected to exceed those on a naturally occurring question distribution, by an
unknown amount.

Author bias is unmitigated: the same person chose the entities, wrote the
questions, wrote the reference answers, and wrote the adjacency signatures. An
independent item set would be a substantial improvement.

## 5. Statistical power is limited

Sixty-three factual items, 23 pairs, 44 family clusters.

**Consequence:** the cluster bootstrap resamples 44 units, so intervals are wide,
and the study is powered to detect large differences between arms rather than
modest ones. A null result should be read as "this corpus cannot resolve a
difference of this size," not as "the arms are equivalent."

McNemar's test is exact, so it is valid at small discordant counts, but validity is
not power.

## 6. Verification checks provenance, not editorial quality

`tools/verify_sources.py` confirms that a cited section exists, that a verbatim
anchor appears in it, that the paragraph markers appear, and that the anchor sits
inside the cited paragraph.

**It cannot confirm** that the anchor is the most relevant span, that the reference
answer is complete, that the answer key captures what matters, or that a clause has
not been superseded by guidance or a letter of interpretation. Those judgements
are the author's, and they are not independently reviewed.

The verification passing is weaker evidence than it looks, and the record here
shows why. An independent audit re-ran the verifier against live eCFR and
reproduced 68/68, then found that `fe-002` cited `1910.157(d)` while its anchor
text belonged to `(e)(2)` - a defect the 68/68 result did not expose, because the
checks then in place could not. The item has been repointed to `(d)(6)` with an
anchor that is genuinely in that paragraph, and the locality check that catches
this class was added. The general lesson stands: a verifier only refutes the
errors it was built to look for, and the count of items it passes says nothing
about the errors it cannot see.

Two items (`rp-001`, `rp-002`) depend on values read by a human from a table whose
row/column alignment is not recoverable from the eCFR XML. They are flagged
individually.

## 7. Regulations change

The verification report is pinned to eCFR edition 2026-08-01. The offline recheck
run by the test suite confirms the report still covers the corpus; **it cannot
detect that a regulation changed after the report was written.** Only a fresh
online run does that. Re-run `tools/verify_sources.py` before publishing anything.

## 8. Utility weights and risk tiers are policy

The safety-policy weights and the per-item risk tiers are stated judgements, not
measured or elicited quantities. No expert panel set them. Any number derived from
them is a policy-weighted summary and must name the policy. They should not be
described as measurements, and the headline outcome counts are never risk-weighted.

## 9. Contamination is unmeasured

Portions of 29 CFR circulate widely on the open web. An ungrounded system may have
memorised some answers, which would compress the measured difference between arms
and understate the value of grounding. This has not been probed. Doing so with
edition-specific values is required before publication.

## 10. Citation checking is structural

Citation grading parses clause identifiers and compares them to the item's source.
It confirms that a system named the right clause. **It does not confirm that the
system retrieved that clause, read it, or used it** - a system that emits a
plausible clause identifier from parametric memory scores the same as one that
retrieved it. Distinguishing those requires instrumenting the retrieval step, which
is the adapter author's responsibility.

## 11. The motivating incident is n = 1

The rupture-disk substitution that motivated this work is a single anecdote. It
demonstrates nothing on its own. Every empirical claim must rest on the constructed
item set.

## 12. Single-author, unreviewed

The corpus, the scorer, the statistics and this document are the work of one
person. There has been no external review of the answer keys. A reviewer with
domain expertise disagreeing with a reference answer would be identifying a real
defect, and the item-level JSON is structured to make that disagreement easy to
express precisely.
