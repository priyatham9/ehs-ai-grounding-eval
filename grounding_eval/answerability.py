"""Exact single-excerpt score ceilings and retrieval-error decomposition.

This is a deterministic analysis of a finite answer space, not a model run.
Every factual question is scored against every indexed source excerpt using
the existing answer format and rubric. The resulting upper bound is conditional
on that format, index and lexical scorer; it is not semantic answerability.

Reproduce: python3 -m grounding_eval.answerability [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .corpus import Corpus, load_corpus
from .harness import corpus_digest
from .schema import AdapterResponse, Item, Outcome
from .scoring.concepts import assess_concepts
from .scoring.score import ScoringConfig, score_item

ROOT = Path(__file__).resolve().parents[1]
RETRIEVERS = ("retrieval_tfidf", "retrieval_bm25", "retrieval_bm25plus", "retrieval_rrf")


def excerpt_text(source_item: Item, wrapped: bool = True) -> str:
    """Exactly reproduce the committed retrievers' response template."""
    src = source_item.source
    if not wrapped:
        return src.anchor_text
    return "%s (%s, %s) states: %s See %s." % (
        src.standard, src.clause, src.section_title, src.anchor_text, src.clause
    )


def diagnose(item: Item, text: str) -> dict:
    response = AdapterResponse(item_id=item.id, text=text)
    score = score_item(item, response)
    concepts = assess_concepts(text, item.answer_key)
    return {
        "passes": score.outcome is Outcome.CORRECT,
        "outcome": score.outcome.value,
        "concept_coverage": score.concept_coverage,
        "numeric_coverage": score.numeric_coverage,
        "missing_concepts": [list(group) for group in concepts.missing_groups],
        "missing_numeric": list(concepts.missing_numeric),
        "forbidden_asserted": list(concepts.forbidden_asserted),
    }


def pair_counts(pairs: Sequence[tuple], passes: Mapping[str, bool]) -> dict:
    counts = {"both_pass": 0, "one_pass": 0, "neither_pass": 0}
    for a, b in pairs:
        count = int(passes[a.id]) + int(passes[b.id])
        counts[("neither_pass", "one_pass", "both_pass")[count]] += 1
    return {"n_pairs": len(pairs), **counts}


def threshold_curve(rows: Sequence[dict], pairs: Sequence[tuple]) -> list:
    """Only achievable thresholds; tied scores enter together, no interpolation.

    Pair counts retain all pairs as denominator. A pair with just one member
    answered is reported separately and cannot count as demonstrated success.
    """
    if not rows:
        return []
    if any(row["score"] is None for row in rows):
        raise ValueError("threshold analysis requires a score for every item")
    points = []
    for threshold in sorted({row["score"] for row in rows}, reverse=True):
        answered = {row["item_id"]: row for row in rows if row["score"] >= threshold}
        n_pass = sum(row["passes"] for row in answered.values())
        complete = sum(a.id in answered and b.id in answered for a, b in pairs)
        fractured = sum((a.id in answered) != (b.id in answered) for a, b in pairs)
        joint = sum(
            a.id in answered and b.id in answered
            and answered[a.id]["passes"] and answered[b.id]["passes"]
            for a, b in pairs
        )
        points.append({
            "threshold": threshold, "n_answered": len(answered), "n_pass": n_pass,
            "coverage": len(answered) / len(rows), "score_accuracy": n_pass / len(answered),
            "complete_pairs": complete, "fractured_pairs": fractured, "both_pass_pairs": joint,
        })
    return points


def analyse_space(corpus: Corpus) -> dict:
    """Exhaust the indexed response space; include counterexamples, not just totals."""
    items = []
    for item in corpus.factual():
        passing = {"wrapped": [], "bare": []}
        maximum_coverage = -1.0
        maximum_coverage_ids = []
        for source_item in corpus:
            for mode in passing:
                text = excerpt_text(source_item, wrapped=mode == "wrapped")
                score = score_item(item, AdapterResponse(item_id=item.id, text=text))
                if mode == "wrapped":
                    if score.concept_coverage > maximum_coverage:
                        maximum_coverage, maximum_coverage_ids = score.concept_coverage, [source_item.id]
                    elif score.concept_coverage == maximum_coverage:
                        maximum_coverage_ids.append(source_item.id)
                if score.outcome is Outcome.CORRECT:
                    passing[mode].append(source_item.id)
        own_wrapped = diagnose(item, excerpt_text(item))
        own_bare = diagnose(item, excerpt_text(item, wrapped=False))
        items.append({
            "item_id": item.id, "domain": item.domain, "family": item.family,
            "pair_id": item.pair_id, "question": item.question,
            "source_clause": item.source.clause, "source_title": item.source.section_title,
            "source_anchor": item.source.anchor_text,
            "reference_answer": item.correct_answer,
            "required_concepts": [list(group) for group in item.answer_key.required_concepts],
            "forbidden_concepts": [list(group) for group in item.answer_key.forbidden_concepts],
            "numeric_facts": [{"label": fact.label, "value": fact.value, "unit": fact.unit,
                               "tolerance_abs": fact.tolerance_abs} for fact in item.answer_key.numeric_facts],
            "own_wrapped": own_wrapped, "own_bare": own_bare,
            "passing_excerpt_ids": passing["wrapped"],
            "passing_bare_excerpt_ids": passing["bare"],
            "max_indexed_concept_coverage": maximum_coverage,
            "max_coverage_excerpt_ids": maximum_coverage_ids,
        })
    n = len(items)
    possible = {row["item_id"]: bool(row["passing_excerpt_ids"]) for row in items}
    return {
        "n_factual": n, "n_indexed_excerpts": len(corpus),
        "n_combinations_per_format": n * len(corpus),
        "n_attainable_wrapped": sum(possible.values()),
        "n_attainable_bare": sum(bool(row["passing_bare_excerpt_ids"]) for row in items),
        "n_own_wrapped_pass": sum(row["own_wrapped"]["passes"] for row in items),
        "n_own_bare_pass": sum(row["own_bare"]["passes"] for row in items),
        "metadata_dependent_own_pass_ids": [
            row["item_id"] for row in items if row["own_wrapped"]["passes"] and not row["own_bare"]["passes"]
        ],
        "pair_ceiling": pair_counts(corpus.pairs(), possible),
        "items": items,
    }


def load_baseline(name: str, corpus: Corpus, results_dir: Path) -> dict:
    """Reject stale, mock, repeated, incomplete or format-incompatible runs."""
    payload = json.loads((results_dir / ("baselines_%s.json" % name)).read_text())
    if payload["adapter"] != name or len(payload["runs"]) != 1:
        raise ValueError("expected one run for " + name)
    run = payload["runs"][0]
    if run["is_mock_demonstration"] or run["corpus_digest"] != corpus_digest(corpus):
        raise ValueError("mock or mismatched corpus for " + name)
    if run["scoring_config"] != ScoringConfig().describe():
        raise ValueError("scoring configuration mismatch for " + name)
    for key in ("responses", "scores"):
        ids = [row["item_id"] for row in run[key]]
        if len(ids) != len(set(ids)) or set(ids) != set(corpus.ids()):
            raise ValueError("incomplete or duplicated " + key)
    scores = {row["item_id"]: row for row in run["scores"]}
    for response in run["responses"]:
        item = corpus.by_id(response["item_id"])
        matched = corpus.by_id(response["metadata"]["matched_item"])
        if response["text"] != excerpt_text(matched):
            raise ValueError("stored excerpt no longer matches corpus: " + item.id)
        current = diagnose(item, response["text"])
        for key in ("outcome", "concept_coverage", "numeric_coverage"):
            if current[key] != scores[item.id][key]:
                raise ValueError("stored score no longer reproduces: " + item.id)
    return run


def analyse(root: Path = ROOT) -> dict:
    corpus = load_corpus(str(root / "corpus/items"))
    space = analyse_space(corpus)
    attainable = {row["item_id"]: bool(row["passing_excerpt_ids"]) for row in space["items"]}
    baseline_rows = []
    for name in RETRIEVERS:
        run = load_baseline(name, corpus, root / "results")
        responses = {row["item_id"]: row for row in run["responses"]}
        scores = {row["item_id"]: row for row in run["scores"]}
        rows = []
        for item in corpus.factual():
            response, score = responses[item.id], scores[item.id]
            passes = score["outcome"] == "correct"
            rows.append({
                "item_id": item.id, "matched_item_id": response["metadata"]["matched_item"],
                "score": response["confidence"], "passes": passes,
                "self_match": response["metadata"]["matched_item"] == item.id,
                "diagnostics": diagnose(item, response["text"]),
                "failure_class": "passes" if passes else (
                    "recoverable_by_reranking" if attainable[item.id] else "no_excerpt_passes"
                ),
            })
        passes = {row["item_id"]: row["passes"] for row in rows}
        curve = threshold_curve(rows, corpus.pairs())
        baseline_rows.append({
            "adapter": name, "n_pass": sum(passes.values()),
            "n_recoverable_by_reranking": sum(row["failure_class"] == "recoverable_by_reranking" for row in rows),
            "n_no_excerpt_passes": sum(row["failure_class"] == "no_excerpt_passes" for row in rows),
            "n_self_match": sum(row["self_match"] for row in rows),
            "n_self_match_but_fail": sum(row["self_match"] and not row["passes"] for row in rows),
            "pairs": pair_counts(corpus.pairs(), passes),
            "threshold_curve": curve, "items": rows,
        })
    source_paths = list((root / "corpus/items").glob("*.json"))
    source_paths += [root / "results" / ("baselines_%s.json" % name) for name in RETRIEVERS]
    source_paths += list((root / "grounding_eval/scoring").glob("*.py"))
    source_paths += [root / "grounding_eval" / name for name in ("text.py", "schema.py", "answerability.py")]
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(source_paths)}
    return {
        "schema": "excerpt_answerability_v1", "evidence_type": "deterministic_non_llm_analysis",
        "scope": "Fixed 68-excerpt index, unchanged lexical scorer, single excerpt with source metadata; no language model evaluated.",
        "limitations": [
            "A score pass is not proof of semantic correctness; numeric-only keys can accept unrelated excerpts.",
            "This is an exhaustive in-sample score ceiling, not a bound for full documents, multi-excerpt synthesis, or language models.",
            "Lexical omissions can be genuine missing information or unrecognized wording; this analysis does not adjudicate them.",
            "Score thresholds are descriptive and chosen on these same items, not validated deployment policies.",
            "Cosine similarity is not a correctness probability merely because it is bounded to [0,1].",
            "The random_floor baseline reads the answer key and selects between prepared correct and adjacent-wrong answers; it is not an information-matched comparator for excerpt retrieval.",
        ],
        "scoring_config": ScoringConfig().describe(), "corpus_digest": corpus_digest(corpus),
        "input_sha256": hashes, "answer_space": space, "baselines": baseline_rows,
        "excerpts": [{
            "item_id": item.id, "domain": item.domain, "clause": item.source.clause,
            "section_title": item.source.section_title, "anchor_text": item.source.anchor_text,
            "wrapped_text": excerpt_text(item), "source_url": item.source.url,
        } for item in corpus],
        "score_validity_examples": [
            {
                "question_id": "hc-005", "excerpt_id": excerpt_id,
                "kind": "numeric_referent_counterexample",
                "claim": explanation,
                "diagnostics": diagnose(corpus.by_id("hc-005"), excerpt_text(corpus.by_id(excerpt_id))),
                "audit_check": "Does this source sentence answer how many hazard pictograms are designated, or does eight refer to a different quantity?",
            }
            for excerpt_id, explanation in (
                ("hn-001", "The eight in this noise-exposure excerpt denotes hours, not a count of hazard pictograms; the numeric-only pictogram key still passes it."),
                ("rk-001", "The eight in this fatality-reporting excerpt denotes hours, not a count of hazard pictograms; the numeric-only pictogram key still passes it."),
            )
        ] + [{
            "question_id": "psm-007", "excerpt_id": "psm-007",
            "kind": "partial_answer_passes_threshold",
            "claim": "Source metadata raises coverage to three of four groups. The excerpt does not explicitly answer the requested contrast with incident-investigation report retention, but the default 75% threshold accepts it.",
            "diagnostics": diagnose(corpus.by_id("psm-007"), excerpt_text(corpus.by_id("psm-007"))),
            "audit_check": "Can you point to text in the excerpt that explicitly distinguishes accident-history inclusion from incident-report retention? This is a partial-answer scoring example, not an adjudicated false claim.",
        }],
    }


def render_markdown(result: dict) -> str:
    space = result["answer_space"]
    n, attainable = space["n_factual"], space["n_attainable_wrapped"]
    lines = [
        "# The excerpt ceiling: measure the answer space before blaming the ranker", "",
        "**No language model was evaluated.** This is an exhaustive analysis of the committed lexical retrieval baselines and their finite answer space.", "",
        "## Finding", "",
        f"Only **{attainable}/{n} factual questions ({attainable/n:.1%})** have any scorer-passing answer among the {space['n_indexed_excerpts']} indexed single-excerpt responses. We tested all **{space['n_combinations_per_format']:,} question/excerpt combinations** under the unchanged scoring rule. Supplying each question's designated excerpt already reaches this ceiling.", "",
        "| Retriever | Passes | Recoverable by reranking | No indexed excerpt passes | Retrieves own excerpt but fails | Both members pass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["baselines"]:
        lines.append(f"| {row['adapter']} | {row['n_pass']}/{n} | {row['n_recoverable_by_reranking']} | {row['n_no_excerpt_passes']} | {row['n_self_match_but_fail']} | {row['pairs']['both_pass']}/{row['pairs']['n_pairs']} |")
    lines += [
        "", "Reranking can recover the middle column, but cannot make the final failure class pass without changing the answer space or scorer. These are exhaustive counts for this corpus; no population inference or significance claim is needed.", "",
        "## The pair constraint has its own ceiling", "",
        f"Even an answer-key-informed selector can make both members pass for only **{space['pair_ceiling']['both_pass']}/{space['pair_ceiling']['n_pairs']} pairs**. An observed 3/23 joint score therefore mixes ranking failures with an answer-space limitation. It cannot on its own establish entity confusion or a language-model failure.", "",
        "## The measuring instrument matters", "",
        f"Removing source titles and clause metadata reduces the attainable score from {attainable}/{n} to **{space['n_attainable_bare']}/{n}**. The designated-excerpt passes that depend on metadata are: {', '.join(space['metadata_dependent_own_pass_ids'])}.", "",
        "For `cs-004`, the anchor contains 19.5% and 23.5%, while the clause metadata supplies the phrases 'oxygen deficient' and 'oxygen enriched'. For `psm-007`, the clause identifier supplies a third required group, reaching the 75% coverage threshold while the requested contrast with incident-report retention remains absent. Metadata is part of the evaluated answer, so this is format sensitivity, not automatically an error.", "",
        "A sharper counterexample is `hc-005`: its numeric-only key asks for eight pictograms, but excerpts `hn-001` (noise exposure duration) and `rk-001` (fatality reporting interval) also pass. Both contain eight. The lexical scorer does not establish what the number refers to. The ceiling is consequently a **score ceiling**, not a ceiling on factually correct answers.", "",
        "## Scores do not certify an answer", "",
        "The JSON contains every achievable score threshold, with counts of answered questions, score passes, complete pairs, fractured pairs, and both-pass pairs. Equal scores enter together. A pair with only one member answered is not a demonstrated distinction.", "",
        "| Retriever | First threshold reaching 25% coverage | Passes / answered | Complete pairs | Both-pass pairs |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result["baselines"]:
        point = next(p for p in row["threshold_curve"] if p["coverage"] >= .25)
        lines.append(f"| {row['adapter']} | {point['coverage']:.1%} | {point['n_pass']}/{point['n_answered']} | {point['complete_pairs']} | {point['both_pass_pairs']} |")
    lines += [
        "", "Cosine similarity may rank answers without estimating their probability of correctness. Being bounded between zero and one does not establish probability semantics. Published Brier/ECE values for raw cosine should be read only as a hypothetical score-as-probability diagnostic.", "",
        "## Reproduce and inspect", "",
        "```sh", "python3 -m grounding_eval.answerability", "python3 -m grounding_eval.answerability --check", "python3 -m unittest tests.test_answerability -v", "```", "",
        "`results/answerability.json` contains the per-item passing excerpt IDs, designated-excerpt diagnostics, retrieved IDs, failure decomposition, pair counts, threshold curves, scoring configuration, corpus digest and SHA-256 hashes of input artifacts and scoring code. The generator writes an identical public copy to `docs/evidence-audit-data.json`; `--check` verifies both JSON files and this report. Runs are rejected if mock, duplicated, stale, incompatible with the answer format, or inconsistent with rescoring. The complete corpus is indexed; only the 63 factual items enter question-level denominators. The five category-error questions remain excluded from factual accuracy.", "",
        "## Limits", "",
    ]
    lines += ["- " + limit for limit in result["limitations"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = analyse()
    outputs = {
        ROOT / "results/answerability.json": json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        ROOT / "results/answerability.md": render_markdown(result),
    }
    outputs[ROOT / "docs/evidence-audit-data.json"] = outputs[ROOT / "results/answerability.json"]
    for path, content in outputs.items():
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise SystemExit("stale analysis; run python3 -m grounding_eval.answerability")
        else:
            path.write_text(content)
    print("Answerability artifacts " + ("verified" if args.check else "written") + ".")


if __name__ == "__main__":
    main()
