"""Export a blinded sample of scored responses for human adjudication.

Why this module exists
----------------------
The automatic scorer is lexical, and its assertion-versus-mention heuristic
misfires in both directions. The adjacent-substitution rate is the benchmark's
headline error class and is therefore the number most exposed to scorer error.

The preregistered plan (``docs/preregistration.md``) requires that any published
adjacent-substitution rate be checked against a human-adjudicated subsample, with
the human figure reported as primary and the automatic score treated as a screen.
This module produces that sample.

Blinding
--------
The exported sheet deliberately omits the arm, the adapter name and the automatic
outcome. A rater who can see that a response came from the "ungrounded" arm, or
that the scorer already called it a substitution, is not an independent check on
the scorer. The mapping back to those fields is written to a separate key file
that the rater does not open.

The sample is stratified by automatic outcome so that the classes of interest are
represented, and shuffled with a recorded seed so the order carries no signal.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .corpus import Corpus
from .harness import RunResult
from .schema import Outcome

__all__ = ["AdjudicationSample", "build_sample", "write_sample", "agreement"]

#: Outcomes worth adjudicating, and the default share of each to sample.
#: Correct responses are sampled too: checking only the errors would measure the
#: scorer's false-positive rate while leaving its false-negative rate unknown.
DEFAULT_STRATA: Dict[str, float] = {
    Outcome.ADJACENT_SUBSTITUTION.value: 0.2,
    Outcome.CORRECT.value: 0.2,
    Outcome.OTHER_INCORRECT.value: 0.2,
    Outcome.ABSTAINED.value: 0.2,
}


@dataclass(frozen=True)
class AdjudicationSample:
    """A blinded rating sheet plus the key needed to score it afterwards."""

    sheet: Tuple[Dict[str, object], ...]
    key: Tuple[Dict[str, object], ...]
    seed: int
    strata: Dict[str, float]

    def __len__(self) -> int:
        return len(self.sheet)


def build_sample(
    runs: Sequence[RunResult],
    corpus: Corpus,
    strata: Optional[Dict[str, float]] = None,
    seed: int = 20260904,
) -> AdjudicationSample:
    """Draw a blinded, stratified sample of responses for human rating.

    Each sheet row carries the question, the response text, the reference answer
    and the governing clause, which is everything a rater needs. It does not carry
    the arm, the adapter name, or the automatic outcome.
    """
    shares = dict(strata or DEFAULT_STRATA)
    by_item = {item.id: item for item in corpus}

    pools: Dict[str, List[Dict[str, object]]] = {}
    for run in runs:
        responses = {r.item_id: r for r in run.responses}
        for score in run.scores:
            response = responses.get(score.item_id)
            item = by_item.get(score.item_id)
            if response is None or item is None:
                continue
            pools.setdefault(score.outcome.value, []).append(
                {
                    "adapter": run.adapter.get("name"),
                    "arm": run.adapter.get("arm"),
                    "repeat": run.repeat,
                    "item_id": item.id,
                    "automatic_outcome": score.outcome.value,
                    "question": item.question,
                    "response_text": response.text,
                    "reference_answer": item.correct_answer,
                    "clause": item.source.clause,
                    "adjacent_label": item.adjacent_wrong.label,
                }
            )

    rng = random.Random(seed)
    selected: List[Dict[str, object]] = []
    for outcome, share in sorted(shares.items()):
        pool = pools.get(outcome, [])
        if not pool:
            continue
        take = max(1, int(round(len(pool) * share)))
        selected.extend(rng.sample(pool, min(take, len(pool))))

    rng.shuffle(selected)

    sheet: List[Dict[str, object]] = []
    key: List[Dict[str, object]] = []
    for index, record in enumerate(selected):
        rating_id = "adj-%04d" % index
        sheet.append(
            {
                "rating_id": rating_id,
                "question": record["question"],
                "system_response": record["response_text"],
                "reference_answer": record["reference_answer"],
                "governing_clause": record["clause"],
                "entity_that_would_be_the_wrong_one": record["adjacent_label"],
                "rating": "",
                "rater_notes": "",
                "instructions": (
                    "Choose one: correct | adjacent_substitution | other_incorrect | "
                    "abstained | unscorable. Judge the response against the "
                    "governing clause, not against the reference wording. Mark "
                    "adjacent_substitution only when the response answers about "
                    "the wrong entity named above; a response that merely names "
                    "that entity in order to rule it out is not a substitution."
                ),
            }
        )
        key.append(
            {
                "rating_id": rating_id,
                "item_id": record["item_id"],
                "adapter": record["adapter"],
                "arm": record["arm"],
                "repeat": record["repeat"],
                "automatic_outcome": record["automatic_outcome"],
            }
        )

    return AdjudicationSample(
        sheet=tuple(sheet), key=tuple(key), seed=seed, strata=shares
    )


def write_sample(sample: AdjudicationSample, directory: str) -> Tuple[str, str]:
    """Write the rating sheet and the key to separate files.

    Two files, not one, so that opening the sheet cannot accidentally reveal the
    arm or the automatic score.
    """
    os.makedirs(directory, exist_ok=True)
    sheet_path = os.path.join(directory, "adjudication_sheet.json")
    key_path = os.path.join(directory, "adjudication_key.json")

    with open(sheet_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "note": (
                    "Blinded rating sheet. Do not open adjudication_key.json until "
                    "every rating field is filled in."
                ),
                "n_items": len(sample.sheet),
                "ratings": list(sample.sheet),
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    with open(key_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "note": "Unblinding key. Open only after rating is complete.",
                "seed": sample.seed,
                "strata": sample.strata,
                "key": list(sample.key),
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    return sheet_path, key_path


def agreement(
    human: Dict[str, str],
    automatic: Dict[str, str],
) -> Dict[str, object]:
    """Compare human ratings with automatic outcomes on the same rating ids.

    Returns raw agreement and Cohen's kappa. Kappa is reported because raw
    agreement is inflated when one outcome class dominates, which it does here.

    This measures agreement between a human and the scorer. It is not
    inter-rater reliability; that requires two humans and should be reported
    separately when more than one rater is used.
    """
    shared = sorted(set(human) & set(automatic))
    if not shared:
        return {"n": 0, "raw_agreement": float("nan"), "cohens_kappa": float("nan")}

    n = len(shared)
    observed = sum(1 for rid in shared if human[rid] == automatic[rid]) / n

    labels = sorted({human[r] for r in shared} | {automatic[r] for r in shared})
    expected = 0.0
    for label in labels:
        p_human = sum(1 for r in shared if human[r] == label) / n
        p_auto = sum(1 for r in shared if automatic[r] == label) / n
        expected += p_human * p_auto

    kappa = (observed - expected) / (1 - expected) if expected < 1.0 else float("nan")
    disagreements = [
        {"rating_id": rid, "human": human[rid], "automatic": automatic[rid]}
        for rid in shared
        if human[rid] != automatic[rid]
    ]
    return {
        "n": n,
        "raw_agreement": observed,
        "expected_agreement": expected,
        "cohens_kappa": kappa,
        "n_disagreements": len(disagreements),
        "disagreements": disagreements,
        "note": (
            "Agreement between one human rater and the automatic scorer. Not "
            "inter-rater reliability. Where they disagree, the human rating is "
            "the primary figure; see docs/preregistration.md."
        ),
    }
