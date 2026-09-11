"""Aggregation and presentation of scored runs.

What this module reports, and why in this order
-----------------------------------------------
1. **Outcome composition**, not a single accuracy number. An adjacent
   substitution and a plainly wrong answer have different consequences, and a
   benchmark that adds them together destroys the distinction it was built to
   measure.
2. **Paired accuracy** over minimal pairs as the primary discrimination
   statistic. Per-item accuracy rewards a system that answers every question with
   generic on-topic content; requiring both members of a pair does not.
3. **Category-error items separately.** Their correct behaviour is to decline.
   Folding them into overall accuracy would let a system that answers everything
   confidently outscore a properly calibrated one, which inverts the objective.
4. **Run-to-run spread** alongside every centre, because a single run is not a
   result.
5. **Citation grades**, so that "cited something" is never mistaken for "cited
   the right thing".

Every table is a pandas DataFrame so it can be inspected, joined and exported
rather than only printed.

The mock banner is enforced here. :func:`format_summary` raises if asked to
render a mock run without displaying the banner.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import pandas as pd

from . import MOCK_PROVENANCE
from .corpus import Corpus
from .harness import RunResult
from .schema import ItemType, Outcome
from .stats import (
    McNemarResult,
    cluster_bootstrap_ci,
    mcnemar_exact,
    paired_accuracy,
    risk_coverage_curve,
    wilson_interval,
)

__all__ = [
    "scores_frame",
    "runs_frame",
    "outcome_composition",
    "arm_summary",
    "paired_accuracy_table",
    "compare_arms",
    "citation_table",
    "format_summary",
]

#: Outcomes that count as a success for the primary statistic. Abstention is not
#: a success on an answerable item; it is credited separately and by policy.
SUCCESS_OUTCOMES = (Outcome.CORRECT,)


def scores_frame(run: RunResult) -> pd.DataFrame:
    """One row per item for a single run, with run identity attached."""
    frame = pd.DataFrame([score.to_dict() for score in run.scores])
    if frame.empty:
        return frame
    frame["adapter"] = run.adapter.get("name", "adapter")
    frame["arm"] = run.adapter.get("arm", "unknown")
    frame["repeat"] = run.repeat
    frame["is_mock"] = run.is_mock
    frame["correct"] = frame["outcome"] == Outcome.CORRECT.value
    frame["adjacent"] = frame["outcome"] == Outcome.ADJACENT_SUBSTITUTION.value
    frame["abstained_flag"] = frame["outcome"] == Outcome.ABSTAINED.value
    return frame


def runs_frame(runs: Sequence[RunResult]) -> pd.DataFrame:
    """Concatenate several runs into one long frame."""
    frames = [scores_frame(run) for run in runs]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def outcome_composition(frame: pd.DataFrame, by: Sequence[str] = ("adapter",)) -> pd.DataFrame:
    """Proportion of each outcome class, grouped by ``by``.

    Factual and category-error items are reported separately because their
    correct behaviours differ; mixing them produces a number that means nothing.
    """
    if frame.empty:
        return pd.DataFrame()
    keys = list(by) + ["item_type"]
    counts = (
        frame.groupby(keys + ["outcome"], dropna=False).size().rename("n").reset_index()
    )
    totals = counts.groupby(keys)["n"].transform("sum")
    counts["proportion"] = counts["n"] / totals
    return counts.sort_values(keys + ["outcome"]).reset_index(drop=True)


def arm_summary(
    frame: pd.DataFrame,
    corpus: Corpus,
    bootstrap_resamples: int = 2000,
    seed: int = 20260904,
) -> pd.DataFrame:
    """Headline table: one row per adapter, aggregated across repeats.

    The interval reported is a cluster bootstrap resampling item families, since
    minimal-pair members are not independent observations. A Wilson interval that
    ignores clustering is reported alongside it so the difference is visible
    rather than assumed away.
    """
    if frame.empty:
        return pd.DataFrame()

    pairs = [(a.id, b.id) for a, b in corpus.pairs()]
    rows: List[Dict[str, Any]] = []

    for adapter, group in frame.groupby("adapter"):
        factual = group[group["item_type"] == ItemType.FACTUAL.value]
        category = group[group["item_type"] == ItemType.CATEGORY_ERROR.value]

        n_correct = int(factual["correct"].sum())
        n_factual = int(len(factual))
        boot = cluster_bootstrap_ci(
            factual["correct"].astype(float).tolist(),
            factual["family"].tolist(),
            n_resamples=bootstrap_resamples,
            seed=seed,
        )
        wilson = wilson_interval(n_correct, n_factual)

        # Paired accuracy is computed per repeat and then averaged, because a
        # pair is only "both correct" within a single run.
        paired_rates: List[float] = []
        for _, run_group in factual.groupby("repeat"):
            success = dict(zip(run_group["item_id"], run_group["correct"].astype(bool)))
            both, evaluated = paired_accuracy(success, pairs)
            if evaluated:
                paired_rates.append(both / evaluated)

        per_repeat_acc = factual.groupby("repeat")["correct"].mean()

        rows.append(
            {
                "adapter": adapter,
                "arm": group["arm"].iloc[0],
                "is_mock": bool(group["is_mock"].any()),
                "n_repeats": int(group["repeat"].nunique()),
                "n_factual_obs": n_factual,
                "accuracy": n_correct / n_factual if n_factual else float("nan"),
                "accuracy_ci_low": boot.low,
                "accuracy_ci_high": boot.high,
                "accuracy_ci_method": boot.method,
                "accuracy_wilson_low": wilson.low,
                "accuracy_wilson_high": wilson.high,
                "accuracy_spread_across_repeats": (
                    float(per_repeat_acc.max() - per_repeat_acc.min()) if len(per_repeat_acc) > 1 else 0.0
                ),
                "paired_accuracy": (
                    sum(paired_rates) / len(paired_rates) if paired_rates else float("nan")
                ),
                "adjacent_substitution_rate": float(factual["adjacent"].mean()) if n_factual else float("nan"),
                "abstention_rate_factual": float(factual["abstained_flag"].mean()) if n_factual else float("nan"),
                "category_error_declined_rate": (
                    float(category["abstained_flag"].mean()) if len(category) else float("nan")
                ),
                "unscorable_rate": float((group["outcome"] == Outcome.UNSCORABLE.value).mean()),
                "citation_correct_clause_rate": float(
                    (group["citation_grade"] == "correct_clause").mean()
                ),
                "fabricated_paywalled_citation_rate": float(
                    group["unsupported_paywalled_citations"].apply(lambda v: bool(v)).mean()
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("adapter").reset_index(drop=True)


def paired_accuracy_table(frame: pd.DataFrame, corpus: Corpus) -> pd.DataFrame:
    """Per-pair outcome detail: which minimal pairs each adapter got wholly right.

    This is the table to read when a headline paired accuracy looks surprising.
    It shows the specific distinctions a system failed to make.
    """
    if frame.empty:
        return pd.DataFrame()
    pair_items = {a.id: (a, b) for a, b in corpus.pairs()}
    rows: List[Dict[str, Any]] = []
    for (adapter, repeat), group in frame.groupby(["adapter", "repeat"]):
        success = dict(zip(group["item_id"], group["correct"].astype(bool)))
        outcomes = dict(zip(group["item_id"], group["outcome"]))
        for a_id, (a_item, b_item) in pair_items.items():
            b_id = b_item.id
            if a_id not in success or b_id not in success:
                continue
            rows.append(
                {
                    "adapter": adapter,
                    "repeat": repeat,
                    "pair_id": a_item.pair_id,
                    "family": a_item.family,
                    "a_id": a_id,
                    "b_id": b_id,
                    "a_outcome": outcomes.get(a_id),
                    "b_outcome": outcomes.get(b_id),
                    "both_correct": bool(success[a_id] and success[b_id]),
                }
            )
    return pd.DataFrame(rows)


def compare_arms(
    frame: pd.DataFrame,
    adapter_a: str,
    adapter_b: str,
    repeat: int = 0,
    item_type: ItemType = ItemType.FACTUAL,
) -> Dict[str, Any]:
    """Paired comparison of two adapters on the same items in the same repeat.

    Uses McNemar's exact test. The two adapters must have answered exactly the
    same item set; a mismatch raises rather than silently comparing an
    intersection, because an unnoticed intersection is a fabricated comparison.
    """
    subset = frame[(frame["repeat"] == repeat) & (frame["item_type"] == item_type.value)]
    a_frame = subset[subset["adapter"] == adapter_a].set_index("item_id").sort_index()
    b_frame = subset[subset["adapter"] == adapter_b].set_index("item_id").sort_index()

    if a_frame.empty or b_frame.empty:
        raise ValueError(
            "no rows for %r and/or %r at repeat %d" % (adapter_a, adapter_b, repeat)
        )
    if list(a_frame.index) != list(b_frame.index):
        missing_a = sorted(set(b_frame.index) - set(a_frame.index))
        missing_b = sorted(set(a_frame.index) - set(b_frame.index))
        raise ValueError(
            "paired comparison requires identical item sets; %s is missing %s and %s is missing %s"
            % (adapter_a, missing_a[:5], adapter_b, missing_b[:5])
        )

    result: McNemarResult = mcnemar_exact(
        a_frame["correct"].astype(bool).tolist(),
        b_frame["correct"].astype(bool).tolist(),
    )
    return {
        "adapter_a": adapter_a,
        "adapter_b": adapter_b,
        "repeat": repeat,
        "item_type": item_type.value,
        "accuracy_a": float(a_frame["correct"].mean()),
        "accuracy_b": float(b_frame["correct"].mean()),
        "difference_a_minus_b": float(a_frame["correct"].mean() - b_frame["correct"].mean()),
        "mcnemar": result.describe(),
        "note": (
            "McNemar's exact test on discordant items. This compares two arms on a "
            "fixed item set; it is not evidence about any population of questions "
            "beyond this corpus."
        ),
    }


def citation_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Distribution of citation grades per adapter.

    Kept separate from accuracy on purpose. A right answer with a fabricated
    citation and a right answer with the correct clause are not the same event,
    and in a regulated setting the difference is the point.
    """
    if frame.empty:
        return pd.DataFrame()
    counts = frame.groupby(["adapter", "citation_grade"]).size().rename("n").reset_index()
    totals = counts.groupby("adapter")["n"].transform("sum")
    counts["proportion"] = counts["n"] / totals
    return counts.sort_values(["adapter", "citation_grade"]).reset_index(drop=True)


def selective_prediction_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Risk-coverage points per adapter, from reported confidence.

    An empty result for an adapter means it reported no confidence, which is
    itself worth knowing: without a confidence signal there is no way to trade
    coverage for precision, and every answer must be reviewed as if it were
    equally likely to be wrong.
    """
    if frame.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for adapter, group in frame.groupby("adapter"):
        curve = risk_coverage_curve(
            group["confidence"].tolist(),
            group["correct"].astype(bool).tolist(),
        )
        for point in curve:
            point = dict(point)
            point["adapter"] = adapter
            rows.append(point)
    return pd.DataFrame(rows)


def format_summary(
    frame: pd.DataFrame,
    corpus: Corpus,
    show_mock_banner: bool = True,
    bootstrap_resamples: int = 2000,
) -> str:
    """Render a plain-text summary suitable for a terminal or a log.

    Raises if a mock run would be rendered with the banner suppressed. That is
    deliberate: the one way this repository could mislead someone is by having a
    fixture run read as a result, so removing the banner is made impossible
    rather than merely discouraged.
    """
    if frame.empty:
        return "no scored runs"

    is_mock = bool(frame["is_mock"].any())
    if is_mock and not show_mock_banner:
        raise ValueError(
            "refusing to render a mock-adapter run without its provenance banner; "
            "these numbers are a demonstration fixture, not results"
        )

    lines: List[str] = []
    if is_mock:
        rule = "=" * 78
        lines += [
            rule,
            MOCK_PROVENANCE,
            "",
            "These numbers come from a simulated adapter that composes its answers",
            "from the corpus itself. They demonstrate that the harness runs and show",
            "the report format. They are NOT experimental results and say nothing",
            "about any real system.",
            rule,
            "",
        ]

    summary = arm_summary(frame, corpus, bootstrap_resamples=bootstrap_resamples)
    lines.append("Per-adapter summary (factual items; category-error items reported separately)")
    lines.append("")
    display_cols = [
        "adapter",
        "arm",
        "n_repeats",
        "accuracy",
        "accuracy_ci_low",
        "accuracy_ci_high",
        "paired_accuracy",
        "adjacent_substitution_rate",
        "abstention_rate_factual",
        "category_error_declined_rate",
        "citation_correct_clause_rate",
    ]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        lines.append(summary[display_cols].to_string(index=False, float_format=lambda v: "%.3f" % v))

    lines += [
        "",
        "accuracy            correct / factual items answered, pooled over repeats",
        "accuracy_ci_*       95%% cluster bootstrap over item families (%d resamples)" % bootstrap_resamples,
        "paired_accuracy     fraction of minimal pairs where BOTH members were correct",
        "adjacent_substitution_rate",
        "                    answered about the semantically adjacent wrong entity",
        "category_error_declined_rate",
        "                    fraction of false-premise items correctly declined",
        "citation_correct_clause_rate",
        "                    cited the exact governing clause, not merely a section",
        "",
        "Reported separately and never summed: an adjacent substitution is a",
        "different failure from a plainly wrong answer, and abstention on an",
        "answerable item is a different event from either.",
    ]
    return "\n".join(lines)
