"""Answer-or-abstain scoring: risk-coverage curves and selective accuracy.

A system that can say "I do not know" is evaluated by what it does when allowed
to decline. Given a confidence score per answer, a threshold turns the system
into a selective classifier: answer the items whose confidence is at or above
the threshold, abstain on the rest. Coverage is the fraction answered; selective
risk is the error rate on the answered items; selective accuracy is one minus
that (El-Yaniv and Wiener, JMLR 2010; Geifman and El-Yaniv, NeurIPS 2017).

This module differs from ``stats.risk_coverage_curve`` in three ways: ties in
confidence are handled explicitly (the retrievers' scores are rounded to four
decimals and do tie), the curve is summarised by the area under the
risk-coverage curve (AURC) and its excess over the best possible ordering
(E-AURC; Geifman, Uziel and El-Yaniv, ICLR 2019), and every summary carries a
cluster-bootstrap interval that resamples item families.

Tie handling. Items with equal confidence cannot be separated by any threshold,
so they enter the answered set together. For the AURC the prefix of size ``k``
is evaluated at its expected number of correct answers when the order inside a
tie group is random: a group with ``m`` items of which ``h`` are correct,
preceded by ``H`` correct answers, contributes ``H + (k - start) * h / m`` at
prefix size ``k`` inside the group. For selective accuracy at a coverage target
the threshold is applied literally, so the reported coverage can exceed the target.

Nothing here estimates a probability of injury, and a confidence score from a
retriever is a similarity, not a calibrated probability. These metrics measure
how well a score ranks correct answers above incorrect ones, not whether the score
is a probability.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "risk_coverage_points",
    "aurc",
    "aurc_random",
    "aurc_optimal",
    "e_aurc",
    "selective_accuracy_at_coverage",
    "summarize_selective",
    "cluster_bootstrap_selective",
]


def _clean(confidences: Sequence[Optional[float]], correct: Sequence[bool]) -> List[Tuple[float, bool]]:
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be aligned")
    out = []
    for c, ok in zip(confidences, correct):
        if c is None or (isinstance(c, float) and math.isnan(c)):
            continue
        out.append((float(c), bool(ok)))
    return out


def _tie_groups(pairs: Sequence[Tuple[float, bool]]) -> List[Tuple[float, int, int]]:
    """(confidence, group size, correct count), highest confidence first."""
    groups: Dict[float, List[int]] = {}
    for c, ok in pairs:
        g = groups.setdefault(c, [0, 0])
        g[0] += 1
        g[1] += 1 if ok else 0
    return [(c, g[0], g[1]) for c, g in sorted(groups.items(), key=lambda kv: -kv[0])]


def risk_coverage_points(
    confidences: Sequence[Optional[float]], correct: Sequence[bool]
) -> List[Dict[str, float]]:
    """One point per distinct confidence value, answering everything at or above it.

    Items with no confidence are dropped; an empty list means the system reported
    no usable confidence.
    """
    pairs = _clean(confidences, correct)
    n = len(pairs)
    points: List[Dict[str, float]] = []
    answered = hits = 0
    for conf, size, good in _tie_groups(pairs):
        answered += size
        hits += good
        points.append({
            "threshold": conf,
            "n_answered": float(answered),
            "coverage": answered / n,
            "selective_accuracy": hits / answered,
            "selective_risk": 1.0 - hits / answered,
        })
    return points


def aurc(confidences: Sequence[Optional[float]], correct: Sequence[bool]) -> float:
    """Mean selective risk over prefix sizes 1..n, ties at their random-order expectation.

    NaN when no item has a confidence.
    """
    pairs = _clean(confidences, correct)
    n = len(pairs)
    if n == 0:
        return float("nan")
    total = 0.0
    before_n = before_hits = 0
    for _, size, good in _tie_groups(pairs):
        rate = good / size
        for j in range(1, size + 1):
            k = before_n + j
            total += 1.0 - (before_hits + j * rate) / k
        before_n += size
        before_hits += good
    return total / n


def aurc_random(correct: Sequence[bool]) -> float:
    """AURC of an uninformative score: the overall error rate."""
    flags = [bool(c) for c in correct]
    return 1.0 - sum(flags) / len(flags) if flags else float("nan")


def aurc_optimal(correct: Sequence[bool]) -> float:
    """AURC of the best possible ordering: every correct answer ranked first."""
    flags = [bool(c) for c in correct]
    n = len(flags)
    if n == 0:
        return float("nan")
    h = sum(flags)
    total = sum((k - h) / k for k in range(h + 1, n + 1))
    return total / n


def e_aurc(confidences: Sequence[Optional[float]], correct: Sequence[bool]) -> float:
    """Excess AURC: AURC minus the optimal-ordering AURC, over the same items."""
    pairs = _clean(confidences, correct)
    if not pairs:
        return float("nan")
    flags = [ok for _, ok in pairs]
    return aurc(confidences, correct) - aurc_optimal(flags)


def selective_accuracy_at_coverage(
    confidences: Sequence[Optional[float]], correct: Sequence[bool], coverage: float
) -> Tuple[float, float]:
    """Accuracy on the answered set at the first threshold reaching ``coverage``.

    Returns ``(selective_accuracy, achieved_coverage)``; achieved coverage can
    exceed the target because tied items cannot be split. ``(nan, nan)`` when no
    confidences exist.
    """
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    points = risk_coverage_points(confidences, correct)
    for p in points:
        if p["coverage"] >= coverage - 1e-12:
            return p["selective_accuracy"], p["coverage"]
    return float("nan"), float("nan")


def cluster_bootstrap_selective(
    confidences: Sequence[Optional[float]],
    correct: Sequence[bool],
    clusters: Sequence[str],
    statistic,
    n_resamples: int = 2000,
    seed: int = 20260904,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Percentile cluster bootstrap of ``statistic(confidences, correct)``.

    Item families are resampled whole, as in ``stats.cluster_bootstrap_ci``.
    Replicates where the statistic is NaN are dropped. Returns (point, low, high).
    """
    triples = [(c, ok, f) for c, ok, f in zip(confidences, correct, clusters) if c is not None]
    if not triples:
        nan = float("nan")
        return nan, nan, nan
    grouped: Dict[str, List[Tuple[float, bool]]] = {}
    for c, ok, f in triples:
        grouped.setdefault(f, []).append((c, bool(ok)))
    labels = sorted(grouped)
    point = statistic([c for g in labels for c, _ in grouped[g]], [ok for g in labels for _, ok in grouped[g]])
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_resamples):
        picks = rng.integers(0, len(labels), size=len(labels))
        cs = [c for j in picks for c, _ in grouped[labels[j]]]
        oks = [ok for j in picks for _, ok in grouped[labels[j]]]
        v = statistic(cs, oks)
        if not math.isnan(v):
            reps.append(v)
    if not reps:
        return point, float("nan"), float("nan")
    low, high = np.percentile(reps, [100 * alpha / 2.0, 100 * (1 - alpha / 2.0)])
    return float(point), float(low), float(high)


COVERAGE_TARGETS = (0.25, 0.5, 0.75)


def summarize_selective(
    confidences: Sequence[Optional[float]],
    correct: Sequence[bool],
    clusters: Sequence[str],
    n_resamples: int = 2000,
    seed: int = 20260904,
) -> Dict[str, object]:
    """Headline answer-or-abstain numbers for one system on one item set."""
    n = len(correct)
    with_conf = sum(1 for c in confidences if c is not None)
    row: Dict[str, object] = {
        "n_items": n,
        "n_with_confidence": with_conf,
        "n_distinct_confidences": len({c for c in confidences if c is not None}),
        "accuracy_full_coverage": sum(bool(c) for c in correct) / n if n else float("nan"),
    }
    if with_conf == 0:
        nan = float("nan")
        for key in ("aurc", "aurc_random", "aurc_optimal", "e_aurc"):
            row[key] = nan
        row["aurc_ci_low"] = row["aurc_ci_high"] = nan
        for cov in COVERAGE_TARGETS:
            row["sel_acc_at_%d" % round(cov * 100)] = nan
            row["achieved_cov_at_%d" % round(cov * 100)] = nan
            row["sel_acc_at_%d_ci_low" % round(cov * 100)] = nan
            row["sel_acc_at_%d_ci_high" % round(cov * 100)] = nan
        return row
    flags = [bool(ok) for c, ok in zip(confidences, correct) if c is not None]
    point, low, high = cluster_bootstrap_selective(confidences, correct, clusters, aurc, n_resamples, seed)
    row.update({
        "aurc": point, "aurc_ci_low": low, "aurc_ci_high": high,
        "aurc_random": aurc_random(flags), "aurc_optimal": aurc_optimal(flags),
        "e_aurc": e_aurc(confidences, correct),
    })
    for cov in COVERAGE_TARGETS:
        tag = "sel_acc_at_%d" % round(cov * 100)
        acc, got = selective_accuracy_at_coverage(confidences, correct, cov)
        _, lo, hi = cluster_bootstrap_selective(
            confidences, correct, clusters,
            lambda c, o, cov=cov: selective_accuracy_at_coverage(c, o, cov)[0],
            n_resamples, seed,
        )
        row[tag] = acc
        row["achieved_cov_at_%d" % round(cov * 100)] = got
        row[tag + "_ci_low"] = lo
        row[tag + "_ci_high"] = hi
    return row
