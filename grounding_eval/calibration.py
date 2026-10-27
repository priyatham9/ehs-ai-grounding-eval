"""Calibration metrics: Brier score, expected calibration error, reliability bins.

Selective scoring (``selective.py``) asks whether a confidence ranks correct
answers above incorrect ones. Calibration asks a different question: when the
system reports confidence ``c``, is it correct about a fraction ``c`` of the
time? Both need a per-answer confidence and a correct / incorrect flag.

Definitions.
* Brier score: mean of ``(c - y)^2`` with ``y`` 1 for a correct answer and 0 for
  an incorrect one (Brier, Monthly Weather Review 78(1), 1950). Lower is better.
  ``brier_constant`` is the Brier score of always reporting the observed accuracy
  ``p``, which is ``p * (1 - p)``; a score above it is worse than ignoring the
  input and reporting the base rate.
* Expected calibration error (ECE): confidences are grouped into ``n_bins``
  equal-width bins over [0, 1]; ECE is the sum over bins of
  ``(n_bin / n) * |accuracy_bin - mean_confidence_bin|`` (Naeini, Cooper and
  Hauskrecht, AAAI 2015; the equal-width form of Guo et al., ICML 2017). The
  last bin includes 1.0. ECE depends on the number of bins and is biased on small
  samples; the bin count is reported with every value.
* Reliability table: one row per bin with its count, mean confidence and observed
  accuracy.

Domain. These metrics treat the confidence as a probability, so every
confidence must lie in [0, 1]; a value outside raises ``ValueError`` rather
than being clipped or rescaled. Whether a given system's score is on a
probability scale is a property of the system, not something this module can
establish. Items with no confidence are dropped, as in ``selective.py``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .selective import _clean, cluster_bootstrap_selective

__all__ = [
    "brier_score",
    "brier_constant",
    "reliability_table",
    "expected_calibration_error",
    "summarize_calibration",
]

N_BINS = 10


def _checked(confidences: Sequence[Optional[float]], correct: Sequence[bool]) -> List[Tuple[float, bool]]:
    pairs = _clean(confidences, correct)
    for c, _ in pairs:
        if not 0.0 <= c <= 1.0:
            raise ValueError("calibration needs confidences in [0, 1]; got %r" % c)
    return pairs


def brier_score(confidences: Sequence[Optional[float]], correct: Sequence[bool]) -> float:
    """Mean squared difference between confidence and outcome; NaN with no confidence."""
    pairs = _checked(confidences, correct)
    if not pairs:
        return float("nan")
    return sum((c - (1.0 if ok else 0.0)) ** 2 for c, ok in pairs) / len(pairs)


def brier_constant(confidences: Sequence[Optional[float]], correct: Sequence[bool]) -> float:
    """Brier score of reporting the observed accuracy for every item: p * (1 - p)."""
    pairs = _clean(confidences, correct)
    if not pairs:
        return float("nan")
    p = sum(1 for _, ok in pairs if ok) / len(pairs)
    return p * (1.0 - p)


def _bin_index(c: float, n_bins: int) -> int:
    return min(int(c * n_bins), n_bins - 1)


def reliability_table(
    confidences: Sequence[Optional[float]], correct: Sequence[bool], n_bins: int = N_BINS
) -> List[Dict[str, float]]:
    """Per-bin count, mean confidence and accuracy over ``n_bins`` equal-width bins.

    Empty bins are kept with count 0 and NaN means so that every table has the
    same rows.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    pairs = _checked(confidences, correct)
    sums = [[0, 0.0, 0] for _ in range(n_bins)]
    for c, ok in pairs:
        s = sums[_bin_index(c, n_bins)]
        s[0] += 1
        s[1] += c
        s[2] += 1 if ok else 0
    rows = []
    for i, (n, csum, hits) in enumerate(sums):
        rows.append({
            "bin": i,
            "low": i / n_bins,
            "high": (i + 1) / n_bins,
            "count": n,
            "mean_confidence": csum / n if n else float("nan"),
            "accuracy": hits / n if n else float("nan"),
        })
    return rows


def expected_calibration_error(
    confidences: Sequence[Optional[float]], correct: Sequence[bool], n_bins: int = N_BINS
) -> float:
    """Equal-width-bin ECE; NaN with no confidence."""
    pairs = _checked(confidences, correct)
    if not pairs:
        return float("nan")
    total = 0.0
    for r in reliability_table(confidences, correct, n_bins):
        if r["count"]:
            total += r["count"] / len(pairs) * abs(r["accuracy"] - r["mean_confidence"])
    return total


def summarize_calibration(
    confidences: Sequence[Optional[float]],
    correct: Sequence[bool],
    clusters: Sequence[str],
    n_bins: int = N_BINS,
    n_resamples: int = 2000,
    seed: int = 20260904,
) -> Dict[str, object]:
    """Headline calibration numbers for one system on one item set."""
    nan = float("nan")
    n_with = sum(1 for c in confidences if c is not None)
    row: Dict[str, object] = {
        "n_items": len(correct),
        "n_with_confidence": n_with,
        "n_bins": n_bins,
        "accuracy": sum(bool(c) for c in correct) / len(correct) if len(correct) else nan,
    }
    keys = ("mean_confidence", "brier", "brier_constant", "ece")
    if n_with == 0:
        for k in keys:
            row[k] = nan
        for k in ("brier", "ece"):
            row[k + "_ci_low"] = row[k + "_ci_high"] = nan
        return row
    pairs = _checked(confidences, correct)
    row["mean_confidence"] = sum(c for c, _ in pairs) / len(pairs)
    row["brier_constant"] = brier_constant(confidences, correct)
    for key, stat in (
        ("brier", brier_score),
        ("ece", lambda c, o: expected_calibration_error(c, o, n_bins)),
    ):
        point, lo, hi = cluster_bootstrap_selective(confidences, correct, clusters, stat, n_resamples, seed)
        row[key], row[key + "_ci_low"], row[key + "_ci_high"] = point, lo, hi
    return row
