"""Statistics for a paired, clustered benchmark design.

Two design facts govern everything in this module.

**The design is paired.** Every item is answered by every arm, and most items
belong to a minimal pair that differs only in the entity asked about. Comparing
two arms with an independent-samples test throws away the pairing and answers a
question nobody asked. The comparison implemented here is McNemar's exact test on
the discordant pairs.

**The observations are clustered.** Two members of a minimal pair are not
independent draws: they share a question stem, a source section and an author.
Bootstrapping over items would therefore understate the uncertainty. The
bootstrap here resamples *families* (the cluster unit) with replacement and
recomputes the statistic on the resampled clusters.

Everything is implemented on the standard library plus numpy. There is no scipy
in this environment, so the exact binomial tail is summed directly; that is exact
rather than approximate, and at the sample sizes involved it is cheap.

Nothing in this module estimates a probability of injury, and none of it should
be read as calibration evidence. It compares arms on a fixed item set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "McNemarResult",
    "ProportionCI",
    "BootstrapResult",
    "mcnemar_exact",
    "wilson_interval",
    "cluster_bootstrap_ci",
    "paired_accuracy",
    "risk_coverage_curve",
]


# --------------------------------------------------------------------------- #
# Exact binomial machinery
# --------------------------------------------------------------------------- #


def _binom_pmf(k: int, n: int, p: float = 0.5) -> float:
    """Exact binomial pmf via :func:`math.comb`, avoiding any scipy dependency."""
    return math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))


def _binom_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Two-sided exact binomial p-value by the method of small probabilities.

    Sums the probability of every outcome no more likely than the observed one.
    For p = 0.5 this coincides with doubling the smaller tail except for ties,
    which it handles correctly.
    """
    if n == 0:
        return 1.0
    observed = _binom_pmf(k, n, p)
    tol = observed * (1 + 1e-9)
    total = sum(_binom_pmf(i, n, p) for i in range(n + 1) if _binom_pmf(i, n, p) <= tol)
    return min(1.0, total)


@dataclass(frozen=True)
class McNemarResult:
    """Outcome of a paired binary comparison between two arms.

    n_both, n_neither
        Concordant counts. They carry no information about a difference and are
        reported only so the reader can see the whole table.
    n_a_only, n_b_only
        Discordant counts. These are the entire basis of the test.
    p_value
        Exact two-sided p-value from the binomial distribution on the discordant
        pairs. Exact rather than chi-square because the discordant count is often
        small on a 68-item corpus, which is exactly where the chi-square
        approximation misbehaves.
    """

    n_items: int
    n_both: int
    n_a_only: int
    n_b_only: int
    n_neither: int
    p_value: float

    @property
    def n_discordant(self) -> int:
        return self.n_a_only + self.n_b_only

    def describe(self) -> Dict[str, object]:
        return {
            "test": "mcnemar_exact_binomial",
            "n_items": self.n_items,
            "n_both": self.n_both,
            "n_a_only": self.n_a_only,
            "n_b_only": self.n_b_only,
            "n_neither": self.n_neither,
            "n_discordant": self.n_discordant,
            "p_value": self.p_value,
        }


def mcnemar_exact(a_success: Sequence[bool], b_success: Sequence[bool]) -> McNemarResult:
    """McNemar's exact test for two arms evaluated on the same ordered items.

    Both sequences must be aligned index for index on the same items. Raises
    :class:`ValueError` otherwise, because a silent misalignment here would
    produce a plausible number from mismatched data.
    """
    if len(a_success) != len(b_success):
        raise ValueError(
            "paired comparison needs equal-length aligned sequences; got %d and %d"
            % (len(a_success), len(b_success))
        )
    both = a_only = b_only = neither = 0
    for a, b in zip(a_success, b_success):
        if a and b:
            both += 1
        elif a and not b:
            a_only += 1
        elif b and not a:
            b_only += 1
        else:
            neither += 1
    p = _binom_two_sided_p(a_only, a_only + b_only, 0.5)
    return McNemarResult(
        n_items=len(a_success),
        n_both=both,
        n_a_only=a_only,
        n_b_only=b_only,
        n_neither=neither,
        p_value=p,
    )


# --------------------------------------------------------------------------- #
# Interval estimates
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProportionCI:
    point: float
    low: float
    high: float
    n: int
    method: str

    def describe(self) -> Dict[str, object]:
        return {
            "point": self.point,
            "ci_low": self.low,
            "ci_high": self.high,
            "n": self.n,
            "method": self.method,
        }


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> ProportionCI:
    """Wilson score interval for a binomial proportion.

    Used for single-arm rates. It is preferred to the normal approximation
    because rates near 0 and near 1 are common on this corpus and the Wald
    interval misbehaves badly there, including producing bounds outside [0, 1].

    This interval treats items as independent, which they are not. It is reported
    for orientation; the clustered bootstrap is the interval to quote when
    clustering matters.
    """
    if n <= 0:
        return ProportionCI(point=float("nan"), low=float("nan"), high=float("nan"), n=0, method="wilson")
    p = successes / n
    denom = 1.0 + (z ** 2) / n
    centre = (p + (z ** 2) / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + (z ** 2) / (4 * n * n))
    return ProportionCI(
        point=p,
        low=max(0.0, centre - half),
        high=min(1.0, centre + half),
        n=n,
        method="wilson",
    )


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    low: float
    high: float
    n_clusters: int
    n_resamples: int
    seed: int
    method: str = "cluster_bootstrap_percentile"

    def describe(self) -> Dict[str, object]:
        return {
            "point": self.point,
            "ci_low": self.low,
            "ci_high": self.high,
            "n_clusters": self.n_clusters,
            "n_resamples": self.n_resamples,
            "seed": self.seed,
            "method": self.method,
        }


def cluster_bootstrap_ci(
    values: Sequence[float],
    clusters: Sequence[str],
    statistic: Optional[Callable[[np.ndarray], float]] = None,
    n_resamples: int = 2000,
    seed: int = 20260904,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Percentile bootstrap CI resampling whole clusters with replacement.

    ``values`` is one number per item (typically 1.0/0.0 for a binary outcome) and
    ``clusters`` is the cluster label for each item, index-aligned. Items sharing
    a label are resampled together, so a minimal pair either enters a replicate
    whole or not at all.

    The default statistic is the mean. The seed is recorded in the result so a
    published interval can be regenerated exactly.
    """
    if len(values) != len(clusters):
        raise ValueError(
            "values and clusters must be aligned; got %d and %d" % (len(values), len(clusters))
        )
    stat = statistic or (lambda arr: float(np.mean(arr)))
    if not values:
        return BootstrapResult(float("nan"), float("nan"), float("nan"), 0, n_resamples, seed)

    grouped: Dict[str, List[float]] = {}
    for value, cluster in zip(values, clusters):
        grouped.setdefault(cluster, []).append(float(value))
    labels = sorted(grouped)
    arrays = [np.asarray(grouped[label], dtype=float) for label in labels]

    observed = stat(np.concatenate(arrays))
    rng = np.random.default_rng(seed)
    n_clusters = len(labels)
    replicates = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        picks = rng.integers(0, n_clusters, size=n_clusters)
        replicates[i] = stat(np.concatenate([arrays[j] for j in picks]))

    low, high = np.percentile(replicates, [100 * alpha / 2.0, 100 * (1 - alpha / 2.0)])
    return BootstrapResult(
        point=float(observed),
        low=float(low),
        high=float(high),
        n_clusters=n_clusters,
        n_resamples=n_resamples,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# Design-specific statistics
# --------------------------------------------------------------------------- #


def paired_accuracy(
    success_by_item: Mapping[str, bool],
    pairs: Iterable[Tuple[str, str]],
) -> Tuple[int, int]:
    """Count minimal pairs where BOTH members were answered correctly.

    Returns ``(n_pairs_both_correct, n_pairs_evaluated)``.

    This is the benchmark's primary discrimination statistic, following the
    scoring discipline used for minimal-pair retrieval evaluation: a system that
    answers every question with generic on-topic content can score well on
    per-item accuracy while getting the *distinction* wrong every time. Requiring
    both members of a pair strips that strategy of its reward.

    Pairs where either member has no recorded outcome are skipped rather than
    counted as failures, and the denominator returned reflects that.
    """
    both = 0
    evaluated = 0
    for a_id, b_id in pairs:
        if a_id not in success_by_item or b_id not in success_by_item:
            continue
        evaluated += 1
        if success_by_item[a_id] and success_by_item[b_id]:
            both += 1
    return both, evaluated


def risk_coverage_curve(
    confidences: Sequence[Optional[float]],
    correct: Sequence[bool],
) -> List[Dict[str, float]]:
    """Selective-prediction curve: accuracy as a function of coverage.

    Items are sorted by reported confidence, descending, and the curve reports
    accuracy over the most-confident fraction answered. For a safety application
    the useful summary is not raw accuracy but the coverage a system can sustain
    at an acceptable precision, so this is the shape to look at rather than a
    single accuracy number.

    Items with no confidence are excluded and the returned points say how many
    were used. If a system reports no confidence at all the curve is empty, which
    is itself a finding about that system.
    """
    usable = [(c, bool(ok)) for c, ok in zip(confidences, correct) if c is not None]
    if not usable:
        return []
    usable.sort(key=lambda t: -t[0])
    out: List[Dict[str, float]] = []
    hits = 0
    total = len(usable)
    for i, (conf, ok) in enumerate(usable, start=1):
        hits += 1 if ok else 0
        out.append(
            {
                "n_answered": float(i),
                "coverage": i / total,
                "accuracy": hits / i,
                "confidence_threshold": float(conf),
            }
        )
    return out
