"""Paired significance testing between arms, with Holm correction.

Every arm answers the same items, so arms are compared on the items they
disagree about. For each pair of arms this module reports

* the accuracy difference (A minus B) with a **paired cluster-bootstrap**
  percentile interval: the per-item difference (1, 0 or -1) is resampled by
  whole question family, so the pairing and the minimal-pair clustering are both
  kept (``stats.cluster_bootstrap_ci`` applied to the paired differences);
* the **exact McNemar** test (``stats.mcnemar_exact``) on the discordant items;
* a **Holm** step-down adjusted p-value across the whole family of comparisons.

Holm (1979) controls the family-wise error rate under any dependence between the
tests, with no assumption of independence, and is uniformly more powerful than
Bonferroni. Adjusted p-values are made monotone (a larger raw p-value never gets a
smaller adjusted one) and capped at 1.

The exact McNemar test treats items as independent. Question families (minimal
pairs) are not, so the McNemar p-value is optimistic when discordant items cluster
in families; the cluster-bootstrap interval is the cluster-aware check, and a
difference is called significant here only when the Holm-adjusted McNemar p is
below alpha. The bootstrap interval is reported beside it, not combined with it.

Nothing here is a probability of injury; it compares arms on a fixed item set.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

from .stats import cluster_bootstrap_ci, mcnemar_exact

__all__ = ["holm_adjust", "paired_difference", "compare_arms"]


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values, returned in the input order.

    Sort ascending; the i-th smallest (0-indexed) of m p-values is multiplied by
    ``m - i``; a running maximum keeps the sequence monotone; values cap at 1.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[idx]))
        adjusted[idx] = running
    return adjusted


def paired_difference(
    a_success: Sequence[bool],
    b_success: Sequence[bool],
    clusters: Sequence[str],
    n_resamples: int = 2000,
    seed: int = 20260904,
) -> Dict[str, object]:
    """Accuracy of A minus accuracy of B on aligned items, with test and interval."""
    if not (len(a_success) == len(b_success) == len(clusters)):
        raise ValueError("a_success, b_success and clusters must be aligned")
    mc = mcnemar_exact(a_success, b_success)
    diffs = [float(bool(a)) - float(bool(b)) for a, b in zip(a_success, b_success)]
    boot = cluster_bootstrap_ci(diffs, list(clusters), n_resamples=n_resamples, seed=seed)
    return {
        "n_items": mc.n_items,
        "acc_a": sum(bool(a) for a in a_success) / mc.n_items,
        "acc_b": sum(bool(b) for b in b_success) / mc.n_items,
        "diff": boot.point,
        "diff_ci_low": boot.low,
        "diff_ci_high": boot.high,
        "n_a_only": mc.n_a_only,
        "n_b_only": mc.n_b_only,
        "n_discordant": mc.n_discordant,
        "p_mcnemar": mc.p_value,
    }


def compare_arms(
    success: Mapping[str, Sequence[bool]],
    clusters: Sequence[str],
    pairs: Sequence[Sequence[str]],
    alpha: float = 0.05,
    n_resamples: int = 2000,
    seed: int = 20260904,
    extra_p: Optional[Mapping[str, Sequence[float]]] = None,
) -> List[Dict[str, object]]:
    """Run every listed (A, B) comparison and Holm-adjust across all of them.

    ``success`` maps arm name to a per-item success sequence, all aligned to
    ``clusters``. ``extra_p`` optionally maps ``"A|B"`` to further raw McNemar
    p-values (a sensitivity analysis); their min and max are reported unadjusted.
    """
    rows: List[Dict[str, object]] = []
    for a, b in pairs:
        row = paired_difference(success[a], success[b], clusters, n_resamples, seed)
        row = {"arm_a": a, "arm_b": b, **row}
        key = "%s|%s" % (a, b)
        if extra_p and key in extra_p:
            ps = list(extra_p[key])
            row["p_mcnemar_min_over_repeats"] = min(ps)
            row["p_mcnemar_max_over_repeats"] = max(ps)
        rows.append(row)
    adj = holm_adjust([float(r["p_mcnemar"]) for r in rows])
    for r, p in zip(rows, adj):
        r["p_holm"] = p
        r["significant"] = bool(p < alpha)
        r["alpha"] = alpha
        r["n_comparisons"] = len(rows)
    return rows
