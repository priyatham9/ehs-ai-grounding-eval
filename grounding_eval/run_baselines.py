"""Run the non-LLM baselines over the corpus and write results/.

Usage::

    python3 -m grounding_eval.run_baselines [--out results] [--seeds 5]

Adapters:

* ``random_floor``: uniform choice between the item's two answer options and a
  uniformly random citation. Seeded; run once per seed.
* ``retrieval_tfidf``: TF-IDF retrieval over the source excerpts; answers with
  the best-matching excerpt and cites its clause. Deterministic; run once.
* ``retrieval_bm25``: Okapi BM25 (k1=1.2, b=0.75) over the same documents.
* ``retrieval_bm25plus``: BM25+ (k1=1.2, b=0.75, delta=1.0, non-negative IDF).
* ``retrieval_rrf``: reciprocal rank fusion (k=60) of the TF-IDF and BM25 rankings.
* ``oracle``: reference answer plus gold citation. Deterministic; run once.

Outputs: ``results/baselines_<adapter>.json`` (every run, every response, every
score), ``results/baselines_summary.csv`` (one row per adapter),
``results/baselines_table.md`` (the same table rendered for docs/results.md) and
``results/baselines_breakdown.csv`` / ``results/baselines_breakdown.md``
(factual accuracy by domain and on minimal-pair items, with cluster-bootstrap CIs) and
``results/selective_summary.csv`` / ``.md`` / ``selective_curves.csv`` (answer-or-abstain
scoring from each system's confidence; see ``grounding_eval/selective.py``), and
``results/calibration_summary.csv`` / ``.md`` / ``calibration_reliability.csv`` (Brier,
ECE and reliability bins; see ``grounding_eval/calibration.py``).
No language model is involved anywhere in this script.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence

import pandas as pd

from .adapters.bm25 import BM25Adapter
from .adapters.bm25plus import BM25PlusAdapter, RRFAdapter
from .adapters.oracle import OracleAdapter
from .adapters.random_floor import RandomFloorAdapter
from .adapters.retrieval import RetrievalAdapter
from .corpus import Corpus, load_corpus
from .harness import RunResult, run_once
from .report import arm_summary, runs_frame
from .schema import ItemType
from .scoring import ScoringConfig
from .scoring.score import utility_of
from .calibration import reliability_table, summarize_calibration
from .selective import risk_coverage_points, summarize_selective
from .stats import cluster_bootstrap_ci

BASE_SEED = 20260907
BOOT_SEED = 20260904
ADAPTER_ORDER = [
    "random_floor", "retrieval_tfidf", "retrieval_bm25", "retrieval_bm25plus", "retrieval_rrf", "oracle",
]
RETRIEVERS = ["retrieval_tfidf", "retrieval_bm25", "retrieval_bm25plus", "retrieval_rrf"]


def run_all(corpus: Corpus, n_seeds: int = 5) -> Dict[str, List[RunResult]]:
    clause_pool = [item.source.clause for item in corpus]
    out: Dict[str, List[RunResult]] = {}
    out["random_floor"] = [
        run_once(RandomFloorAdapter(seed=BASE_SEED + k, repeat=k, clause_pool=clause_pool), corpus, repeat=k, seed=BASE_SEED + k)
        for k in range(n_seeds)
    ]
    out["retrieval_tfidf"] = [run_once(RetrievalAdapter(corpus), corpus)]
    out["retrieval_bm25"] = [run_once(BM25Adapter(corpus), corpus)]
    out["retrieval_bm25plus"] = [run_once(BM25PlusAdapter(corpus), corpus)]
    out["retrieval_rrf"] = [run_once(RRFAdapter(corpus), corpus)]
    out["oracle"] = [run_once(OracleAdapter(), corpus)]
    return out


def _ci_row(values: Sequence[float], clusters: Sequence[str]) -> Dict[str, float]:
    b = cluster_bootstrap_ci(list(values), list(clusters), seed=BOOT_SEED)
    return {"point": b.point, "low": b.low, "high": b.high}


def summarize(runs_by_adapter: Dict[str, List[RunResult]], corpus: Corpus) -> pd.DataFrame:
    """arm_summary plus cluster-bootstrap CIs for the other headline rates."""
    all_runs = [r for runs in runs_by_adapter.values() for r in runs]
    frame = runs_frame(all_runs)
    base = arm_summary(frame, corpus, seed=BOOT_SEED)
    by_id = {item.id: item for item in corpus}
    cfg = ScoringConfig()
    extra: List[Dict[str, object]] = []
    for adapter, group in frame.groupby("adapter"):
        factual = group[group["item_type"] == ItemType.FACTUAL.value]
        fam = factual["family"].tolist()
        adj = _ci_row(factual["adjacent"].astype(float).tolist(), fam)
        cite = _ci_row((group["citation_grade"] == "correct_clause").astype(float).tolist(), group["family"].tolist())
        # Abstention credit: mean policy utility under the named "safety" policy.
        runs = runs_by_adapter[adapter]
        utils, ufam = [], []
        for run in runs:
            for score in run.scores:
                utils.append(utility_of(score, by_id[score.item_id], cfg))
                ufam.append(score.family)
        util = _ci_row(utils, ufam)
        extra.append(
            {
                "adapter": adapter,
                "n_items": len(corpus),
                "adjacent_ci_low": adj["low"],
                "adjacent_ci_high": adj["high"],
                "citation_ci_low": cite["low"],
                "citation_ci_high": cite["high"],
                "safety_utility": util["point"],
                "safety_utility_ci_low": util["low"],
                "safety_utility_ci_high": util["high"],
                "utility_policy": cfg.describe().get("policy", "safety"),
            }
        )
    return base.merge(pd.DataFrame(extra), on="adapter")


def render_table(summary: pd.DataFrame) -> str:
    def pct(x: float) -> str:
        return "%.1f%%" % (100.0 * x)

    lines = [
        "| adapter | arm | repeats | accuracy (factual, n=%d) [95%% CI] | adjacent substitution [95%% CI] | citation correct clause [95%% CI] | category-error declined | safety utility [95%% CI] |"
        % int(summary["n_factual_obs"].iloc[0] / summary["n_repeats"].iloc[0]),
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in ADAPTER_ORDER:
        r = summary[summary["adapter"] == name].iloc[0]
        lines.append(
            "| %s | %s | %d | %s [%s, %s] | %s [%s, %s] | %s [%s, %s] | %s | %.2f [%.2f, %.2f] |"
            % (
                name, r["arm"], r["n_repeats"],
                pct(r["accuracy"]), pct(r["accuracy_ci_low"]), pct(r["accuracy_ci_high"]),
                pct(r["adjacent_substitution_rate"]), pct(r["adjacent_ci_low"]), pct(r["adjacent_ci_high"]),
                pct(r["citation_correct_clause_rate"]), pct(r["citation_ci_low"]), pct(r["citation_ci_high"]),
                pct(r["category_error_declined_rate"]),
                r["safety_utility"], r["safety_utility_ci_low"], r["safety_utility_ci_high"],
            )
        )
    return "\n".join(lines) + "\n"


def breakdown(runs_by_adapter: Dict[str, List[RunResult]], corpus: Corpus) -> pd.DataFrame:
    """Factual accuracy by domain, on minimal-pair items, and pair-level (both correct).

    Every row carries a percentile cluster bootstrap CI with the same seed as the
    headline table. Domain and item-subset rows resample item families; the
    ``pairs_both_correct`` row has one value per (repeat, pair) and resamples pairs.
    """
    frame = runs_frame([r for runs in runs_by_adapter.values() for r in runs])
    factual = frame[frame["item_type"] == ItemType.FACTUAL.value]
    pair_of = {item.id: item.pair_id for pair in corpus.pairs() for item in pair}
    rows: List[Dict[str, object]] = []

    def add(adapter: str, subset: str, values: Sequence[float], clusters: Sequence[str], n_items: int) -> None:
        ci = _ci_row(values, clusters)
        rows.append({
            "adapter": adapter, "subset": subset, "n_items": n_items, "n_obs": len(values),
            "accuracy": ci["point"], "ci_low": ci["low"], "ci_high": ci["high"],
        })

    for adapter in [a for a in ADAPTER_ORDER if a in runs_by_adapter]:
        g = factual[factual["adapter"] == adapter]
        n_rep = max(int(g["repeat"].nunique()), 1)
        add(adapter, "all_factual", g["correct"].astype(float).tolist(), g["family"].tolist(), len(g) // n_rep)
        for domain in sorted(g["domain"].unique()):
            d = g[g["domain"] == domain]
            add(adapter, "domain:%s" % domain, d["correct"].astype(float).tolist(), d["family"].tolist(), len(d) // n_rep)
        paired = g[g["item_id"].isin(pair_of)]
        add(adapter, "minimal_pair_items", paired["correct"].astype(float).tolist(), paired["family"].tolist(), len(paired) // n_rep)
        unpaired = g[~g["item_id"].isin(pair_of)]
        add(adapter, "unpaired_items", unpaired["correct"].astype(float).tolist(), unpaired["family"].tolist(), len(unpaired) // n_rep)
        both: List[float] = []
        labels: List[str] = []
        for _, run_group in paired.groupby("repeat"):
            success = dict(zip(run_group["item_id"], run_group["correct"].astype(bool)))
            for a, b in corpus.pairs():
                if a.id in success and b.id in success:
                    both.append(float(success[a.id] and success[b.id]))
                    labels.append(a.pair_id)
        add(adapter, "pairs_both_correct", both, labels, len(corpus.pairs()))
    return pd.DataFrame(rows)


def render_breakdown(table: pd.DataFrame) -> str:
    """Wide markdown: one row per subset, one column per adapter."""
    adapters = [a for a in ADAPTER_ORDER if a in set(table["adapter"])]
    subsets = list(dict.fromkeys(table["subset"]))
    lines = [
        "| subset | n | " + " | ".join(adapters) + " |",
        "|---|---|" + "---|" * len(adapters),
    ]
    for subset in subsets:
        cells = []
        n = None
        for a in adapters:
            r = table[(table["adapter"] == a) & (table["subset"] == subset)].iloc[0]
            n = int(r["n_items"])
            cells.append("%.1f%% [%.1f, %.1f]" % (100 * r["accuracy"], 100 * r["ci_low"], 100 * r["ci_high"]))
        label = subset.replace("domain:", "")
        lines.append("| %s | %d | %s |" % (label, n, " | ".join(cells)))
    return "\n".join(lines) + "\n"


def selective(runs_by_adapter: Dict[str, List[RunResult]]) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Answer-or-abstain scoring on factual items: summary rows and curve points.

    Abstaining means declining the items whose confidence falls below a threshold.
    Only systems that report a confidence can be scored; random_floor and oracle
    report none, so their rows carry no curve (stated, not imputed).
    """
    frame = runs_frame([r for runs in runs_by_adapter.values() for r in runs])
    factual = frame[frame["item_type"] == ItemType.FACTUAL.value]
    summary_rows: List[Dict[str, object]] = []
    curve_rows: List[Dict[str, object]] = []
    for adapter in [a for a in ADAPTER_ORDER if a in runs_by_adapter]:
        g = factual[factual["adapter"] == adapter]
        conf = [None if pd.isna(c) else float(c) for c in g["confidence"]]
        ok = g["correct"].astype(bool).tolist()
        row = summarize_selective(conf, ok, g["family"].tolist(), seed=BOOT_SEED)
        row["adapter"] = adapter
        summary_rows.append(row)
        for p in risk_coverage_points(conf, ok):
            curve_rows.append({"adapter": adapter, **p})
    cols = ["adapter"]
    summary = pd.DataFrame(summary_rows)
    summary = summary[cols + [c for c in summary.columns if c != "adapter"]]
    return summary, pd.DataFrame(curve_rows)


def render_selective(summary: pd.DataFrame) -> str:
    def f(x: float) -> str:
        return "n/a" if pd.isna(x) else "%.3f" % x

    def pct(x: float, lo: float, hi: float, cov: float) -> str:
        if pd.isna(x):
            return "n/a"
        return "%.1f%% [%.1f, %.1f] (answers %.0f%%)" % (100 * x, 100 * lo, 100 * hi, 100 * cov)

    lines = [
        "| adapter | accuracy at full coverage | AURC [95% CI] | AURC, uninformative score | AURC, best ordering | E-AURC | selective accuracy at 25% coverage | at 50% | at 75% |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        aurc_cell = "n/a" if pd.isna(r["aurc"]) else "%s [%s, %s]" % (f(r["aurc"]), f(r["aurc_ci_low"]), f(r["aurc_ci_high"]))
        cells = [pct(r["sel_acc_at_%d" % c], r["sel_acc_at_%d_ci_low" % c], r["sel_acc_at_%d_ci_high" % c], r["achieved_cov_at_%d" % c]) for c in (25, 50, 75)]
        lines.append("| %s | %.1f%% | %s | %s | %s | %s | %s |" % (
            r["adapter"], 100 * r["accuracy_full_coverage"], aurc_cell,
            f(r["aurc_random"]), f(r["aurc_optimal"]), f(r["e_aurc"]), " | ".join(cells)))
    return "\n".join(lines) + "\n"


# Systems whose confidence is on a probability scale by construction. TF-IDF
# cosine similarity lies in [0, 1]. Okapi BM25 and BM25+ scores are unbounded
# sums of weights (observed ranges -4.69 to 49.20 and 16.47 to 145.53), and reciprocal rank fusion
# scores are bounded by 2 / (k + 1) = 0.033 (observed 0.032 to 0.033), so neither is a probability and
# calibration metrics are not computed for them.
CALIBRATION_SCORED = ("retrieval_tfidf",)


def calibration(runs_by_adapter: Dict[str, List[RunResult]]) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Brier, ECE and reliability bins on factual items, for systems on a [0, 1] scale."""
    frame = runs_frame([r for runs in runs_by_adapter.values() for r in runs])
    factual = frame[frame["item_type"] == ItemType.FACTUAL.value]
    summary_rows: List[Dict[str, object]] = []
    bin_rows: List[Dict[str, object]] = []
    for adapter in [a for a in ADAPTER_ORDER if a in runs_by_adapter]:
        g = factual[factual["adapter"] == adapter]
        ok = g["correct"].astype(bool).tolist()
        scored = adapter in CALIBRATION_SCORED
        conf = [None if (pd.isna(c) or not scored) else float(c) for c in g["confidence"]]
        row = summarize_calibration(conf, ok, g["family"].tolist(), seed=BOOT_SEED)
        row["adapter"] = adapter
        row["has_confidence"] = bool(g["confidence"].notna().any())
        row["computed"] = scored and row["n_with_confidence"] > 0
        summary_rows.append(row)
        if row["computed"]:
            for b in reliability_table(conf, ok):
                bin_rows.append({"adapter": adapter, **b})
    summary = pd.DataFrame(summary_rows)
    summary = summary[["adapter"] + [c for c in summary.columns if c != "adapter"]]
    return summary, pd.DataFrame(bin_rows)


def render_calibration(summary: pd.DataFrame, bins: pd.DataFrame) -> str:
    def f(x: float) -> str:
        return "n/a" if pd.isna(x) else "%.3f" % x

    why = {True: "no confidence reported"}
    lines = [
        "| adapter | accuracy | mean confidence | Brier [95% CI] | Brier, constant at accuracy | ECE (10 bins) [95% CI] |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        if r["computed"]:
            lines.append("| %s | %.1f%% | %s | %s [%s, %s] | %s | %s [%s, %s] |" % (
                r["adapter"], 100 * r["accuracy"], f(r["mean_confidence"]),
                f(r["brier"]), f(r["brier_ci_low"]), f(r["brier_ci_high"]), f(r["brier_constant"]),
                f(r["ece"]), f(r["ece_ci_low"]), f(r["ece_ci_high"])))
        else:
            reason = "not on a [0, 1] scale" if r["has_confidence"] else why[True]
            lines.append("| %s | %.1f%% | n/a | n/a (%s) | n/a | n/a |" % (r["adapter"], 100 * r["accuracy"], reason))
    out = "\n".join(lines) + "\n"
    for adapter in bins["adapter"].unique() if len(bins) else []:
        out += "\nReliability bins, %s:\n\n| bin | items | mean confidence | accuracy |\n|---|---|---|---|\n" % adapter
        for _, b in bins[bins["adapter"] == adapter].iterrows():
            if b["count"]:
                out += "| [%.1f, %.1f%s | %d | %.3f | %.1f%% |\n" % (
                    b["low"], b["high"], "]" if b["high"] >= 1.0 else ")", b["count"], b["mean_confidence"], 100 * b["accuracy"])
    return out


def main(argv: Sequence[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    corpus = load_corpus()
    runs_by_adapter = run_all(corpus, args.seeds)
    for name, runs in runs_by_adapter.items():
        path = os.path.join(args.out, "baselines_%s.json" % name)
        with open(path, "w") as fh:
            json.dump({"schema": "grounding_eval.baselines/1", "adapter": name, "runs": [r.to_dict() for r in runs]}, fh, indent=1, sort_keys=True)
    summary = summarize(runs_by_adapter, corpus)
    summary.to_csv(os.path.join(args.out, "baselines_summary.csv"), index=False)
    table = render_table(summary)
    with open(os.path.join(args.out, "baselines_table.md"), "w") as fh:
        fh.write(table)
    print(table)
    detail = breakdown(runs_by_adapter, corpus)
    detail.to_csv(os.path.join(args.out, "baselines_breakdown.csv"), index=False)
    detail_md = render_breakdown(detail)
    with open(os.path.join(args.out, "baselines_breakdown.md"), "w") as fh:
        fh.write(detail_md)
    print(detail_md)
    sel, curves = selective(runs_by_adapter)
    sel.to_csv(os.path.join(args.out, "selective_summary.csv"), index=False)
    curves.to_csv(os.path.join(args.out, "selective_curves.csv"), index=False)
    sel_md = render_selective(sel)
    with open(os.path.join(args.out, "selective_summary.md"), "w") as fh:
        fh.write(sel_md)
    print(sel_md)
    cal, cal_bins = calibration(runs_by_adapter)
    cal.to_csv(os.path.join(args.out, "calibration_summary.csv"), index=False)
    cal_bins.to_csv(os.path.join(args.out, "calibration_reliability.csv"), index=False)
    cal_md = render_calibration(cal, cal_bins)
    with open(os.path.join(args.out, "calibration_summary.md"), "w") as fh:
        fh.write(cal_md)
    print(cal_md)
    # Retrieval diagnostics: how often the top document was the item's own source.
    for adapter_name in RETRIEVERS:
        if adapter_name in runs_by_adapter:
            ret = runs_by_adapter[adapter_name][0]
            self_match = sum(1 for r in ret.responses if r.metadata.get("self_match"))
            print("%s self-match: %d/%d items retrieved their own source document" % (adapter_name, self_match, len(ret.responses)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
