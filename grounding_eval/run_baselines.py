"""Run the three non-LLM baselines over the corpus and write results/.

Usage::

    python3 -m grounding_eval.run_baselines [--out results] [--seeds 5]

Adapters:

* ``random_floor``: uniform choice between the item's two answer options and a
  uniformly random citation. Seeded; run once per seed.
* ``retrieval_tfidf``: TF-IDF retrieval over the source excerpts; answers with
  the best-matching excerpt and cites its clause. Deterministic; run once.
* ``oracle``: reference answer plus gold citation. Deterministic; run once.

Outputs: ``results/baselines_<adapter>.json`` (every run, every response, every
score), ``results/baselines_summary.csv`` (one row per adapter) and
``results/baselines_table.md`` (the same table rendered for docs/results.md).
No language model is involved anywhere in this script.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence

import pandas as pd

from .adapters.bm25 import BM25Adapter
from .adapters.oracle import OracleAdapter
from .adapters.random_floor import RandomFloorAdapter
from .adapters.retrieval import RetrievalAdapter
from .corpus import Corpus, load_corpus
from .harness import RunResult, run_once
from .report import arm_summary, runs_frame
from .schema import ItemType
from .scoring import ScoringConfig
from .scoring.score import utility_of
from .stats import cluster_bootstrap_ci

BASE_SEED = 20260907
BOOT_SEED = 20260904


def run_all(corpus: Corpus, n_seeds: int = 5) -> Dict[str, List[RunResult]]:
    clause_pool = [item.source.clause for item in corpus]
    out: Dict[str, List[RunResult]] = {}
    out["random_floor"] = [
        run_once(RandomFloorAdapter(seed=BASE_SEED + k, repeat=k, clause_pool=clause_pool), corpus, repeat=k, seed=BASE_SEED + k)
        for k in range(n_seeds)
    ]
    out["retrieval_tfidf"] = [run_once(RetrievalAdapter(corpus), corpus)]
    out["retrieval_bm25"] = [run_once(BM25Adapter(corpus), corpus)]
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
    order = ["random_floor", "retrieval_tfidf", "retrieval_bm25", "oracle"]
    for name in order:
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
    # Retrieval diagnostics: how often the top document was the item's own source.
    for adapter_name in ["retrieval_tfidf", "retrieval_bm25"]:
        if adapter_name in runs_by_adapter:
            ret = runs_by_adapter[adapter_name][0]
            self_match = sum(1 for r in ret.responses if r.metadata.get("self_match"))
            print("%s self-match: %d/%d items retrieved their own source document" % (adapter_name, self_match, len(ret.responses)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
