#!/usr/bin/env python3
"""Run the benchmark through the grounded assistant and write results/assistant_eval.{csv,md}.

No language model is used anywhere. The answer composer (MockModel) is extractive:
it quotes the top retrieved passage. Deterministic, no network, no timestamps, so
regenerating gives byte-identical files.

Usage: python3 scripts/run_assistant_eval.py [--check] [--gate]
  --check  exit 1 if committed results differ from regenerated output
  --gate   exit 1 if the release gate (below) fails
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from grounding_eval.adapters.bm25plus import BM25PlusAdapter  # noqa: E402
from grounding_eval.assistant import ABSTAIN_THRESHOLD, Assistant, AssistantAdapter  # noqa: E402
from grounding_eval.corpus import load_corpus  # noqa: E402
from grounding_eval.harness import run_once  # noqa: E402
from grounding_eval.schema import CitationGrade, ItemType, Outcome  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
SWEEP = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]
# Release gate, fixed before the first run. Stated policy, not a finding.
GATE_MIN_CITATION_PRECISION = 0.80
COLUMNS = [
    "system", "threshold", "n_factual", "n_category_error", "accuracy_factual",
    "abstention_rate_factual", "abstention_rate_all", "category_error_declined",
    "n_answered_factual", "accuracy_when_answered", "citation_precision_answered",
    "citation_section_or_better_answered", "adjacent_substitutions", "category_error_answered",
]


def _metrics(system: str, threshold, run) -> Dict[str, object]:
    fac = [s for s in run.scores if s.item_type is ItemType.FACTUAL]
    cat = [s for s in run.scores if s.item_type is ItemType.CATEGORY_ERROR]
    ans = [s for s in fac if not s.abstained]
    ok = CitationGrade.CLAUSE
    sec = (CitationGrade.CLAUSE, CitationGrade.SECTION)
    return {
        "system": system,
        "threshold": "" if threshold is None else "%.2f" % threshold,
        "n_factual": len(fac),
        "n_category_error": len(cat),
        "accuracy_factual": sum(s.outcome is Outcome.CORRECT for s in fac) / len(fac),
        "abstention_rate_factual": sum(s.abstained for s in fac) / len(fac),
        "abstention_rate_all": sum(s.abstained for s in run.scores) / len(run.scores),
        "category_error_declined": sum(s.abstained for s in cat) / len(cat),
        "n_answered_factual": len(ans),
        "accuracy_when_answered": (sum(s.outcome is Outcome.CORRECT for s in ans) / len(ans)) if ans else None,
        "citation_precision_answered": (sum(s.citation_grade is ok for s in ans) / len(ans)) if ans else None,
        "citation_section_or_better_answered": (sum(s.citation_grade in sec for s in ans) / len(ans)) if ans else None,
        "adjacent_substitutions": sum(s.outcome is Outcome.ADJACENT_SUBSTITUTION for s in run.scores),
        "category_error_answered": sum(not s.abstained for s in cat),
    }


def evaluate() -> dict:
    corpus = load_corpus()
    rows: List[Dict[str, object]] = []
    base = run_once(BM25PlusAdapter(corpus), corpus)
    rows.append(_metrics("retrieval_bm25plus (no gate, no abstention)", None, base))
    audit: List[dict] = []
    for t in SWEEP:
        adapter = AssistantAdapter(Assistant(corpus, "bm25plus", t))
        run = run_once(adapter, corpus)
        rows.append(_metrics("assistant", t, run))
        if abs(t - ABSTAIN_THRESHOLD) < 1e-9:
            audit = adapter.audit
    return {"rows": rows, "audit": audit, "digest": Assistant(corpus).digest, "n_items": len(corpus)}


def default_row(res: dict) -> dict:
    return next(r for r in res["rows"] if r["threshold"] == "%.2f" % ABSTAIN_THRESHOLD)


def gate(res: dict) -> List[tuple]:
    r = default_row(res)
    cp = r["citation_precision_answered"]
    return [
        ("zero adjacent substitutions", r["adjacent_substitutions"] == 0, str(r["adjacent_substitutions"])),
        ("zero category-error items answered", r["category_error_answered"] == 0, "%d of %d" % (r["category_error_answered"], r["n_category_error"])),
        ("citation precision on answered >= %.2f" % GATE_MIN_CITATION_PRECISION,
         cp is not None and cp >= GATE_MIN_CITATION_PRECISION, "n/a" if cp is None else "%.3f" % cp),
    ]


def _fmt(v) -> str:
    if v is None:
        return ""
    return "%.4f" % v if isinstance(v, float) else str(v)


def render_csv(res: dict) -> str:
    out = [",".join(COLUMNS)]
    for r in res["rows"]:
        out.append(",".join('"%s"' % r[c] if c == "system" else _fmt(r[c]) for c in COLUMNS))
    return "\n".join(out) + "\n"


def _pct(v) -> str:
    return "n/a" if v is None else "%.1f%%" % (100 * v)


def render_md(res: dict) -> str:
    r = default_row(res)
    L = [
        "# Grounded assistant: benchmark evaluation",
        "",
        "**No language model was used.** The answer composer is `MockModel`, which is extractive: it quotes the "
        "top retrieved corpus passage verbatim and cites its clause. Retrieval is the repository's BM25+ index. "
        "Everything below is deterministic and regenerated by `python3 scripts/run_assistant_eval.py`.",
        "",
        "Corpus digest `%s`; %d items (%d factual, %d category-error). Abstain rule: answer only if the top "
        "passage's IDF-weighted query-token coverage is at least the threshold (default %.2f; see "
        "`docs/assistant.md`)." % (res["digest"][:16], res["n_items"], r["n_factual"], r["n_category_error"], ABSTAIN_THRESHOLD),
        "",
        "Caveat: as in the repository's retrieval baselines, the governing passage of every question is in the "
        "index (the question text is not removed from it), so retrieval is tested for selecting that passage, "
        "not for finding evidence in an unseen corpus. Citation precision is the share of answered factual items "
        "whose cited clause grades `correct_clause` under the repository's citation scorer.",
        "",
        "## Headline (default threshold %.2f)" % ABSTAIN_THRESHOLD,
        "",
        "| metric | value |",
        "|---|---|",
        "| accuracy, factual items (abstention counts as not correct) | %s |" % _pct(r["accuracy_factual"]),
        "| abstention rate, factual items | %s |" % _pct(r["abstention_rate_factual"]),
        "| abstention rate, all items | %s |" % _pct(r["abstention_rate_all"]),
        "| category-error items declined | %s (%d of %d) |" % (_pct(r["category_error_declined"]), r["n_category_error"] - r["category_error_answered"], r["n_category_error"]),
        "| answered factual items | %d of %d |" % (r["n_answered_factual"], r["n_factual"]),
        "| accuracy when answered | %s |" % _pct(r["accuracy_when_answered"]),
        "| citation precision (answered, correct clause) | %s |" % _pct(r["citation_precision_answered"]),
        "| citation correct section or better (answered) | %s |" % _pct(r["citation_section_or_better_answered"]),
        "| adjacent substitutions | %d |" % r["adjacent_substitutions"],
        "",
        "## Release gate",
        "",
        "Gate criteria were fixed before the first run. They are a stated policy, not findings.",
        "",
        "| criterion | observed | result |",
        "|---|---|---|",
    ]
    g = gate(res)
    for name, passed, obs in g:
        L.append("| %s | %s | %s |" % (name, obs, "PASS" if passed else "FAIL"))
    L += ["", "**Gate: %s.**" % ("PASS" if all(p for _, p, _ in g) else "FAIL"), "",
          "## Sensitivity to the abstain threshold",
          "",
          "Shown for transparency. The default was not chosen from this table.",
          "",
          "| system | threshold | accuracy (factual) | abstained (factual) | cat.-error declined | answered | acc. when answered | citation precision | adjacent subs. |",
          "|---|---|---|---|---|---|---|---|---|"]
    for x in res["rows"]:
        L.append("| %s | %s | %s | %s | %s | %d | %s | %s | %d |" % (
            x["system"], x["threshold"] or "none", _pct(x["accuracy_factual"]), _pct(x["abstention_rate_factual"]),
            _pct(x["category_error_declined"]), x["n_answered_factual"], _pct(x["accuracy_when_answered"]),
            _pct(x["citation_precision_answered"]), x["adjacent_substitutions"]))
    L += ["", "Per-answer audit records for the default threshold: `results/assistant_audit.jsonl`.", ""]
    return "\n".join(L)


def render_audit(res: dict) -> str:
    return "".join(json.dumps(a, sort_keys=True) + "\n" for a in res["audit"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args()
    res = evaluate()
    files = {"assistant_eval.csv": render_csv(res), "assistant_eval.md": render_md(res),
             "assistant_audit.jsonl": render_audit(res)}
    if args.check:
        bad = [n for n, t in files.items() if open(os.path.join(RESULTS, n)).read() != t]
        print("stale: %s" % bad if bad else "up to date")
        return 1 if bad else 0
    for n, t in files.items():
        with open(os.path.join(RESULTS, n), "w") as fh:
            fh.write(t)
    print(files["assistant_eval.md"])
    if args.gate and not all(p for _, p, _ in gate(res)):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
