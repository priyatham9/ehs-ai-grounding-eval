"""Verify every number in docs/manuscript/manuscript.md against its source file.

The manuscript's Appendix A is a table ``| ID | Printed | Source | Locator |``. Each row
names a committed file and a locator that this module resolves by reading the file
(or recomputing from it). The printed value must equal the resolved value rounded to
the printed precision. A second check requires every numeric token in the body of the
manuscript to occur among the appendix's printed values, so a number cannot appear in
the prose without a source row.

Locator forms (fields separated by ``::``):

* ``csv::FILE::col=val&col=val::COLUMN``  value of a column in the one matching row
* ``csvn::FILE::filters::COL1*COL2``      product of two columns, rounded to an integer
* ``json::FILE::dotted.path``             value at a path in a JSON file
* ``jsonlen::FILE::path[?key=val]``       length of a list, optionally filtered
  (an empty ``val`` selects entries lacking the key)
* ``corpus::ATTR``                        a count taken from ``load_corpus()``
* ``grep::FILE::literal``                 the literal occurs in the file and contains
  every number of the printed value
* ``grep100::FILE::literal``              as ``grep``, but each printed number must equal 100 times
  (or 100 times one minus) a decimal in the literal, for a coverage target or a 95% level
* ``cp::lo|hi::IDK::IDN``                 exact (Clopper-Pearson) 95% bound for K of N,
  where IDK and IDN are other appendix IDs
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from typing import Dict, List, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

MANUSCRIPT = os.path.join(REPO_ROOT, "docs", "manuscript", "manuscript.md")
APPENDIX_MARK = "## Appendix A"
FILE_KINDS = ("csv", "csvn", "json", "jsonlen", "grep", "grep100")
NUM = re.compile(r"(?<![A-Za-z\d.])\d+(?:\.\d+)?")


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact two-sided binomial interval by bisection on the binomial tails."""

    def tail_ge(p: float) -> float:  # P(X >= k)
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))

    def tail_le(p: float) -> float:  # P(X <= k)
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))

    def bisect(f, target, increasing):
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if (f(mid) < target) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    low = 0.0 if k == 0 else bisect(tail_ge, alpha / 2, True)
    high = 1.0 if k == n else bisect(tail_le, alpha / 2, False)
    return low, high


def _filters(spec: str) -> Dict[str, str]:
    return dict(p.split("=", 1) for p in spec.split("&")) if spec else {}


def _csv_row(path: str, spec: str) -> Dict[str, str]:
    want = _filters(spec)
    with open(os.path.join(REPO_ROOT, path), newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if all(r.get(k) == v for k, v in want.items())]
    if len(rows) != 1:
        raise ValueError("%s [%s] matched %d rows" % (path, spec, len(rows)))
    return rows[0]


def _dig(obj, dotted: str):
    for part in dotted.split("."):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def parse_appendix(text: str = None) -> List[Dict[str, str]]:
    if text is None:
        with open(MANUSCRIPT, encoding="utf-8") as fh:
            text = fh.read()
    rows = []
    for line in text[text.index(APPENDIX_MARK):].splitlines():
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) == 4 and re.fullmatch(r"(?=.*\d)[A-Z][A-Z0-9]*", cells[0]):
            rows.append(dict(zip(("id", "printed", "source", "locator"), cells)))
    return rows


def resolve(row: Dict[str, str], known: Dict[str, float]) -> object:
    kind, *args = row["locator"].split("::")
    if kind == "csv":
        raw = _csv_row(args[0], args[1])[args[2]]
        return float(raw)
    if kind == "csvn":
        r = _csv_row(args[0], args[1])
        a, b = args[2].split("*")
        return float(round(float(r[a]) * float(r[b])))
    if kind == "json":
        with open(os.path.join(REPO_ROOT, args[0]), encoding="utf-8") as fh:
            val = _dig(json.load(fh), args[1])
        return float(val)
    if kind == "jsonlen":
        path, _, flt = args[1].partition("?")
        with open(os.path.join(REPO_ROOT, args[0]), encoding="utf-8") as fh:
            seq = _dig(json.load(fh), path)
        if flt:
            key, val = flt.split("=", 1)
            seq = [x for x in seq if (x.get(key) or "") == val]
        return float(len(seq))
    if kind == "corpus":
        from grounding_eval.corpus import load_corpus

        c = load_corpus()
        return float(
            {
                "n_items": len(c.items),
                "n_factual": len(c.factual()),
                "n_category": len(c.category_errors()),
                "n_pairs": len(c.pairs()),
                "n_families": len(c.families()),
                "n_domains": len(c.domains()),
            }[args[0]]
        )
    if kind == "cp":
        lo, hi = clopper_pearson(int(known[args[1]]), int(known[args[2]]))
        return lo if args[0] == "lo" else hi
    if kind == "grep":
        with open(os.path.join(REPO_ROOT, args[0]), encoding="utf-8") as fh:
            body = fh.read()
        if args[1] not in body:
            raise ValueError("literal not found in %s: %r" % (args[0], args[1]))
        return args[1]
    if kind == "grep100":
        with open(os.path.join(REPO_ROOT, args[0]), encoding="utf-8") as fh:
            body = fh.read()
        if args[1] not in body:
            raise ValueError("literal not found in %s: %r" % (args[0], args[1]))
        floats = [float(t) for t in NUM.findall(args[1]) if "." in t]
        pcts = {"%g" % round(100 * x, 6) for x in floats} | {"%g" % round(100 * (1 - x), 6) for x in floats}
        return " ".join(sorted(pcts))
    raise ValueError("unknown locator kind %r" % kind)


def printed_matches(printed: str, value: object) -> bool:
    if isinstance(value, str):  # grep: every number of the printed value is in the literal
        return all(t in NUM.findall(value) for t in NUM.findall(printed))
    m = re.fullmatch(r"([+-]?)(\d+(?:\.(\d+))?)(%| pts)?", printed.strip())
    if not m:
        return False
    dec = len(m.group(3) or "")
    scaled = value * 100 if m.group(4) else value
    return abs(float(m.group(1) + m.group(2)) - round(scaled, dec)) < 1e-9


def check_rows() -> List[str]:
    """Return a list of problems; empty means every appendix row resolves and matches."""
    problems, known = [], {}
    rows = parse_appendix()
    seen = set()
    for r in rows:
        if r["id"] in seen:
            problems.append("duplicate id " + r["id"])
        seen.add(r["id"])
        if not os.path.exists(os.path.join(REPO_ROOT, r["source"])):
            problems.append("%s: missing source file %s" % (r["id"], r["source"]))
            continue
        try:
            val = resolve(r, known)
        except Exception as exc:  # report, do not abort the whole check
            problems.append("%s: %s" % (r["id"], exc))
            continue
        if not isinstance(val, str):
            known[r["id"]] = val
        if not printed_matches(r["printed"], val):
            problems.append("%s: printed %s but source gives %r" % (r["id"], r["printed"], val))
        if r["locator"].split("::")[0] in FILE_KINDS and r["source"] not in r["locator"]:
            problems.append("%s: source column is not the file the locator reads" % r["id"])
    return problems


def body_tokens_without_source() -> List[str]:
    """Numeric tokens in the manuscript body that no appendix row prints."""
    with open(MANUSCRIPT, encoding="utf-8") as fh:
        text = fh.read()
    body = text[: text.index(APPENDIX_MARK)]
    body = re.sub(r"```.*?```", " ", body, flags=re.S)
    body = re.sub(r"^#+ .*$", " ", body, flags=re.M)  # headings (section numbers)
    body = re.sub(r"^\d+\. ", " ", body, flags=re.M)  # numbered-list markers
    body = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", body)  # ISO dates
    body = re.sub(r"\b(19|20)\d{2}\b", " ", body)  # years
    body = re.sub(r"\b(Table|Section|Sections|Appendix|Figure)\s+\d+\w*", " ", body)
    allowed = {"0", "1"}  # trivial: interval ends [0, 1], "1 item"
    for r in parse_appendix():
        allowed.update(NUM.findall(r["printed"]))
    return sorted({t for t in NUM.findall(body) if t not in allowed})


if __name__ == "__main__":
    bad = check_rows() + ["unsourced number in body: " + t for t in body_tokens_without_source()]
    print("\n".join(bad) if bad else "manuscript numbers: all sourced and verified")
    sys.exit(1 if bad else 0)
