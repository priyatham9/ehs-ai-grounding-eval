#!/usr/bin/env python3
"""Re-verify every corpus item's cited source against the live eCFR API.

Why this exists
---------------
Every item in this corpus asserts that a specific paragraph of a specific federal
regulation says a specific thing. Those assertions are the only reason the answer
keys can be called correct. An assertion that nobody can re-check is worth
nothing, so this script re-checks all of them from the primary source and writes
a dated report.

What it checks, per item
------------------------
1. The cited section can be fetched from the eCFR versioner API.
2. The item's ``source.anchor_text`` appears in that section's text, after
   whitespace and quotation-mark normalisation. The anchor is a short verbatim
   span from the regulation, so this is a direct textual check rather than a
   judgement.
3. The cited paragraph identifier (for example ``(c)(6)(i)(D)``) appears in the
   section as a paragraph marker.

What it does NOT check
----------------------
It does not check that the item's ``correct_answer`` is a good answer, that the
answer key is well constructed, or that the anchor is the *most* relevant span in
the section. Those are editorial judgements and a script cannot make them. It
checks provenance only.

Copyrighted standards (ASME BPVC, API 520/521, NFPA 70E, ISO 4126) are not
fetched and cannot be verified this way. No item's answer key depends on their
text; where they are mentioned it is as a ``cross_reference`` whose paragraph
identifier is corroborated by a public federal source that names it.

Network use
-----------
This is the only part of the repository that touches the network, and it is never
called by the harness, the scorer or the test suite. Offline, run with
``--offline`` to re-check the corpus against the stored report instead.

Usage
-----
    python3 tools/verify_sources.py                  # fetch and write a report
    python3 tools/verify_sources.py --offline        # re-check stored report only
    python3 tools/verify_sources.py --date 2026-08-01
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grounding_eval.corpus import DEFAULT_ITEMS_DIR, REPO_ROOT  # noqa: E402

ECFR_FULL = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml"
REPORT_PATH = os.path.join(REPO_ROOT, "corpus", "verification", "source_verification.json")

USER_AGENT = (
    "ehs-ai-grounding-eval source verifier "
    "(https://github.com/priyatham9/ehs-ai-grounding-eval)"
)

#: Standards whose text is copyrighted and is therefore never fetched or quoted.
PAYWALLED_STANDARDS = ("ASME", "API", "NFPA", "ISO", "ANSI", "IEC")


# --------------------------------------------------------------------------- #
# Text normalisation
# --------------------------------------------------------------------------- #

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_SMART = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "−": "-",
    " ": " ",
}


def normalize(text: str) -> str:
    """Lowercase, strip markup, and flatten whitespace and typographic quotes.

    The comparison has to survive the difference between how eCFR marks up a
    paragraph and how a human transcribed it, without becoming so loose that it
    matches text that is not there. Normalising case, quotes, dashes and runs of
    whitespace is enough for that, and nothing more is done.
    """
    text = _TAG_RE.sub(" ", text)
    for bad, good in _SMART.items():
        text = text.replace(bad, good)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&#8212;", "-").replace("&#8217;", "'")
    return _WS_RE.sub(" ", text).strip().lower()


# --------------------------------------------------------------------------- #
# Citation parsing
# --------------------------------------------------------------------------- #

_SECTION_RE = re.compile(r"\b(\d{1,2})\s*CFR\s*([0-9]+\.[0-9A-Za-z\-]+)")
_PARA_RE = re.compile(r"\((?:[a-z]{1,2}|[0-9]{1,2}|[ivxlc]{1,5}|[A-Z]{1,2})\)")


def parse_citation(clause: str) -> Optional[Tuple[str, str]]:
    """Extract ``(title, section)`` from a clause string such as '29 CFR 1910.147(c)(1)'."""
    match = _SECTION_RE.search(clause)
    if not match:
        return None
    return match.group(1), match.group(2)


def parse_paragraph_path(clause: str) -> Tuple[str, ...]:
    """Return the trailing paragraph markers of a clause, e.g. ('(c)', '(6)', '(i)').

    Only the run of markers immediately following the section number is taken, so
    that prose after the citation ("and Note to (c)(7)") does not contribute.
    """
    match = _SECTION_RE.search(clause)
    if not match:
        return ()
    tail = clause[match.end():]
    out: List[str] = []
    pos = 0
    while True:
        m = _PARA_RE.match(tail, pos)
        if not m:
            break
        out.append(m.group(0))
        pos = m.end()
    return tuple(out)


def part_of(section: str) -> str:
    """'1910.147' -> '1910'; '54.15-13' -> '54'."""
    return section.split(".", 1)[0]


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #


def fetch_section(title: str, section: str, date: str, pause: float = 1.0) -> str:
    """Fetch one CFR section as XML text from the eCFR versioner API.

    The endpoint requires that the client accept compression, so the response is
    decompressed here rather than relying on urllib defaults.
    """
    url = "%s?part=%s&section=%s" % (
        ECFR_FULL.format(date=date, title=title),
        part_of(section),
        section,
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip", "Accept": "application/xml"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
    time.sleep(pause)  # be a well-behaved client against a public API
    return raw.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# Checking
# --------------------------------------------------------------------------- #


def paragraph_markers(section_xml: str) -> set:
    """Every parenthesised marker that begins a paragraph in the section text.

    eCFR renders paragraph designators inline at the start of the paragraph, so
    the markers can be recovered from the flattened text.

    Case is deliberately preserved here, unlike in :func:`normalize`. CFR
    paragraph hierarchy distinguishes ``(a)`` from ``(A)`` and ``(i)`` from
    ``(I)``, so lowercasing before this search would make an uppercase marker
    unfindable and report a false failure.
    """
    flat = _WS_RE.sub(" ", _TAG_RE.sub(" ", section_xml))
    return set(_PARA_RE.findall(flat))


def all_cited_markers(clause: str) -> Tuple[str, ...]:
    """Every paragraph marker named anywhere after the section number.

    ``parse_paragraph_path`` deliberately reads only the first run, so that
    prose after the citation does not contribute to the hierarchy check. For
    locality this is too strict: a clause such as "1910.147(c)(5) and (c)(6)"
    anchors on the later member, so every marker it names has to count.
    """
    match = _SECTION_RE.search(clause)
    if not match:
        return ()
    return tuple(m.group(0) for m in _PARA_RE.finditer(clause[match.end():]))


def anchor_paragraph_locality(section_text: str, anchor: str, path: Sequence[str]) -> Optional[bool]:
    """Whether the anchor sits just after the deepest cited paragraph marker.

    The marker check alone is weak: it confirms that ``(d)`` occurs *somewhere*
    in the section, not that the anchor is inside paragraph (d). An item can
    therefore cite one paragraph while anchoring on text that lives in another,
    which is a real provenance defect and was found in this corpus once.

    Returns True when the deepest cited marker is among the last few markers
    preceding the anchor, False when it is not, and None when the question does
    not apply — no anchor, no paragraph path, or a citation into an alphabetical
    definitions block, where the governing designator is a defined term rather
    than a marker and locality cannot be judged this way.
    """
    if not anchor or not path:
        return None
    flat = _WS_RE.sub(" ", _TAG_RE.sub(" ", section_text))
    needle = _WS_RE.sub(" ", anchor).strip()
    index = flat.lower().find(needle.lower())
    if index < 0:
        return None
    preceding = [m.group(0) for m in _PARA_RE.finditer(flat[:index])]
    # Allow a little slack: the anchor may start a few markers into the cited
    # paragraph. A clause naming several paragraphs ("(c)(5) and (c)(6)")
    # anchors on one of them, so any cited deepest marker satisfies this.
    return any(marker in preceding[-6:] for marker in path)


#: Clauses that point at a definitions block rather than a numbered paragraph.
#: CFR definitions are alphabetical, so paragraph-locality does not apply.
_DEFINITIONS_RE = re.compile(r"\bdefinitions?\b", re.IGNORECASE)


def check_item(item: dict, section_text: str) -> Dict[str, object]:
    """Check one item's anchor text and paragraph path against fetched section text."""
    source = item["source"]
    anchor = (source.get("anchor_text") or "").strip()
    clause = source.get("clause", "")
    flat = normalize(section_text)

    anchor_found = bool(anchor) and normalize(anchor) in flat
    path = parse_paragraph_path(clause)
    markers = paragraph_markers(section_text)
    missing_markers = [p for p in path if p not in markers]

    if _DEFINITIONS_RE.search(clause):
        locality: Optional[bool] = None
    else:
        locality = anchor_paragraph_locality(section_text, anchor, all_cited_markers(clause))

    return {
        "item_id": item["id"],
        "clause": clause,
        "anchor_present": anchor_found,
        "anchor_length": len(anchor),
        "paragraph_path": list(path),
        "paragraph_markers_missing": missing_markers,
        "anchor_in_cited_paragraph": locality,
        "ok": bool(anchor_found and not missing_markers),
    }


def load_raw_items(items_dir: str = DEFAULT_ITEMS_DIR) -> List[dict]:
    """Read corpus item dicts straight from JSON, bypassing the typed loader."""
    out: List[dict] = []
    for name in sorted(os.listdir(items_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(items_dir, name), "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for raw in payload.get("items", []):
            raw = dict(raw)
            raw["_file"] = name
            out.append(raw)
    return sorted(out, key=lambda r: r["id"])


def verify(date: str, items_dir: str = DEFAULT_ITEMS_DIR, pause: float = 1.0) -> Dict[str, object]:
    """Fetch every distinct cited section once and check every item against it."""
    items = load_raw_items(items_dir)

    wanted: Dict[Tuple[str, str], List[dict]] = {}
    unparsed: List[dict] = []
    for item in items:
        parsed = parse_citation(item["source"].get("clause", ""))
        if parsed is None:
            unparsed.append({"item_id": item["id"], "clause": item["source"].get("clause")})
            continue
        wanted.setdefault(parsed, []).append(item)

    results: List[Dict[str, object]] = []
    fetch_errors: List[Dict[str, object]] = []
    sections_seen: List[str] = []

    for (title, section) in sorted(wanted):
        label = "%s CFR %s" % (title, section)
        sys.stderr.write("fetching %s ...\n" % label)
        try:
            xml = fetch_section(title, section, date, pause=pause)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            fetch_errors.append({"section": label, "error": str(exc)})
            for item in wanted[(title, section)]:
                results.append(
                    {
                        "item_id": item["id"],
                        "clause": item["source"].get("clause"),
                        "ok": False,
                        "error": "fetch_failed: %s" % exc,
                    }
                )
            continue
        sections_seen.append(label)
        for item in wanted[(title, section)]:
            results.append(check_item(item, xml))

    n_ok = sum(1 for r in results if r.get("ok"))
    locality_warnings = [
        {"item_id": r["item_id"], "clause": r.get("clause")}
        for r in results
        if r.get("anchor_in_cited_paragraph") is False
    ]
    return {
        "anchor_locality_warnings": locality_warnings,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ecfr_edition_date": date,
        "api": ECFR_FULL,
        "n_items": len(items),
        "n_checked": len(results),
        "n_ok": n_ok,
        "n_failed": len(results) - n_ok,
        "unparsed_citations": unparsed,
        "fetch_errors": fetch_errors,
        "sections_fetched": sorted(sections_seen),
        "results": sorted(results, key=lambda r: r["item_id"]),
        "note": (
            "Checks provenance only: that the cited section exists in the eCFR, that "
            "the item's verbatim anchor_text appears in it, and that the cited "
            "paragraph markers appear in it. It does not evaluate whether the "
            "answer key is a good answer. Copyrighted standards are never fetched."
        ),
    }


def recheck_offline(report_path: str, items_dir: str = DEFAULT_ITEMS_DIR) -> Tuple[bool, List[str]]:
    """Confirm the stored report still covers the corpus and records no failures.

    This is what the test suite runs. It cannot detect that the regulation changed
    since the report was written; only a fresh online run can do that.
    """
    problems: List[str] = []
    if not os.path.exists(report_path):
        return False, ["verification report not found: %s" % report_path]

    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)

    verified = {r["item_id"]: r for r in report.get("results", [])}
    for item in load_raw_items(items_dir):
        record = verified.get(item["id"])
        if record is None:
            problems.append("%s: no entry in the verification report" % item["id"])
            continue
        if not record.get("ok"):
            problems.append("%s: verification failed (%s)" % (item["id"], record))
        if record.get("clause") != item["source"].get("clause"):
            problems.append(
                "%s: clause changed since verification (%r -> %r)"
                % (item["id"], record.get("clause"), item["source"].get("clause"))
            )
    return (not problems), problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--date",
        default="2026-08-01",
        help="eCFR edition date to fetch (default: %(default)s)",
    )
    parser.add_argument("--out", default=REPORT_PATH, help="report path")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="do not fetch; re-check the corpus against the stored report",
    )
    parser.add_argument("--pause", type=float, default=1.0, help="seconds between fetches")
    args = parser.parse_args(argv)

    if args.offline:
        ok, problems = recheck_offline(args.out)
        for problem in problems:
            print("FAIL %s" % problem)
        print("offline recheck: %s" % ("OK" if ok else "%d problem(s)" % len(problems)))
        return 0 if ok else 1

    report = verify(args.date, pause=args.pause)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)
        handle.write("\n")

    print(
        "checked %d items across %d sections: %d ok, %d failed"
        % (
            report["n_checked"],
            len(report["sections_fetched"]),
            report["n_ok"],
            report["n_failed"],
        )
    )
    for result in report["results"]:
        if not result.get("ok"):
            print("FAIL %s %s %s" % (result["item_id"], result.get("clause"), result))
    for warning in report["anchor_locality_warnings"]:
        print(
            "WARN %s: anchor text was not found inside the cited paragraph %s; "
            "check that the clause and the anchor refer to the same place"
            % (warning["item_id"], warning["clause"])
        )
    print("report written to %s" % args.out)
    return 0 if report["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
