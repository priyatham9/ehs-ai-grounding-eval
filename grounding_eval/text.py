"""Deterministic text normalisation and phrase matching.

The matcher here is lexical, not distributional. There are no embeddings and no
learned components anywhere in the scoring path. That is a deliberate design
choice with a real cost, and the cost is stated in docs/limitations.md: a
paraphrase that avoids every listed alternative for a concept group will be
scored as a miss.

The reason for accepting that cost is that the benchmark's subject is whether a
system distinguishes near-synonymous technical entities. Scoring that with an
embedding model would use, as the measuring instrument, the same mechanism whose
failure is under study. A concept-coverage matcher with explicit, auditable
alternation lists keeps the answer key readable and every score reproducible from
the corpus file alone.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, Sequence, Tuple

# Characters that appear in regulatory text and would otherwise defeat naive
# substring matching. Mapped to ASCII before anything else happens.
_CHAR_MAP = {
    "\u2018": "'",       # left single quote
    "\u2019": "'",       # right single quote / apostrophe
    "\u201a": "'",       # single low quote
    "\u201c": '"',       # left double quote
    "\u201d": '"',       # right double quote
    "\u2013": "-",       # en dash
    "\u2014": "-",       # em dash
    "\u2212": "-",       # minus sign
    "\u00a0": " ",       # no-break space
    "\u2007": " ",       # figure space
    "\u202f": " ",       # narrow no-break space
    "\u00b0": " degrees ",
    "\u00ba": " degrees ",
    "\u2044": "/",       # fraction slash
    "\ufeff": "",        # byte order mark
    "\u00ad": "",        # soft hyphen
    "\u00a7": " section ",
}

_WS_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")

# Cues that turn a mention into a contrast rather than an assertion. A forbidden
# or adjacent-signature phrase preceded closely by one of these is not counted.
CONTRAST_CUES: Tuple[str, ...] = (
    "unlike",
    "not a",
    "not an",
    "is not",
    "are not",
    "does not",
    "do not",
    "cannot",
    "no such",
    "rather than",
    "instead of",
    "as opposed to",
    "whereas",
    "in contrast",
    "by contrast",
    "differs from",
    "different from",
    "would be wrong",
    "incorrect to",
    "there is no",
    "has no",
    "have no",
    "without a",
    "never",
)

# How many characters before a match are inspected for a contrast cue.
CONTRAST_WINDOW_CHARS = 70


def normalize(text: str) -> str:
    """Fold text to a comparable form: NFKC, ASCII punctuation, lowercase, single spaces.

    The result keeps punctuation, because clause identifiers such as
    ``1910.147(c)(6)`` depend on it. It is intended for substring and regex
    matching, not for display.
    """
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    for src, dst in _CHAR_MAP.items():
        out = out.replace(src, dst)
    out = out.lower()
    out = _WS_RE.sub(" ", out)
    return out.strip()


def _phrase_pattern(phrase: str) -> re.Pattern:
    """Build a boundary-aware regex for one concept alternative.

    A leading boundary is always applied, so ``5 years`` does not match inside
    ``15 years`` and ``10`` does not match inside ``110``.

    A trailing boundary is applied only when the phrase ends in a digit. Phrases
    ending in a letter are prefix-matched, so ``annual`` matches ``annually`` and
    ``eliminat`` matches ``eliminated``. That asymmetry is intentional: numeric
    facts must match exactly, and morphological variants of English words should
    not each need their own entry in the corpus.
    """
    normalized = normalize(phrase)
    parts = [re.escape(tok) for tok in normalized.split(" ") if tok]
    body = r"\s+".join(parts)
    lead = r"(?<![a-z0-9])"
    trail = r"(?![0-9])" if normalized and normalized[-1].isdigit() else ""
    return re.compile(lead + body + trail)


def find_phrase(haystack_norm: str, phrase: str) -> List[Tuple[int, int]]:
    """Return the (start, end) spans where ``phrase`` occurs in normalised text."""
    if not phrase.strip():
        return []
    return [m.span() for m in _phrase_pattern(phrase).finditer(haystack_norm)]


def contains_phrase(haystack_norm: str, phrase: str) -> bool:
    return bool(find_phrase(haystack_norm, phrase))


def is_contrasted(haystack_norm: str, start: int, window: int = CONTRAST_WINDOW_CHARS) -> bool:
    """True when a contrast cue appears shortly before position ``start``.

    This is a heuristic, and it is the single weakest link in automatic scoring.
    It exists so that a correct answer of the form "a rupture disk has no
    blowdown, unlike a relief valve" is not penalised for containing the word
    blowdown. It will occasionally misfire in both directions. Every reported
    adjacent-substitution rate should therefore be checked against a
    human-adjudicated subsample; see grounding_eval.adjudication.
    """
    left = haystack_norm[max(0, start - window):start]
    return any(cue in left for cue in CONTRAST_CUES)


def find_asserted(haystack_norm: str, phrase: str) -> bool:
    """True when ``phrase`` occurs at least once outside a contrast construction."""
    for start, _end in find_phrase(haystack_norm, phrase):
        if not is_contrasted(haystack_norm, start):
            return True
    return False


def _build_number_words() -> Dict[str, float]:
    """Spelled-out integers 0-99, plus the round hundreds this domain uses.

    Regulatory prose mixes digits and words freely: 29 CFR 1904.39 writes
    "within twenty-four (24) hours", and a system answering "twenty-four hours"
    has given the right number. Scoring that as a numeric miss would measure
    formatting rather than correctness.

    The range is generated rather than hand-listed so that adding an item with a
    new numeric fact does not require editing a lookup table. It stops at 99 plus
    a few round hundreds on purpose: a fully general parser would have to handle
    constructions like "one and one-half" and "twenty-five hundred", and getting
    those subtly wrong is worse than not attempting them. Both hyphenated and
    spaced compounds are accepted, since usage varies.
    """
    units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
    ]
    tens = {
        20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
        60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
    }
    words: Dict[str, float] = {word: float(i) for i, word in enumerate(units)}
    for base, name in tens.items():
        words[name] = float(base)
        for offset in range(1, 10):
            words["%s-%s" % (name, units[offset])] = float(base + offset)
            words["%s %s" % (name, units[offset])] = float(base + offset)
    for hundred in (100, 120, 180, 200, 250, 300, 500):
        words["%d hundred" % (hundred // 100)] = float(hundred - hundred % 100)
    words["one hundred"] = 100.0
    words["one hundred twenty"] = 120.0
    words["one hundred eighty"] = 180.0
    return words


#: Spelled-out integers recognised by :func:`extract_numbers`.
NUMBER_WORDS: Dict[str, float] = _build_number_words()

_NUMBER_WORD_RE = re.compile(
    r"(?<![\w-])(" + "|".join(
        re.escape(word) for word in sorted(NUMBER_WORDS, key=len, reverse=True)
    ) + r")(?![\w-])"
)


def extract_numbers(text_norm: str) -> List[float]:
    """Pull every number out of normalised text, handling comma grouping.

    ``10,000 pounds`` yields ``10000.0``. Spelled-out integers listed in
    :data:`NUMBER_WORDS` are recognised too, so "sixteen sections" and "16
    sections" are treated as the same claim.

    Numbers embedded in clause identifiers such as ``1910.147`` are extracted
    too; that is harmless, because numeric checks only ask whether a target value
    is present, never whether extra numbers are absent.
    """
    values: List[float] = []
    for match in _NUMBER_RE.finditer(text_norm):
        raw = match.group(1).replace(",", "")
        try:
            values.append(float(raw))
        except ValueError:  # pragma: no cover - regex guarantees parseability
            continue
    for match in _NUMBER_WORD_RE.finditer(text_norm):
        values.append(NUMBER_WORDS[match.group(1)])
    return values


def has_number(text_norm: str, target: float, tolerance_abs: float = 0.0) -> bool:
    """True when ``target`` appears in the text within ``tolerance_abs``."""
    for value in extract_numbers(text_norm):
        if abs(value - target) <= tolerance_abs:
            return True
    return False


def render_number_variants(value: float) -> List[str]:
    """Plain-text renderings of a number, for presence checks against source text.

    ``10000.0`` renders as ``10000`` and ``10,000``; ``1.3`` renders as ``1.3``.
    Used by the clause verifier to look for a numeric fact in primary-source text
    that may or may not use comma grouping.
    """
    out: List[str] = []
    if float(value).is_integer():
        as_int = int(value)
        out.append(str(as_int))
        out.append("{:,}".format(as_int))
    else:
        out.append(("%f" % value).rstrip("0").rstrip("."))
    seen = set()
    deduped = []
    for token in out:
        if token not in seen:
            seen.add(token)
            deduped.append(token)
    return deduped


def any_present(haystack_norm: str, alternatives: Sequence[str]) -> bool:
    return any(contains_phrase(haystack_norm, alt) for alt in alternatives)


def any_asserted(haystack_norm: str, alternatives: Iterable[str]) -> List[str]:
    """Return the alternatives that appear as assertions rather than contrasts."""
    return [alt for alt in alternatives if find_asserted(haystack_norm, alt)]
