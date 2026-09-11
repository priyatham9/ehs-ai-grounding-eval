#!/usr/bin/env python3
"""Generate corpus statistics and markdown table.

Prints a markdown table per regulatory domain showing:
- Domain name
- Item count
- Complete minimal pairs count
- Mean question length in words
- Distinct CFR titles

Usage::

    python3 tools/corpus_stats.py
"""

import os
import re
import sys
from typing import Dict, List, Set, Tuple

# Add parent directory to path so we can import from grounding_eval
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from grounding_eval.corpus import load_corpus


def extract_cfr_title(clause: str) -> str:
    """Extract CFR title (e.g., '1910', '1926') from a clause string.

    Examples:
    '29 CFR 1910.146(c)(5)(i)(A)' -> '1910'
    '29 CFR 1926.102(b)' -> '1926'
    """
    match = re.search(r"(\d+)\.", clause)
    if match:
        return match.group(1)
    return ""


def corpus_stats() -> str:
    """Generate and return the corpus statistics markdown table."""
    corpus = load_corpus()

    # Organize by domain
    by_domain: Dict[str, List] = {}
    for item in corpus:
        if item.domain not in by_domain:
            by_domain[item.domain] = []
        by_domain[item.domain].append(item)

    # Count pairs per domain
    pairs = corpus.pairs()
    pairs_by_domain: Dict[str, int] = {}
    for a, b in pairs:
        domain = a.domain
        pairs_by_domain[domain] = pairs_by_domain.get(domain, 0) + 1

    # Build table rows
    rows: List[Tuple[str, int, int, float, int]] = []
    for domain in sorted(by_domain.keys()):
        items = by_domain[domain]
        n_items = len(items)
        n_pairs = pairs_by_domain.get(domain, 0)

        # Mean question length in words
        question_lengths = [len(item.question.split()) for item in items]
        mean_question_length = sum(question_lengths) / len(question_lengths) if question_lengths else 0

        # Distinct CFR titles
        cfr_titles: Set[str] = set()
        for item in items:
            title = extract_cfr_title(item.source.clause)
            if title:
                cfr_titles.add(title)
        n_cfr_titles = len(cfr_titles)

        rows.append((domain, n_items, n_pairs, mean_question_length, n_cfr_titles))

    # Render markdown table
    lines = [
        "| domain | items | minimal pairs | mean question length | distinct CFR titles |",
        "|---|---|---|---|---|",
    ]

    for domain, n_items, n_pairs, mean_q_len, n_cfr_titles in rows:
        lines.append(
            "| %s | %d | %d | %.1f | %d |"
            % (domain, n_items, n_pairs, mean_q_len, n_cfr_titles)
        )

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(corpus_stats(), end="")
