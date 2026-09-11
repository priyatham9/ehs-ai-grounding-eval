"""Tests for corpus statistics generation and README consistency."""

from __future__ import annotations

import os
import sys
import unittest

# Add tools directory to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

from corpus_stats import corpus_stats


class CorpusStatsTests(unittest.TestCase):
    def test_corpus_stats_generates_markdown_table(self):
        """Verify corpus_stats produces a valid markdown table."""
        output = corpus_stats()
        self.assertIn("| domain | items | minimal pairs", output)
        self.assertIn("|---|---|---|---|---|", output)
        # Should have 8 domain rows plus header
        lines = output.strip().split("\n")
        self.assertGreaterEqual(len(lines), 10)

    def test_readme_table_matches_generated_stats(self):
        """Verify the corpus stats table in README.md is up to date."""
        readme_path = os.path.join(REPO_ROOT, "README.md")
        with open(readme_path, "r") as f:
            readme_content = f.read()

        # Extract the corpus stats table from README using the markers
        start_marker = "<!-- corpus-stats:start -->"
        end_marker = "<!-- corpus-stats:end -->"
        start_idx = readme_content.find(start_marker)
        end_idx = readme_content.find(end_marker)

        self.assertGreater(start_idx, 0, "corpus-stats:start marker not found in README")
        self.assertGreater(end_idx, start_idx, "corpus-stats:end marker not found or out of order")

        # Extract table content
        readme_table = readme_content[start_idx + len(start_marker) : end_idx].strip()

        # Generate current stats
        generated = corpus_stats().strip()

        # Compare
        self.assertEqual(
            generated,
            readme_table,
            "README.md corpus stats table is out of date. Run `python3 tools/corpus_stats.py` to regenerate.",
        )


if __name__ == "__main__":
    unittest.main()
