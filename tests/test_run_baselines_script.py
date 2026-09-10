"""Tests for scripts/run_baselines.sh: existence, executability, and the
refuse-without-a-key guardrail. This script makes real, billed calls to the
Anthropic API, so it must never run without ANTHROPIC_API_KEY set.
"""

from __future__ import annotations

import os
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "run_baselines.sh")
GITIGNORE_PATH = os.path.join(REPO_ROOT, ".gitignore")


def _shell_quote(value: str) -> str:
    """Minimal POSIX single-quote shell escaping (no shlex import needed)."""
    return "'" + value.replace("'", "'\"'\"'") + "'"


class TestRunBaselinesScript(unittest.TestCase):
    def test_script_exists(self) -> None:
        self.assertTrue(os.path.isfile(SCRIPT_PATH), "scripts/run_baselines.sh is missing")

    def test_script_is_executable(self) -> None:
        self.assertTrue(
            os.access(SCRIPT_PATH, os.X_OK),
            "scripts/run_baselines.sh must be executable (chmod +x)",
        )

    def test_refuses_without_api_key(self) -> None:
        capture_path = os.path.join(REPO_ROOT, ".test_run_baselines_output.tmp")
        try:
            path = os.environ.get("PATH", "")
            command = (
                "cd %s && env -i PATH=%s bash %s >%s 2>&1; echo exit:$?"
                % (
                    _shell_quote(REPO_ROOT),
                    _shell_quote(path),
                    _shell_quote(SCRIPT_PATH),
                    _shell_quote(capture_path),
                )
            )
            status_line = os.popen(command).read()
            with open(capture_path, encoding="utf-8") as handle:
                combined_output = handle.read()
            exit_code = int(status_line.strip().rsplit(":", 1)[-1])
            self.assertNotEqual(exit_code, 0)
            self.assertIn("ANTHROPIC_API_KEY", combined_output)
        finally:
            if os.path.exists(capture_path):
                os.remove(capture_path)


class TestPublishedResultsTracked(unittest.TestCase):
    def test_gitignore_tracks_results_published(self) -> None:
        with open(GITIGNORE_PATH, encoding="utf-8") as handle:
            contents = handle.read()
        self.assertIn("results/published", contents)
        self.assertIn("!results/published/*.json", contents)


if __name__ == "__main__":
    unittest.main()
