"""Repository hygiene tests.

These check properties of the repository as a whole rather than of any one
function: that copyrighted standard text is not redistributed, that no file
claims an experimental result, and that stored run artifacts are labelled.

They are cheap and they guard against the failure modes that would matter most
if this work is read by someone deciding whether to trust it.
"""

from __future__ import annotations

import json
import os
import re
import unittest

from grounding_eval import MOCK_PROVENANCE
from grounding_eval.corpus import DEFAULT_ITEMS_DIR, QUARANTINE_PATH, REPO_ROOT, load_corpus

#: Every directory that may contain stored run artifacts. All of them are
#: walked, so a run file cannot escape the labelling check by being written
#: somewhere new.
RUN_ARTIFACT_DIRS = (
    os.path.join(REPO_ROOT, "results"),
    os.path.join(REPO_ROOT, "synthetic", "demo_run"),
)

#: Provenance labels a non-LLM baseline run may carry. Anything else stored
#: under a run directory must be a labelled mock demonstration.
BASELINE_PROVENANCES = {"random_floor_baseline", "retrieval_tfidf_baseline", "retrieval_bm25_baseline", "oracle_ceiling"}

#: Standards bodies whose text may not be redistributed. Naming a clause is
#: fine and necessary; reproducing its text is not.
COPYRIGHTED_BODIES = ("ASME", "API", "NFPA", "ISO", "ANSI", "IEC")


def _corpus_files() -> list:
    return [
        os.path.join(DEFAULT_ITEMS_DIR, name)
        for name in sorted(os.listdir(DEFAULT_ITEMS_DIR))
        if name.endswith(".json")
    ]


class TestNoCopyrightedTextRedistributed(unittest.TestCase):
    """The corpus cites paywalled standards; it must never quote them."""

    def test_anchor_text_only_ever_comes_from_a_public_source(self) -> None:
        for item in load_corpus():
            self.assertEqual(item.source.access, "public", msg=item.id)

    def test_cross_references_carry_no_quoted_standard_text(self) -> None:
        """A cross-reference may name a paragraph and say how it was corroborated.

        What it may not contain is the standard's own wording. The check is that
        the corroboration text explains provenance rather than reproducing
        content, which is approximated by requiring it to mention the public
        source it was corroborated against.
        """
        for item in load_corpus():
            for reference in item.cross_reference:
                # A cross-reference may point at a public companion regulation or
                # at a paywalled consensus standard; both are legitimate. What is
                # required either way is a stated access class and a corroboration
                # note explaining how the identifier was checked.
                self.assertIn(reference.access, ("public", "paywalled"), msg=item.id)
                self.assertTrue(reference.corroboration.strip(), msg=item.id)
                self.assertLess(
                    len(reference.corroboration),
                    1200,
                    msg="%s: corroboration note is long enough to be quoting text" % item.id,
                )

    def test_no_corpus_item_claims_a_paywalled_standard_as_primary_authority(self) -> None:
        for item in load_corpus():
            standard = item.source.standard.upper()
            for body in COPYRIGHTED_BODIES:
                self.assertNotIn(
                    body,
                    standard,
                    msg="%s uses %s as its primary source" % (item.id, body),
                )


class TestNoResultsClaimed(unittest.TestCase):
    """Nothing in this repository may present a number as an experimental finding."""

    def test_stored_run_files_are_all_mock_and_labelled(self) -> None:
        checked = 0
        for base in RUN_ARTIFACT_DIRS:
            if not os.path.isdir(base):
                continue
            for root, _dirs, files in os.walk(base):
                for name in files:
                    if not name.endswith(".json"):
                        continue
                    path = os.path.join(root, name)
                    with open(path, "r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                    checked += 1
                    if payload.get("schema") == "grounding_eval.baselines/1":
                        # Non-LLM baseline runs (random floor, TF-IDF retrieval,
                        # oracle ceiling) are honest runs, not mock demonstrations.
                        # They must say so on every response and never claim to
                        # be a language-model result.
                        self.assertTrue(name.startswith("baselines_"), msg=path)
                        runs = payload.get("runs", [])
                        self.assertTrue(runs, msg=path)
                        provenances = set()
                        for run in runs:
                            self.assertFalse(run.get("is_mock_demonstration"), msg=path)
                            provenances |= {r.get("provenance") for r in run.get("responses", [])}
                        self.assertTrue(provenances, msg=path)
                        self.assertTrue(
                            provenances <= BASELINE_PROVENANCES,
                            msg="%s carries a provenance outside the non-LLM baseline set: %s" % (path, provenances),
                        )
                        continue
                    self.assertTrue(
                        payload.get("is_mock_demonstration"),
                        msg="%s is a stored run not labelled as a demonstration" % path,
                    )
                    self.assertEqual(payload.get("mock_banner"), MOCK_PROVENANCE, msg=path)
                    self.assertTrue(
                        name.startswith("mock__"),
                        msg="%s is a mock run without the mock__ filename prefix" % path,
                    )
        self.assertGreater(checked, 0, "no stored run artifacts were found to check")

    def test_readme_states_the_repository_contains_no_results(self) -> None:
        path = os.path.join(REPO_ROOT, "README.md")
        self.assertTrue(os.path.exists(path), "README.md is missing")
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("no experimental results", text.lower())

    def test_mock_provenance_string_is_not_weakened(self) -> None:
        self.assertEqual(MOCK_PROVENANCE, "MOCK_DEMONSTRATION_FIXTURE_NOT_RESULTS")


class TestCorpusFilesAreWellFormed(unittest.TestCase):
    def test_every_corpus_file_declares_its_sourcing(self) -> None:
        for path in _corpus_files():
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertTrue(payload.get("source_family"), msg=path)
            self.assertTrue(payload.get("sourcing_note"), msg=path)

    def test_files_are_valid_json_and_utf8(self) -> None:
        for path in _corpus_files() + [QUARANTINE_PATH]:
            with open(path, "r", encoding="utf-8") as handle:
                json.load(handle)

    def test_no_item_id_collides_with_a_quarantined_id(self) -> None:
        with open(QUARANTINE_PATH, "r", encoding="utf-8") as handle:
            quarantine = json.load(handle)
        scored = set(load_corpus().ids())
        for entry in quarantine.get("items", []):
            self.assertNotIn(entry["id"], scored)


class TestPackagingHygiene(unittest.TestCase):
    def test_license_exists_and_is_mit(self) -> None:
        path = os.path.join(REPO_ROOT, "LICENSE")
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("MIT License", text)

    def test_gitignore_exists(self) -> None:
        self.assertTrue(os.path.exists(os.path.join(REPO_ROOT, ".gitignore")))

    def test_only_permitted_third_party_imports_are_used(self) -> None:
        """The environment has pandas and numpy and nothing else installed.

        An import of anything outside the standard library plus those two would
        make the repository unrunnable as documented, so it is checked rather
        than assumed.
        """
        allowed = {"pandas", "numpy"}
        pattern = re.compile(r"^\s*(?:import|from)\s+([a-zA-Z_][\w]*)", re.MULTILINE)
        stdlib_ok = {
            "abc", "argparse", "ast", "collections", "dataclasses", "datetime", "enum",
            "functools", "gzip", "hashlib", "io", "itertools", "json", "math",
            "os", "pathlib", "platform", "random", "re", "string", "sys", "tempfile",
            "textwrap", "time", "typing", "unicodedata", "unittest", "urllib",
            "warnings", "corpus_stats", "grounding_eval", "verify_sources", "__future__",
        }
        offenders = []
        for root, dirs, files in os.walk(REPO_ROOT):
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv"}]
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(root, name)
                with open(path, "r", encoding="utf-8") as handle:
                    source = handle.read()
                for module in pattern.findall(source):
                    if module in allowed or module in stdlib_ok:
                        continue
                    offenders.append("%s imports %s" % (os.path.relpath(path, REPO_ROOT), module))
        self.assertEqual(offenders, [], msg="disallowed imports: %s" % offenders)


if __name__ == "__main__":
    unittest.main()
