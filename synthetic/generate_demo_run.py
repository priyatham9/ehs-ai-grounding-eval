#!/usr/bin/env python3
"""Generate the synthetic demonstration run.

SYNTHETIC DATA GENERATOR — THE OUTPUT OF THIS SCRIPT IS NOT A RESULT.

This runs the mock adapter over the real question corpus and writes run files to
``synthetic/demo_run/``. The mock adapter composes its answers from the corpus
item it is answering and decides correctness by a seeded coin flip against a
fixed profile probability, so its scores reproduce the constants typed into the
profile definitions. They measure nothing.

The purpose is to show what the harness produces without asking a reader to trust
an unverifiable number. See ``synthetic/README.md``.

Usage:
    python3 synthetic/generate_demo_run.py
    python3 synthetic/generate_demo_run.py --runs 5 --seed 7
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grounding_eval.adapters.mock import MockAdapter, available_profiles  # noqa: E402
from grounding_eval.corpus import load_corpus  # noqa: E402
from grounding_eval.harness import RunResult, run_repeats, write_run  # noqa: E402
from grounding_eval.report import format_summary, runs_frame  # noqa: E402
from grounding_eval.scoring import ScoringConfig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(HERE, "demo_run")

HEADER = """\
================================================================================
SYNTHETIC DEMONSTRATION RUN - NOT EXPERIMENTAL RESULTS

The adapter below is a simulation that builds its answers out of the corpus
items it is answering. Its accuracy reproduces a constant that was typed into
its profile definition. It measures nothing about any real system.

Read synthetic/README.md before interpreting anything printed here.
================================================================================
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic demonstration run.")
    parser.add_argument("--runs", type=int, default=3, help="repeats per profile")
    parser.add_argument("--seed", type=int, default=20260904, help="base seed")
    parser.add_argument("--out", default=DEMO_DIR, help="output directory")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(available_profiles()),
        help="mock profiles to simulate",
    )
    args = parser.parse_args(argv)

    print(HEADER)

    corpus = load_corpus()
    config = ScoringConfig()
    runs: List[RunResult] = []

    for profile in args.profiles:
        profile_runs = run_repeats(
            lambda repeat, p=profile: MockAdapter(profile=p, seed=args.seed, repeat=repeat),
            n_runs=args.runs,
            corpus=corpus,
            config=config,
        )
        runs.extend(profile_runs)
        for run in profile_runs:
            path = write_run(run, args.out, label=run.adapter["name"])
            print("wrote %s" % os.path.relpath(path, os.path.dirname(HERE)))

    print()
    print(format_summary(runs_frame(runs), corpus))
    print()
    print("Reminder: the table above is simulation output, not a measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
