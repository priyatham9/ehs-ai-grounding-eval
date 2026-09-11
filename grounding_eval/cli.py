"""Command line entry points.

    python3 -m grounding_eval.cli corpus            # describe the corpus
    python3 -m grounding_eval.cli validate          # validate corpus + verification report
    python3 -m grounding_eval.cli demo              # run the mock fixture end to end
    python3 -m grounding_eval.cli run --adapter anthropic --model <id> [--grounded]
                                                      # run a real system under test
    python3 -m grounding_eval.cli report <dir>      # re-report stored run files

The ``demo`` subcommand is the only one that produces numbers from a fixture, and
everything it writes is labelled as a demonstration. The ``run`` subcommand
produces numbers from a real system under test (currently only ``--adapter
anthropic``) and is never labelled as a mock.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Sequence

from . import __version__
from .adapters.mock import MockAdapter, available_profiles
from .corpus import REPO_ROOT, load_corpus, load_quarantine, summarize
from .harness import RunResult, corpus_digest, load_run, run_repeats, write_run
from .report import compare_arms, format_summary, runs_frame
from .schema import AdapterResponse, ItemScore
from .scoring import ScoringConfig

DEFAULT_RESULTS_DIR = os.path.join(REPO_ROOT, "results", "demo")
DEFAULT_REAL_RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def _cmd_corpus(args: argparse.Namespace) -> int:
    corpus = load_corpus()
    info = summarize(corpus)
    info["corpus_digest"] = corpus_digest(corpus)
    if args.json:
        print(json.dumps(info, indent=2))
        return 0
    print("ehs-ai-grounding-eval corpus v%s" % __version__)
    print("  items                %d" % info["n_items"])
    print("  factual              %d" % info["n_factual"])
    print("  category-error       %d" % info["n_category_error"])
    print("  complete minimal pairs %d" % info["n_minimal_pairs"])
    print("  domains              %d" % info["n_domains"])
    print("  families             %d" % info["n_families"])
    print("  digest               %s" % info["corpus_digest"])
    print()
    print("  by domain:")
    for domain, count in info["by_domain"].items():
        print("    %-34s %d" % (domain, count))
    print()
    print("  by risk tier (author judgement, not a measurement):")
    for tier, count in info["by_risk_tier"].items():
        print("    %-34s %d" % (tier, count))
    quarantine = load_quarantine()
    print()
    print("  quarantined (excluded from scoring): %d" % len(quarantine.get("items", [])))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Load the corpus and re-check it against the stored source verification report."""
    sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
    from verify_sources import recheck_offline  # noqa: E402

    corpus = load_corpus()
    print("corpus loads: %d items, digest %s" % (len(corpus), corpus_digest(corpus)))

    ok, problems = recheck_offline(
        os.path.join(REPO_ROOT, "corpus", "verification", "source_verification.json")
    )
    for problem in problems:
        print("FAIL %s" % problem)
    print("source verification: %s" % ("all items verified" if ok else "%d problem(s)" % len(problems)))
    return 0 if ok else 1


def _cmd_demo(args: argparse.Namespace) -> int:
    corpus = load_corpus()
    config = ScoringConfig()
    runs: List[RunResult] = []

    for profile in args.profiles:
        def factory(repeat: int, profile: str = profile) -> MockAdapter:
            return MockAdapter(profile=profile, seed=args.seed, repeat=repeat)

        profile_runs = run_repeats(
            factory,
            n_runs=args.runs,
            corpus=corpus,
            config=config,
            progress=not args.quiet,
        )
        runs.extend(profile_runs)
        if args.out:
            for run in profile_runs:
                path = write_run(run, args.out, label=run.adapter["name"])
                if not args.quiet:
                    sys.stderr.write("wrote %s\n" % path)

    frame = runs_frame(runs)
    print(format_summary(frame, corpus))

    if len(args.profiles) >= 2:
        print()
        print("Paired comparison (repeat 0, factual items only)")
        print()
        comparison = compare_arms(
            frame,
            adapter_a="mock:%s" % args.profiles[0],
            adapter_b="mock:%s" % args.profiles[-1],
            repeat=0,
        )
        print(json.dumps(comparison, indent=2))
        print()
        print(
            "The p-value above describes a simulation whose outcome probabilities\n"
            "were fixed in advance by the mock profiles. It is arithmetic on\n"
            "fabricated data and is not evidence about anything."
        )
    return 0


def _build_adapter(args: argparse.Namespace, corpus):
    if args.adapter == "anthropic":
        from .adapters.anthropic_api import DEFAULT_MODEL, AnthropicAdapter

        return AnthropicAdapter(
            model=args.model or DEFAULT_MODEL,
            arm="grounded" if args.grounded else "ungrounded",
            corpus=corpus,
        )
    raise ValueError("unknown adapter %r" % args.adapter)


def _cmd_run(args: argparse.Namespace) -> int:
    """Run a real system under test (currently only the Anthropic adapter).

    Unlike ``demo``, this never produces a mock-labelled run: the adapter's
    responses carry a real provenance string, so ``RunResult.is_mock`` is
    false and the run file omits the demonstration banner.
    """
    corpus = load_corpus()
    adapter = _build_adapter(args, corpus)

    if args.dry_run:
        from .adapters.anthropic_api import estimate_input_tokens

        prompts = [adapter.build_prompt(item) for item in corpus]
        total_estimated_input_tokens = sum(
            estimate_input_tokens(p) + estimate_input_tokens(adapter.describe().get("model", ""))
            for p in prompts
        )
        print("dry run: adapter=%s model=%s grounded=%s" % (adapter.name, adapter.model, args.grounded))
        print("items: %d" % len(prompts))
        print("estimated input tokens (chars/4, prompts only): %d" % total_estimated_input_tokens)
        print()
        print("first prompt:")
        print("-" * 72)
        print(prompts[0] if prompts else "(corpus is empty)")
        print("-" * 72)
        return 0

    config = ScoringConfig()

    def factory(repeat: int) -> object:
        return _build_adapter(args, corpus)

    runs = run_repeats(
        factory,
        n_runs=args.runs,
        corpus=corpus,
        config=config,
        progress=not args.quiet,
    )

    out_dir = args.out or DEFAULT_REAL_RESULTS_DIR
    for run in runs:
        path = write_run(run, out_dir, label=run.adapter["name"])
        if not args.quiet:
            sys.stderr.write("wrote %s\n" % path)

    frame = runs_frame(runs)
    print(format_summary(frame, corpus))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    corpus = load_corpus()
    paths = sorted(
        os.path.join(args.directory, name)
        for name in os.listdir(args.directory)
        if name.endswith(".json")
    )
    if not paths:
        print("no run files in %s" % args.directory, file=sys.stderr)
        return 1

    runs: List[RunResult] = []
    for path in paths:
        payload = load_run(path, corpus=corpus if args.check_digest else None)
        runs.append(
            RunResult(
                adapter=payload["adapter"],
                corpus_digest=payload["corpus_digest"],
                n_items=payload["n_items"],
                repeat=payload["repeat"],
                seed=payload.get("seed"),
                scoring_config=payload["scoring_config"],
                responses=[AdapterResponse.from_dict(r) for r in payload["responses"]],
                scores=[ItemScore.from_dict(s) for s in payload["scores"]],
                started_utc=payload["started_utc"],
                duration_s=payload["duration_s"],
                environment=payload.get("environment", {}),
            )
        )
    print(format_summary(runs_frame(runs), corpus))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grounding_eval",
        description="Apparatus for measuring source grounding in safety-critical answers.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    corpus_parser = subparsers.add_parser("corpus", help="describe the question corpus")
    corpus_parser.add_argument("--json", action="store_true", help="emit JSON")
    corpus_parser.set_defaults(func=_cmd_corpus)

    validate_parser = subparsers.add_parser(
        "validate", help="validate the corpus and its stored source verification"
    )
    validate_parser.set_defaults(func=_cmd_validate)

    demo_parser = subparsers.add_parser(
        "demo", help="run the mock demonstration fixture (produces no results)"
    )
    demo_parser.add_argument(
        "--profiles",
        nargs="+",
        default=["ungrounded", "pseudo_grounded", "grounded"],
        choices=list(available_profiles()),
        help="mock profiles to simulate",
    )
    demo_parser.add_argument("--runs", type=int, default=3, help="repeats per profile")
    demo_parser.add_argument("--seed", type=int, default=20260904, help="base seed")
    demo_parser.add_argument(
        "--out",
        nargs="?",
        const=DEFAULT_RESULTS_DIR,
        default=None,
        help="write run files to this directory",
    )
    demo_parser.add_argument("--quiet", action="store_true")
    demo_parser.set_defaults(func=_cmd_demo)

    run_parser = subparsers.add_parser("run", help="run a real system under test")
    run_parser.add_argument(
        "--adapter", choices=["anthropic"], default="anthropic", help="which adapter to run"
    )
    run_parser.add_argument(
        "--model", default=None, help="model id (defaults to the adapter's own default)"
    )
    run_parser.add_argument(
        "--grounded", action="store_true", help="use the grounded arm (retrieved source excerpt in prompt)"
    )
    run_parser.add_argument("--runs", type=int, default=1, help="repeats")
    run_parser.add_argument(
        "--out",
        nargs="?",
        const=DEFAULT_REAL_RESULTS_DIR,
        default=None,
        help="write run files to this directory (default: results/)",
    )
    run_parser.add_argument("--quiet", action="store_true")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build every prompt and print the first one plus a count and estimated "
        "input tokens; makes no network calls",
    )
    run_parser.set_defaults(func=_cmd_run)

    report_parser = subparsers.add_parser("report", help="re-report stored run files")
    report_parser.add_argument("directory")
    report_parser.add_argument(
        "--check-digest",
        action="store_true",
        help="fail if a run file was produced against a different corpus",
    )
    report_parser.set_defaults(func=_cmd_report)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
