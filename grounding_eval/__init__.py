"""ehs-ai-grounding-eval: apparatus for measuring whether an AI system's answers to
safety-critical technical questions are grounded in authoritative sources.

This package contains an evaluation harness and a question corpus. It contains no
experimental results. The only run output it can produce out of the box comes from
a deliberately synthetic mock adapter, which is a demonstration fixture and is
labelled as such in every artifact it produces.

Read docs/methodology.md before interpreting anything this package prints.
"""

__version__ = "0.1.0"

# Every artifact produced by a mock-adapter run carries this string. Tests assert
# on it. Do not weaken or remove it.
MOCK_PROVENANCE = "MOCK_DEMONSTRATION_FIXTURE_NOT_RESULTS"

__all__ = ["__version__", "MOCK_PROVENANCE"]
