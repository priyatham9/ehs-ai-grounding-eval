"""Scoring components.

Each component is a small, independently testable function. ``score.py`` composes
them into a single outcome per (item, response) pair using an explicit, published
decision procedure. Nothing in this package is learned, sampled, or stochastic:
given the same corpus and the same response text, the same score comes out.
"""

from .abstention import AbstentionSignal, detect_abstention
from .citations import CitationAssessment, assess_citations
from .concepts import ConceptAssessment, assess_concepts
from .adjacency import AdjacencyAssessment, assess_adjacency
from .score import ScoringConfig, score_item, score_responses

__all__ = [
    "AbstentionSignal",
    "detect_abstention",
    "CitationAssessment",
    "assess_citations",
    "ConceptAssessment",
    "assess_concepts",
    "AdjacencyAssessment",
    "assess_adjacency",
    "ScoringConfig",
    "score_item",
    "score_responses",
]
