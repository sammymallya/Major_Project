"""
Typed output of the KG reranker component.

Represents a KG triple with an additional cross-encoder relevance score.
"""

from dataclasses import dataclass


@dataclass
class RerankedKgTriple:
    """
    A KG triple with a cross-encoder relevance score.

    Attributes:
        subject: Subject node name.
        predicate: Relation type.
        object: Object node name.
        cross_score: Cross-encoder relevance score (higher is better).
    """

    subject: str
    predicate: str
    object: str
    cross_score: float


@dataclass
class KgRerankDebugInfo:
    """
    Debug information for KG reranking.

    Attributes:
        model_name: Cross-encoder model name.
        device: Device used (cpu/cuda).
        input_count: Number of input triples.
        top_cross_scores: List of top cross-encoder scores.
        selected_subject: Subject of the top-ranked triple.
    """

    model_name: str
    device: str | None
    input_count: int
    top_cross_scores: list[float]
    selected_subject: str | None
