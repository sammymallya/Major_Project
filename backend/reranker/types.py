"""
Typed representations used by the cross-encoder re-ranking component.

The re-ranker consumes `VectorResult` candidates from the vector DB component
and emits a best-match selection with an additional cross-encoder score.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RerankedVectorResult:
    """
    A vector candidate re-scored by the cross-encoder.

    Attributes:
        id: Identifier of the stored vector (same as the underlying candidate).
        text: The candidate text content being scored against the query.
        vector_score: Original similarity score from the vector DB retrieval.
        cross_score: Cross-encoder relevance score for (query, text).
        metadata: Original metadata associated with the vector.
    """

    id: str
    text: str
    vector_score: float
    cross_score: float
    metadata: Dict[str, Any]


@dataclass
class RerankDebugInfo:
    """
    Optional debug information about a re-ranking operation.

    Attributes:
        model_name: Cross-encoder model used for scoring.
        device: Device used by the model, if known.
        input_count: Number of candidates scored.
        top_cross_scores: Top cross-encoder scores (descending).
        selected_id: ID of the selected top match, if any.
    """

    model_name: str
    device: Optional[str]
    input_count: int
    top_cross_scores: list[float]
    selected_id: Optional[str] = None

