"""
Typed representations used by the vector database component.

These types are intentionally lightweight so they can be shared across
the wider system without pulling in any heavy third-party dependencies.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class VectorResult:
    """
    A single result returned from the vector database.

    Attributes:
        id: Identifier of the stored vector (as returned by Pinecone).
        text: The primary text content associated with the vector.
        score: Similarity score where higher values indicate closer matches.
        metadata: Arbitrary metadata stored alongside the vector.
    """

    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class VectorQueryDebugInfo:
    """
    Optional debug information about a vector query.

    Attributes:
        model_name: The embedding model used to encode the query.
        top_scores: The top similarity scores returned for the query.
        namespace: The Pinecone namespace that was queried, if any.
    """

    model_name: str
    top_scores: list[float]
    namespace: Optional[str] = None

