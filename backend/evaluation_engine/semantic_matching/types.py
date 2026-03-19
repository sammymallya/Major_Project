"""
Typed representations for the Semantic Matching component.
"""

from dataclasses import dataclass


@dataclass
class SemanticMatchResult:
    """
    Result of semantic matching between expected and actual answers.

    Attributes:
        score: Cosine similarity score (0.0 to 1.0).
        model_name: Embedding model used.
    """

    score: float
    model_name: str