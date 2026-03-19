"""
Semantic Matching module.

Computes semantic similarity between expected and actual answers using embeddings.
"""

from .main import compute_semantic_similarity

__all__ = ["compute_semantic_similarity"]