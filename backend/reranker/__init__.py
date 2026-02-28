"""
Cross-encoder re-ranking component package.

This package provides a high-level `rerank_top_cross_encoder` function and
related types so other layers can re-score vector DB candidates without
depending on sentence-transformers internals.
"""

from .main import rerank_top_cross_encoder  # noqa: F401
from .types import RerankDebugInfo, RerankedVectorResult  # noqa: F401

