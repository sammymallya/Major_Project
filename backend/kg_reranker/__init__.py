"""
KG reranker component package.

Exposes rerank_kg_triples() for use by the orchestration layer.
"""

from .main import rerank_kg_triples  # noqa: F401
from .types import RerankedKgTriple, KgRerankDebugInfo  # noqa: F401
