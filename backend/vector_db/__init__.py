"""
Vector database (Pinecone) component package.

This package exposes a high-level `fetch_top_vectordb` function and related
types that the rest of the system can use to perform semantic search over
the vector index without depending on Pinecone-specific details.
"""

from .main import fetch_top_vectordb  # noqa: F401
from .types import VectorQueryDebugInfo, VectorResult  # noqa: F401

