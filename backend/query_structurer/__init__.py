"""
Query Structurer (LLM-powered) component package.

Exposes structure_query(query, output_kind) and StructuredQuery for use
by the orchestration layer.
"""

from .main import structure_query  # noqa: F401
from .types import StructuredQuery  # noqa: F401
