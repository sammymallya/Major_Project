"""
Typed inputs/outputs for the Prompt Generator component.
"""

from typing import List

from backend.kg.types import KgTriple


class PromptContext:
    """
    Context data for prompt generation.

    Attributes:
        query: Original user query.
        vector_snippet: Top vector result text, if any.
        kg_triples: List of top KG triples, if any.
    """

    def __init__(self, query: str, vector_snippet: str | None = None, kg_triples: List[KgTriple] | None = None):
        self.query = query
        self.vector_snippet = vector_snippet
        self.kg_triples = kg_triples or []