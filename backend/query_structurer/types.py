"""
Typed output of the Query Structurer component.

StructuredQuery holds the optional semantic search string and/or
Cypher query for use by the orchestration layer.
"""

from dataclasses import dataclass


@dataclass
class StructuredQuery:
    """
    Output of the Query Structurer LLM.

    Attributes:
        semantic_search_query: Cleaned query for vector DB search, or None if not requested.
        cypher_query: Cypher query for Neo4j, or None if not requested.
    """

    semantic_search_query: str | None
    cypher_query: str | None
