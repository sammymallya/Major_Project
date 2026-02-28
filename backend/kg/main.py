"""
Public interface for the knowledge graph (Neo4j) component.

In this phase the module exposes fetch_kg(cypher_query) which returns
an empty list (stub). Later, replace the implementation with a real
Neo4j driver that executes the Cypher query and maps results to KgTriple.
"""

from __future__ import annotations

import logging
from typing import List

from .config import KGSettings, get_kg_settings
from .types import KgTriple

logger = logging.getLogger(__name__)

# Stub mode: no real Neo4j connection. Set to False when implementing real client.
_STUB_MODE = True


def fetch_kg(cypher_query: str) -> List[KgTriple]:
    """
    Execute a Cypher query against the KG and return triples.

    In stub mode, always returns an empty list so the pipeline can run
    without a live Neo4j instance. Later: load settings, create driver,
    run cypher_query, map records to KgTriple list.

    Args:
        cypher_query: Cypher query string produced by the Query Structurer.

    Returns:
        List of KgTriple (empty in stub mode).
    """
    if _STUB_MODE:
        logger.debug("KG stub: ignoring Cypher query, returning empty triples")
        return []

    # Placeholder for real implementation: get_kg_settings(), create driver,
    # session.run(cypher_query), map to KgTriple.
    _ = get_kg_settings()
    return []
