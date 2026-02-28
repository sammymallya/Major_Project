"""
Knowledge graph (Neo4j) component package.

Exposes fetch_kg(cypher_query) and KgTriple for use by the orchestration
layer. Currently implemented as a stub returning no triples.
"""

from .main import fetch_kg  # noqa: F401
from .types import KgTriple  # noqa: F401
