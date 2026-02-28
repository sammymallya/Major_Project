"""
Typed representations used by the knowledge graph component.

Lightweight types for KG query results so the rest of the system
does not depend on Neo4j-specific structures.
"""

from dataclasses import dataclass


@dataclass
class KgTriple:
    """
    A single triple (subject, predicate, object) from the KG.

    Attributes:
        subject: Subject node or value.
        predicate: Relation type.
        object: Object node or value.
    """

    subject: str
    predicate: str
    object: str
