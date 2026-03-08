"""
Public interface for the knowledge graph (Neo4j) component.

This module provides a small Neo4j-backed client used by the orchestration
layer. It exposes entity extraction and a simple question->cypher pipeline
as well as `fetch_kg(cypher_query)` which maps query results to `KgTriple`.
"""

from __future__ import annotations

import logging
from typing import List, Any

from .config import KGSettings, get_kg_settings
from .types import KgTriple

logger = logging.getLogger(__name__)

# Driver and cached schema values
_driver = None
_schema_cache = {
    "cities": None,
    "states": None,
    "types": None,
    "districts": None,
}


def _get_driver():
    """Lazily create and return a Neo4j driver instance.

    Raises ImportError if `neo4j` package is not installed.
    """
    global _driver
    if _driver is not None:
        return _driver

    try:
        from neo4j import GraphDatabase
    except Exception as e:  # pragma: no cover - runtime dependency
        logger.error("neo4j driver not available: %s", e)
        raise

    settings = get_kg_settings()
    logger.info("Connecting to Neo4j at %s (user=%s)", settings.neo4j_uri, settings.neo4j_username)
    _driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password))
    return _driver


def _load_schema_values():
    """Load distinct schema values (cities, states, types, districts) and cache them."""
    if _schema_cache["cities"] is not None:
        return

    driver = _get_driver()
    with driver.session() as session:
        try:
            ALL_CITIES = [r["name"].lower() for r in session.run("MATCH (c:City) RETURN DISTINCT c.name AS name") if r["name"]]
        except Exception:
            ALL_CITIES = []
        try:
            ALL_STATES = [r["name"].lower() for r in session.run("MATCH (s:State) RETURN DISTINCT s.name AS name") if r["name"]]
        except Exception:
            ALL_STATES = []
        try:
            ALL_TYPES = [r["type"].lower() for r in session.run("MATCH (p:Place) RETURN DISTINCT p.type AS type") if r["type"]]
        except Exception:
            ALL_TYPES = []
        try:
            ALL_DISTRICTS = [r["name"].lower() for r in session.run("MATCH (d:District) RETURN DISTINCT d.name AS name") if r["name"]]
        except Exception:
            ALL_DISTRICTS = []

    _schema_cache["cities"] = ALL_CITIES
    _schema_cache["states"] = ALL_STATES
    _schema_cache["types"] = ALL_TYPES
    _schema_cache["districts"] = ALL_DISTRICTS


def extract_entities(question: str) -> dict:
    """Deterministic entity extraction using cached schema values."""
    _load_schema_values()
    q = question.lower()
    intent = {"city": None, "state": None, "type": None, "district": None}

    for city in (_schema_cache["cities"] or []):
        if f" {city} " in f" {q} ":
            intent["city"] = city.title()

    for state in (_schema_cache["states"] or []):
        if f" {state} " in f" {q} ":
            intent["state"] = state.title()

    for t in (_schema_cache["types"] or []):
        if f" {t} " in f" {q} ":
            intent["type"] = t.title()

    for district in (_schema_cache["districts"] or []):
        if f" {district} " in f" {q} ":
            intent["district"] = district.title()

    return intent


def build_query(intent: dict) -> tuple[str, dict]:
    """Build a Cypher query and parameter dict from extracted intent."""
    query = """
    MATCH (p:Place)
    OPTIONAL MATCH (p)-[:LOCATED_IN]->(c:City)
    OPTIONAL MATCH (c)-[:IN_DISTRICT]->(d:District)
    OPTIONAL MATCH (d)-[:IN_STATE]->(s:State)
    WHERE 1=1
    """

    params: dict = {}
    if intent.get("city"):
        query += " AND c.name = $city"
        params["city"] = intent["city"]
    if intent.get("state"):
        query += " AND s.name = $state"
        params["state"] = intent["state"]
    if intent.get("type"):
        query += " AND p.type = $type"
        params["type"] = intent["type"]
    if intent.get("district"):
        query += " AND d.name = $district"
        params["district"] = intent["district"]

    query += """
    RETURN DISTINCT p.name AS name,
           p.description AS description,
           p.best_time AS best_time,
           p.entry_fee AS entry_fee
    LIMIT 20
    """

    return query, params


def run_query(query: str, params: dict | None = None) -> list[dict]:
    """Execute a Cypher query and return list of record dicts."""
    driver = _get_driver()
    with driver.session() as session:
        result = session.run(query, params or {})
        return [record.data() for record in result]


def format_answer(question: str, data: list[dict]) -> str:
    if not data:
        return "No relevant places found in the knowledge base."

    answer = f"Here are some results for '{question}':\n\n"
    for place in data:
        answer += f"🔹 {place.get('name')}\n"
        if place.get("description"):
            answer += f"   {place.get('description')}\n"
        if place.get("best_time"):
            answer += f"   Best time to visit: {place.get('best_time')}\n"
        if place.get("entry_fee"):
            answer += f"   Entry fee: {place.get('entry_fee')}\n"
        answer += "\n"
    return answer


def answer_question(question: str) -> str:
    intent = extract_entities(question)
    query, params = build_query(intent)
    data = run_query(query, params)
    return format_answer(question, data)


def fetch_kg(cypher_query: str) -> List[KgTriple]:
    """Execute the provided Cypher query and map results to `KgTriple`.

    Mapping strategy:
      - If records contain keys 'subject','predicate','object', use them.
      - Else, if each record has >=3 values, use the first three as subject/predicate/object.
      - Otherwise return an empty list.
    """
    try:
        driver = _get_driver()
    except Exception:
        logger.exception("Cannot obtain Neo4j driver; returning empty KG results")
        return []

    with driver.session() as session:
        try:
            result = session.run(cypher_query)
            triples: list[KgTriple] = []
            for rec in result:
                data = rec.data()
                if all(k in data for k in ("subject", "predicate", "object")):
                    triples.append(KgTriple(subject=str(data["subject"]), predicate=str(data["predicate"]), object=str(data["object"])))
                elif len(data) >= 3:
                    vals = list(data.values())
                    triples.append(KgTriple(subject=str(vals[0]), predicate=str(vals[1]), object=str(vals[2])))
                else:
                    # cannot map; skip
                    continue
            return triples
        except Exception:
            logger.exception("KG query failed")
            return []
