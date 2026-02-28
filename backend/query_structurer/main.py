"""
Public interface for the Query Structurer (LLM-powered) component.

Produces a StructuredQuery (semantic search string and/or Cypher query)
from a natural language query using the Gemini API. Used by the orchestration
layer to call vector_db and KG.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

import google.generativeai as genai

from .config import QueryStructurerSettings, get_query_structurer_settings
from .types import StructuredQuery

logger = logging.getLogger(__name__)

OutputKind = Literal["vector_only", "kg_only", "both"]

_SYSTEM = """You are a query structurer for a Karnataka tourism (Mangalore, Udupi) system.
Given a user question, you output structured queries in JSON only, no other text.

- For semantic search: output a short, clean "semantic_search_query" string suitable for vector similarity search (key phrases, no full sentences).
- For the knowledge graph: output a "cypher_query" string that is valid Neo4j Cypher. The graph has Place, City, District, State nodes; relations like LOCATED_IN, IN_DISTRICT, IN_STATE, HAS_ACTIVITY, NEARBY. Use MERGE/MATCH and return relevant facts (e.g. MATCH (p:Place)-[:LOCATED_IN]->(c:City) RETURN p.name, c.name LIMIT 10).
Always respond with exactly one JSON object. No markdown, no explanation."""

_PROMPT_TEMPLATES = {
    "vector_only": "User question: {query}\n\nOutput JSON with key: semantic_search_query",
    "kg_only": "User question: {query}\n\nOutput JSON with key: cypher_query",
    "both": "User question: {query}\n\nOutput JSON with keys: semantic_search_query and cypher_query",
}


def _parse_llm_response(text: str, output_kind: OutputKind) -> StructuredQuery:
    """
    Parse LLM response into StructuredQuery. On failure, return safe fallbacks.
    """
    semantic: str | None = None
    cypher: str | None = None

    # Try to extract JSON from the response (in case of markdown or extra text)
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if output_kind in ("vector_only", "both") and "semantic_search_query" in data:
                v = data["semantic_search_query"]
                semantic = v if isinstance(v, str) and v.strip() else None
            if output_kind in ("kg_only", "both") and "cypher_query" in data:
                c = data["cypher_query"]
                cypher = c if isinstance(c, str) and c.strip() else None
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Query structurer JSON parse failed: %s", e)

    return StructuredQuery(
        semantic_search_query=semantic,
        cypher_query=cypher,
    )


def structure_query(query: str, output_kind: OutputKind) -> StructuredQuery:
    """
    Use the configured Gemini model to produce a structured query for vector DB and/or KG.

    Args:
        query: Natural language user question.
        output_kind: vector_only (semantic only), kg_only (cypher only), or both.

    Returns:
        StructuredQuery with semantic_search_query and/or cypher_query set.
        On LLM or parse failure, returns fallbacks: semantic_search_query=query
        when vector requested, cypher_query=None.
    """
    settings = get_query_structurer_settings()
    prompt = _PROMPT_TEMPLATES[output_kind].format(query=query)

    genai.configure(api_key=settings.api_key)
    model = genai.GenerativeModel(
        settings.model_name,
        system_instruction=_SYSTEM,
    )

    try:
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0},
        )
        content = (response.text or "").strip()
    except Exception as e:
        logger.error("Query structurer Gemini call failed: %s", e, exc_info=True)
        return StructuredQuery(
            semantic_search_query=query if output_kind in ("vector_only", "both") else None,
            cypher_query=None,
        )

    result = _parse_llm_response(content, output_kind)

    # If semantic was requested but parsing left it empty, use original query
    if output_kind in ("vector_only", "both") and result.semantic_search_query is None:
        result = StructuredQuery(semantic_search_query=query, cypher_query=result.cypher_query)

    def _trunc(s: str | None, max_len: int = 60) -> str:
        if not s:
            return str(s)
        return s[:max_len] + "..." if len(s) > max_len else s

    logger.debug("Structured query: semantic=%s cypher=%s", _trunc(result.semantic_search_query), _trunc(result.cypher_query))
    return result
