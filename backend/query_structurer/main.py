"""
Public interface for the Query Structurer (LLM-powered) component.

Produces a StructuredQuery (semantic search string and/or Cypher query)
from a natural language query using the Gemini API. Used by the orchestration
layer to call vector_db and KG.
"""

from __future__ import annotations

import json
import ast
import logging
import re
from typing import Literal

import google.generativeai as genai
from typing import Any

from .config import QueryStructurerSettings, get_query_structurer_settings
from .types import StructuredQuery

logger = logging.getLogger(__name__)

OutputKind = Literal["vector_only", "kg_only", "both"]

_SYSTEM = """You are a query structurer for a Karnataka tourism (Mangalore, Udupi) system.
Given a user question, you output structured queries in JSON only, no other text.

The knowledge graph has nodes: Place (properties: name, type, description, best_time, entry_fee), City (name), District (name), State (name).
Relations: LOCATED_IN (Place->City), IN_DISTRICT (City->District), IN_STATE (District->State), HAS_ACTIVITY, NEARBY.

- For semantic search: output a short, clean "semantic_search_query" string suitable for vector similarity search (key phrases, no full sentences).
- For the knowledge graph: output a "cypher_query" string that is valid Neo4j Cypher. Use MATCH and return triples as subject, predicate, object. Example: MATCH (p:Place)-[r:LOCATED_IN]->(c:City) WHERE c.name = 'Mangalore' AND p.type = 'Beach' RETURN p.name AS subject, type(r) AS predicate, c.name AS object LIMIT 10.
Always respond with exactly one JSON object. No markdown, no explanation."""

_PROMPT_TEMPLATES = {
    "vector_only": "User question: {query}\n\nOutput JSON with key: semantic_search_query",
    "kg_only": "User question: {query}\n\nOutput JSON with key: cypher_query",
    "both": """User question: {query}

Output JSON with keys: semantic_search_query and cypher_query.

semantic_search_query: A short string for vector similarity search (e.g., "beaches in Mangalore").
cypher_query: A valid Neo4j Cypher query returning triples (e.g., MATCH (p:Place)-[r:LOCATED_IN]->(c:City) WHERE c.name = 'Mangalore' AND p.type = 'Beach' RETURN p.name AS subject, type(r) AS predicate, c.name AS object LIMIT 10).""",
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
        json_text = json_match.group()
        data = None
        # Primary: try strict JSON
        try:
            data = json.loads(json_text)
        except (json.JSONDecodeError, TypeError) as e_json:
            # Secondary: try Python literal_eval for single-quoted or non-strict dicts
            try:
                data = ast.literal_eval(json_text)
            except Exception as e_ast:
                # Tertiary: naive single-quote -> double-quote attempt (best-effort)
                try:
                    cleaned = json_text.replace("\'", '"')
                    data = json.loads(cleaned)
                except Exception:
                    logger.warning("Query structurer JSON parse failed: %s; literal_eval failed: %s", e_json, e_ast)

        if isinstance(data, dict):
            if output_kind in ("vector_only", "both") and "semantic_search_query" in data:
                v = data["semantic_search_query"]
                semantic = v if isinstance(v, str) and v.strip() else None
            if output_kind in ("kg_only", "both") and "cypher_query" in data:
                c = data["cypher_query"]
                cypher = c if isinstance(c, str) and c.strip() else None
        else:
            # Log a short snippet for debugging when parsing fails
            logger.debug("Unable to parse structured JSON from LLM response (snippet=%r)", (text or '')[:400])

            # Fallback heuristics: try to extract values with regex from non-JSON text
            try:
                if output_kind in ("vector_only", "both"):
                    m = re.search(r"semantic_search_query\s*[:=]\s*[\"']([^\"']+)[\"']", text, re.IGNORECASE)
                    if m:
                        semantic = m.group(1).strip()
                if output_kind in ("kg_only", "both"):
                    m2 = re.search(r"cypher_query\s*[:=]\s*[\"']([^\"']+)[\"']", text, re.IGNORECASE)
                    if m2:
                        cypher = m2.group(1).strip()
            except Exception:
                logger.debug("Fallback regex extraction also failed", exc_info=True)

    return StructuredQuery(
        semantic_search_query=semantic,
        cypher_query=cypher,
    )


def _extract_from_proto_like(obj: Any) -> str | None:
    """Attempt to extract the most likely text content from various
    proto/dict/list shapes returned by different `google.generativeai`
    client versions.

    Tries a number of common attribute/index paths and returns the first
    string found.
    """
    def _get(cur, key):
        # key may be attr name or int index
        try:
            if isinstance(key, int):
                if isinstance(cur, (list, tuple)):
                    return cur[key]
                # try dict-like access
                if isinstance(cur, dict):
                    return list(cur.values())[key]
                return None
            # str key
            if isinstance(cur, dict) and key in cur:
                return cur[key]
            if hasattr(cur, key):
                return getattr(cur, key)
            # dict-like fallback
            if isinstance(cur, dict):
                return cur.get(key)
        except Exception:
            return None
        return None

    paths = [
        ("result", "candidates", 0, "content", "parts", 0, "text"),
        ("candidates", 0, "content", "parts", 0, "text"),
        ("candidates", 0, "content", "text"),
        ("candidates", 0, "output", "content", "parts", 0, "text"),
        ("outputs", 0, "content", 0, "text"),
        ("choices", 0, "message", "content"),
        ("candidates", 0, "text"),
    ]

    for path in paths:
        cur = obj
        ok = True
        for seg in path:
            cur = _get(cur, seg)
            if cur is None:
                ok = False
                break
        if ok and isinstance(cur, str) and cur.strip():
            return cur.strip()

    # Last resort: string-convert the object and try to regex-extract the inner JSON-like
    try:
        s = str(obj)
        m = re.search(r"\{[^{}]*\}", s, re.DOTALL)
        if m:
            return m.group()
    except Exception:
        pass

    return None


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
    logger.info("Query Structurer using Gemini model=%s", settings.model_name)
    prompt = _PROMPT_TEMPLATES[output_kind].format(query=query)

    def _extract_text(obj: Any) -> str | None:
        """
        Heuristic extractor: walk nested response objects (dicts/lists)
        to find the first plausible text field.
        """
        if obj is None:
            return None
        if isinstance(obj, str):
            return obj.strip() or None
        if isinstance(obj, dict):
            # common keys used by different genai shapes
            for k in ("text", "content", "message", "output", "candidates", "response", "result"):
                if k in obj:
                    val = _extract_text(obj[k])
                    if val:
                        return val
            # fallback to first string value
            for v in obj.values():
                val = _extract_text(v)
                if val:
                    return val
            return None
        if isinstance(obj, (list, tuple)):
            for item in obj:
                val = _extract_text(item)
                if val:
                    return val
            return None
        # any other object: try string conversion
        try:
            s = str(obj)
            return s.strip() or None
        except Exception:
            return None

    def _call_gemini(settings: "QueryStructurerSettings", prompt_text: str) -> str:
        """Call the google.generativeai client in a resilient way and return raw text."""
        genai.configure(api_key=settings.api_key)

        # Try high-level generate() API
        try:
            if hasattr(genai, "generate"):
                resp = genai.generate(model=settings.model_name, input=prompt_text, temperature=0.0)
                logger.debug("Gemini raw response (generate): %s", repr(resp)[:2000])
                # Try proto-aware extractor first
                inner = _extract_from_proto_like(resp)
                if inner:
                    # normalize escaped JSON if present
                    try:
                        if inner.startswith('"') and inner.endswith('"'):
                            inner = json.loads(inner)
                    except Exception:
                        pass
                    # unescape common backslash-escaped JSON
                    try:
                        if '\\' in inner:
                            cand = inner.encode('utf-8').decode('unicode_escape')
                            if cand.strip().startswith('{'):
                                inner = cand
                    except Exception:
                        pass
                    logger.debug("Extracted inner text from proto-like resp (truncated): %s", inner[:1000])
                    return inner
                # fallback to generic extractor
                text = _extract_text(resp)
                if text:
                    return text
        except Exception:
            logger.debug("genai.generate() attempt failed; falling back", exc_info=True)

        # Try chat.create() shape
        try:
            if hasattr(genai, "chat") and hasattr(genai.chat, "create"):
                messages = [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt_text},
                ]
                resp = genai.chat.create(model=settings.model_name, messages=messages, temperature=0.0)
                logger.debug("Gemini raw response (chat.create): %s", repr(resp)[:2000])
                inner = _extract_from_proto_like(resp)
                if inner:
                    try:
                        if inner.startswith('"') and inner.endswith('"'):
                            inner = json.loads(inner)
                    except Exception:
                        pass
                    try:
                        if '\\' in inner:
                            cand = inner.encode('utf-8').decode('unicode_escape')
                            if cand.strip().startswith('{'):
                                inner = cand
                    except Exception:
                        pass
                    logger.debug("Extracted inner text from proto-like chat resp (truncated): %s", inner[:1000])
                    return inner
                text = _extract_text(resp)
                if text:
                    return text
        except Exception:
            logger.debug("genai.chat.create() attempt failed; falling back", exc_info=True)

        # Try the GenerativeModel wrapper (older/newer shape compatibility)
        try:
            model = genai.GenerativeModel(settings.model_name, system_instruction=_SYSTEM)
            response = model.generate_content(prompt_text, generation_config={"temperature": 0.0})
            logger.debug("Gemini raw response (GenerativeModel.generate_content): %s", repr(response)[:2000])
            inner = _extract_from_proto_like(response)
            if inner:
                try:
                    if inner.startswith('"') and inner.endswith('"'):
                        inner = json.loads(inner)
                except Exception:
                    pass
                try:
                    if '\\' in inner:
                        cand = inner.encode('utf-8').decode('unicode_escape')
                        if cand.strip().startswith('{'):
                            inner = cand
                except Exception:
                    pass
                logger.debug("Extracted inner text from GenerativeModel resp (truncated): %s", inner[:1000])
                return inner
            # response may expose .text or nested structure
            text = _extract_text(response)
            if text:
                return text
        except Exception:
            logger.exception("Query structurer Gemini call failed (all attempts)")

        # As a last-resort fallback return empty string so callers can handle it
        return ""

    try:
        content = _call_gemini(settings, prompt) or ""
        content = content.strip()
        # Always log the raw content at INFO so callers (and CLI tests) can see
        # exactly what the LLM returned and debug parsing issues.
        logger.info("Raw Gemini content (truncated 2000 chars): %s", content[:2000])
    except Exception:
        logger.exception("Query structurer failed invoking Gemini API")
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
