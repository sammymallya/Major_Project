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
Given a user question, output VALID JSON and NOTHING ELSE.

Requirements:
- Output ONLY a single valid JSON object, no markdown, no explanation, no extra text.
- Use double quotes for all strings (mandatory for JSON).
- Escape internal quotes with backslash.
- Do NOT output Python expressions or pseudo-code.

The knowledge graph has nodes: Place (properties: name, category, city, state, tags), City (name, state), State (name).
Relations: LOCATED_IN (Place->City).

Fields:
- "semantic_search_query": A short, clean string for vector similarity search (key phrases, no full sentences).
- "cypher_query": A valid Neo4j Cypher query returning triples as subject, predicate, object.
  Example: MATCH (p:Place) WHERE p.city = 'Mangalore' AND p.category = 'Beach' RETURN p.name AS subject, 'has_category' AS predicate, p.category AS object LIMIT 10

Example output (valid JSON):
{"semantic_search_query": "beaches in Mangalore", "cypher_query": "MATCH (p:Place) WHERE p.category = 'Beach' RETURN p.name AS subject, 'category' AS predicate, p.category AS object"}"""

_PROMPT_TEMPLATES = {
    "vector_only": "User question: {query}\n\nOutput JSON with key: semantic_search_query",
    "kg_only": "User question: {query}\n\nOutput JSON with key: cypher_query",
    "both": """User question: {query}

Output JSON with keys: semantic_search_query and cypher_query.

semantic_search_query: A short string for vector similarity search (e.g., "activities in Udupi").
cypher_query: A valid Neo4j Cypher query returning triples (e.g., MATCH (p:Place) WHERE p.city = 'Mangalore' AND p.category = 'Beach' RETURN p.name AS subject, 'has_category' AS predicate, p.category AS object LIMIT 10).""",
}


def _find_balanced_json(text: str, start_pos: int = 0) -> str | None:
    """
    Find and extract a balanced JSON object from text starting at start_pos.
    Handles nested braces by tracking opening/closing brackets.
    """
    i = start_pos
    while i < len(text) and text[i] != '{':
        i += 1
    
    if i >= len(text):
        return None
    
    depth = 0
    in_string = False
    escape = False
    start = i
    
    while i < len(text):
        char = text[i]
        
        if escape:
            escape = False
            i += 1
            continue
        
        if char == '\\':
            escape = True
            i += 1
            continue
        
        if char == '"' and not escape:
            in_string = not in_string
        elif not in_string:
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        
        i += 1
    
    return None


def _parse_llm_response(text: str, output_kind: OutputKind) -> StructuredQuery:
    """
    Parse LLM response into StructuredQuery. On failure, return safe fallbacks.
    """
    semantic: str | None = None
    cypher: str | None = None

    # Try to extract balanced JSON from the response
    json_text = _find_balanced_json(text)
    
    if json_text:
        data = None
        # Primary: try strict JSON
        try:
            data = json.loads(json_text)
        except (json.JSONDecodeError, TypeError) as e_json:
            # Secondary: try cleaning common issues (single quotes, escaped quotes)
            try:
                # Replace single quotes with double quotes (careful: only for dict keys/values)
                cleaned = json_text.replace("'", '"')
                data = json.loads(cleaned)
            except Exception as e1:
                # Tertiary: unescape and try
                try:
                    # Try to handle backslash-escaped content
                    if '\\' in json_text:
                        # Don't decode unicode_escape as it can cause issues
                        # Instead, try to fix common escaping issues
                        cleaned = json_text.replace('\\"', '"').replace("\\'", "'")
                        data = json.loads(cleaned)
                except Exception as e2:
                    logger.debug("Query structurer JSON parse attempts failed: strict=%s, cleaned=%s", 
                                str(e_json)[:100], str(e1)[:100])

        if isinstance(data, dict):
            if output_kind in ("vector_only", "both") and "semantic_search_query" in data:
                v = data["semantic_search_query"]
                semantic = v if isinstance(v, str) and v.strip() else None
            if output_kind in ("kg_only", "both") and "cypher_query" in data:
                c = data["cypher_query"]
                cypher = c if isinstance(c, str) and c.strip() else None

    # If JSON parsing failed, use regex fallback heuristics
    if semantic is None or cypher is None:
        logger.debug("JSON parsing failed or incomplete, trying regex fallbacks")
        try:
            if output_kind in ("vector_only", "both") and semantic is None:
                # Try multiple patterns
                patterns = [
                    r"semantic_search_query\s*:\s*[\"']([^\"']*)[\"']",
                    r"semantic_search_query\s*=\s*[\"']([^\"']*)[\"']",
                    r"[\"']semantic_search_query[\"']\s*:\s*[\"']([^\"']+)[\"']",
                ]
                for pattern in patterns:
                    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                    if m:
                        semantic = m.group(1).strip()
                        if semantic:
                            break
            
            if output_kind in ("kg_only", "both") and cypher is None:
                # Try multiple patterns for Cypher (which can be longer)
                patterns = [
                    r"cypher_query\s*:\s*[\"']([^\"']+)[\"']",
                    r"cypher_query\s*=\s*[\"']([^\"']+)[\"']",
                    r"[\"']cypher_query[\"']\s*:\s*[\"']([^\"']+)[\"']",
                    r"MATCH\s+\([^)]*\)[^}]*",  # Try to find MATCH...
                ]
                for pattern in patterns:
                    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                    if m:
                        cypher = m.group(1).strip() if m.lastindex else m.group(0).strip()
                        if cypher:
                            break
        except Exception:
            logger.debug("Regex fallback extraction failed", exc_info=True)

    return StructuredQuery(
        semantic_search_query=semantic,
        cypher_query=cypher,
    )


def _generate_fallback_cypher(query: str) -> str:
    """
    Generate a fallback Cypher query when LLM structuring fails.
    Attempts basic place searches based on common keywords.
    """
    q_lower = query.lower()
    
    # Check for common patterns
    if any(word in q_lower for word in ["beach", "beaches"]):
        return "MATCH (p:Place {category: 'Beach'}) RETURN p.name AS subject, 'category' AS predicate, p.category AS object LIMIT 10"
    elif any(word in q_lower for word in ["temple", "temples", "church"]):
        return "MATCH (p:Place) WHERE p.category IN ['Temple', 'Church'] RETURN p.name AS subject, 'category' AS predicate, p.category AS object LIMIT 10"
    elif any(word in q_lower for word in ["restaurant", "food", "eat"]):
        return "MATCH (p:Place {category: 'Restaurant'}) RETURN p.name AS subject, 'category' AS predicate, p.category AS object LIMIT 10"
    elif any(word in q_lower for word in ["mangalore", "udupi"]):
        city = "Mangalore" if "mangalore" in q_lower else "Udupi"
        return f"MATCH (p:Place) WHERE p.city = '{city}' RETURN p.name AS subject, 'city' AS predicate, p.city AS object LIMIT 10"
    else:
        # Generic fallback: return all places
        return "MATCH (p:Place) RETURN p.name AS subject, 'type' AS predicate, p.category AS object LIMIT 10"


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

    # Limit Gemini API calls per query structuring invocation (avoids runaway retries).
    call_counter = {"count": 0}
    max_calls = 2

    try:
        content = _call_gemini(settings, prompt, call_counter=call_counter, max_calls=max_calls) or ""
        content = content.strip()
        # Always log the raw content at INFO so callers (and CLI tests) can see
        # exactly what the LLM returned and debug parsing issues.
        logger.info("Raw Gemini content (truncated 2000 chars): %s", content[:2000])
    except Exception:
        logger.exception("Query structurer failed invoking Gemini API")
        # Log the fallback being used
        fallback_semantic = query if output_kind in ("vector_only", "both") else None
        logger.warning("Query Structurer exception - using fallback: semantic=%s, cypher=None", fallback_semantic)
        return StructuredQuery(
            semantic_search_query=fallback_semantic,
            cypher_query=None,
        )

    result = _parse_llm_response(content, output_kind)

    # If semantic was requested but parsing left it empty, use original query
    if output_kind in ("vector_only", "both") and result.semantic_search_query is None:
        result = StructuredQuery(semantic_search_query=query, cypher_query=result.cypher_query)
        logger.info("Used fallback semantic_search_query (original query)")
    
    # If Cypher was requested but parsing left it empty, generate a fallback
    if output_kind in ("kg_only", "both") and result.cypher_query is None:
        fallback_cypher = _generate_fallback_cypher(query)
        result = StructuredQuery(semantic_search_query=result.semantic_search_query, cypher_query=fallback_cypher)
        logger.info("Used fallback Cypher query due to LLM parsing failure")

    def _trunc(s: str | None, max_len: int = 60) -> str:
        if not s:
            return str(s)
        return s[:max_len] + "..." if len(s) > max_len else s

    logger.info("Final Query Structurer result: semantic=%s cypher=%s", _trunc(result.semantic_search_query), _trunc(result.cypher_query))
    return result


def _call_gemini(
    settings: "QueryStructurerSettings",
    prompt_text: str,
    call_counter: dict[str, int],
    max_calls: int = 2,
) -> str:
    """Call the google.generativeai client in a resilient way and return raw text.

    Ensures we never call Gemini more than `max_calls` times per invocation.
    """
    if call_counter["count"] >= max_calls:
        raise RuntimeError(
            f"Gemini API rate limit exceeded: attempted {call_counter['count']} calls (max {max_calls})"
        )

    genai.configure(api_key=settings.api_key)

    # Helper to mark a call attempt
    def _mark_call() -> None:
        call_counter["count"] += 1
        if call_counter["count"] > max_calls:
            raise RuntimeError(
                f"Gemini API rate limit exceeded: attempted {call_counter['count']} calls (max {max_calls})"
            )

    # Try high-level generate() API
    try:
        if hasattr(genai, "generate"):
            _mark_call()
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
            _mark_call()
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
        _mark_call()
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
