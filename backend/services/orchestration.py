"""
Rule-based orchestration for the query pipeline.

Given query and test_mode, runs Query Structurer (when needed), vector DB +
reranker, KG + KG reranker, builds a prompt via stub Prompt Generator, and calls
stub Test LLM. Returns a dict suitable for QueryResponse.
"""

from __future__ import annotations

import logging
from typing import Literal

from backend.dto import ContextUsed, QueryResponse
from backend.kg import fetch_kg
from backend.kg.types import KgTriple
from backend.kg_reranker import rerank_kg_triples
from backend.query_structurer import structure_query
from backend.reranker import rerank_top_cross_encoder
from backend.vector_db import fetch_top_vectordb

logger = logging.getLogger(__name__)

TestMode = Literal["vectordb", "kg", "hybrid", "none"]

# Retrieval limits used by the pipeline (configurable later if needed)
VECTOR_TOP_K = 10
RERANK_TOP_N = 1
KG_RERANK_TOP_N = 1


def _build_prompt_stub(query: str, vector_snippet: str | None, kg_triples: list) -> str:
    """
    Stub Prompt Generator: build a single prompt string from context and query.

    If no context, prompt is just the query (e.g. for test_mode=none).
    """
    if not vector_snippet and not kg_triples:
        return query

    parts = []
    if vector_snippet:
        parts.append(f"Context:\n{vector_snippet}")
    if kg_triples:
        triples_text = "\n".join(f"- {t.subject} --{t.predicate}--> {t.object}" for t in kg_triples)
        parts.append(f"Structured facts:\n{triples_text}")
    parts.append("Instruction: Answer only using the context above. If insufficient, say so.")
    parts.append(f"Query: {query}")
    return "\n\n".join(parts)


def _call_test_llm_stub(prompt: str) -> str:
    """
    Stub Test LLM: return a placeholder answer. Replace with real LLM client later.
    """
    return f"[Placeholder answer for prompt length={len(prompt)}]"


def run_pipeline(query: str, test_mode: TestMode) -> QueryResponse:
    """
    Run the full pipeline: route by test_mode, retrieve (vector/kg), build prompt, call Test LLM.

    Args:
        query: User natural language query.
        test_mode: One of vectordb, kg, hybrid, none.

    Returns:
        QueryResponse with answer, test_mode, and optional context_used.
    """
    logger.info("Running pipeline with test_mode=%s", test_mode)

    vector_snippet: str | None = None
    kg_triples: list = []

    if test_mode == "none":
        logger.debug("test_mode=none: skipping retrieval and query structurer")
    else:
        output_kind: Literal["vector_only", "kg_only", "both"] = {
            "vectordb": "vector_only",
            "kg": "kg_only",
            "hybrid": "both",
        }[test_mode]
        structured = structure_query(query, output_kind)

        # Vector path
        if structured.semantic_search_query:
            search_query = structured.semantic_search_query
            logger.debug("Vector path: fetching top_k=%d then reranking top_n=%d", VECTOR_TOP_K, RERANK_TOP_N)
            candidates, _ = fetch_top_vectordb(n=VECTOR_TOP_K, query=search_query, include_debug=False)
            if candidates:
                reranked, _ = rerank_top_cross_encoder(
                    query=search_query,
                    candidates=candidates,
                    top_n=RERANK_TOP_N,
                    include_debug=False,
                )
                if reranked:
                    vector_snippet = reranked[0].text
                    logger.info("Vector path produced snippet (len=%d)", len(vector_snippet or ""))
            else:
                logger.debug("Vector path returned no candidates")
        else:
            logger.debug("No semantic_search_query; skipping vector path")

        # KG path: fetch and rerank triples
        if structured.cypher_query:
            logger.info("Generated Cypher query: %s", structured.cypher_query)
            kg_triples = fetch_kg(structured.cypher_query)
            logger.debug("KG path: fetched %d triple(s), reranking top_n=%d", len(kg_triples), KG_RERANK_TOP_N)
            if kg_triples:
                # Use the semantic search query if available, else use the original query
                rerank_query = structured.semantic_search_query or query
                reranked, _ = rerank_kg_triples(
                    query=rerank_query,
                    triples=kg_triples,
                    top_n=KG_RERANK_TOP_N,
                    include_debug=False,
                )
                if reranked:
                    # Convert reranked KgTriples back to KgTriple format for prompt building
                    kg_triples = [
                        KgTriple(subject=r.subject, predicate=r.predicate, object=r.object)
                        for r in reranked
                    ]
                    logger.info("KG rerank produced %d triple(s)", len(kg_triples))
                else:
                    kg_triples = []
            else:
                logger.debug("KG path returned no triples")
        else:
            logger.debug("No cypher_query; skipping KG path")

    # Set kg_snippet from top triple
    kg_snippet: str | None = None
    if kg_triples:
        top_triple = kg_triples[0]
        kg_snippet = f"{top_triple.subject} --{top_triple.predicate}--> {top_triple.object}"

    prompt = _build_prompt_stub(query, vector_snippet, kg_triples)
    answer = _call_test_llm_stub(prompt)

    # Handle failure for kg mode
    if test_mode == "kg" and not kg_triples:
        answer = "No KG data found for the query."

    context_used = ContextUsed(
        vector_snippet=(vector_snippet[:200] + "...") if vector_snippet and len(vector_snippet) > 200 else vector_snippet,
        kg_triples_count=len(kg_triples) if kg_triples else None,
        kg_snippet=kg_snippet,
    )
    if not context_used.vector_snippet and context_used.kg_triples_count is None and not context_used.kg_snippet:
        context_used = None

    return QueryResponse(
        answer=answer,
        test_mode=test_mode,
        context_used=context_used,
    )
