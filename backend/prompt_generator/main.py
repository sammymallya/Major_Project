"""Prompt generator component.

This module constructs well-structured prompts by combining the user query
with retrieval context from the vector database and knowledge graph.

The prompt format is designed for benchmarking and later swapping in a real LLM.
"""

from __future__ import annotations

import logging
from typing import List

from backend.kg.types import KgTriple

logger = logging.getLogger(__name__)


def build_prompt(query: str, vector_snippet: str | None, kg_triples: list[KgTriple] | None) -> str:
    """Build a chat-style prompt for TinyLlama.

    Args:
        query: User question.
        vector_snippet: Retrieved result text from vector DB (or None).
        kg_triples: Retrieved KG triples (or None).

    Returns:
        A formatted prompt string in chat format.
    """

    system_message = (
        "You are a helpful assistant that answers questions using the provided context. "
        "Use the retrieved context and facts to answer accurately and concisely. "
        "If the provided information is insufficient, say so clearly. "
        "Base your answer only on the given context and structured facts."
    )

    user_content = f"Query: {query}\n\n"

    if vector_snippet:
        user_content += f"Context from vector search:\n{vector_snippet}\n\n"

    if kg_triples:
        formatted_triples = _format_kg_triples(kg_triples)
        user_content += f"Structured facts from knowledge graph:\n{formatted_triples}\n\n"

    user_content += "Please provide a direct answer to the query based on the above information."

    # TinyLlama chat format
    prompt = f"<|system|>\n{system_message}\n<|user|>\n{user_content}\n<|assistant|>\n"

    logger.debug("Built prompt of length %d", len(prompt))
    return prompt


def _format_kg_triples(kg_triples: list[KgTriple] | list[dict] | list[tuple]) -> str:
    """Format KG triples into a readable list."""

    def _extract(triple) -> tuple[str, str, str] | None:
        if hasattr(triple, "subject") and hasattr(triple, "predicate") and hasattr(triple, "object"):
            return (str(triple.subject), str(triple.predicate), str(triple.object))
        if isinstance(triple, dict):
            try:
                return (str(triple["subject"]), str(triple["predicate"]), str(triple["object"]))
            except KeyError:
                return None
        if isinstance(triple, (list, tuple)) and len(triple) >= 3:
            return (str(triple[0]), str(triple[1]), str(triple[2]))
        return None

    lines: list[str] = []
    for i, t in enumerate(kg_triples, start=1):
        extracted = _extract(t)
        if not extracted:
            continue
        subject, predicate, obj = extracted
        lines.append(f"{i}. {subject} --{predicate}--> {obj}")
    return "\n".join(lines)
