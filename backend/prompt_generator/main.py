"""Prompt generator component.

This module constructs well-structured prompts by combining the user query
with retrieval context from the vector database and knowledge graph.

The prompt format is designed for benchmarking and later swapping in a real LLM.
"""

from __future__ import annotations

import logging

from backend.kg.types import KgTriple

logger = logging.getLogger(__name__)

_PROMPT_CONTEXT_BUDGET = 1600


def get_prompt_context_budget() -> int:
    """Return the active context budget in characters."""
    return _PROMPT_CONTEXT_BUDGET


def set_prompt_context_budget(char_budget: int) -> None:
    """Set prompt context budget in characters."""
    global _PROMPT_CONTEXT_BUDGET
    _PROMPT_CONTEXT_BUDGET = max(300, int(char_budget))


def _trim_to_budget(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    return text[: max(0, budget - 3)] + "..."


def build_prompt(query: str, vector_snippet: str | None, kg_triples: list[KgTriple] | None) -> str:
    """Build a FLAN-style instruction prompt.

    Args:
        query: User question.
        vector_snippet: Retrieved result text from vector DB (or None).
        kg_triples: Retrieved KG triples (or None).

    Returns:
        A formatted instruction prompt string.
    """

    instruction = (
        "Answer the query using the supplied context below. "
        "Provide a detailed, natural-sounding response with specific information from the context. "
        "If context is insufficient, explicitly say that the answer is not available in the provided context. "
        "Be informative and include relevant details to make the answer comprehensive and helpful."
    )

    context_sections: list[str] = []

    if vector_snippet:
        context_sections.append(f"Vector context:\n{vector_snippet}")

    if kg_triples:
        formatted_triples = _format_kg_triples(kg_triples)
        context_sections.append(f"Knowledge graph facts:\n{formatted_triples}")

    context_blob = "\n\n".join(context_sections) if context_sections else "No retrieval context provided."
    context_blob = _trim_to_budget(context_blob, _PROMPT_CONTEXT_BUDGET)

    prompt = (
        f"Instruction:\n{instruction}\n\n"
        f"Query:\n{query}\n\n"
        f"Context:\n{context_blob}\n\n"
        "Answer:\n"
    )

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
