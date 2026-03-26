"""Evaluation metrics used by the benchmark engine."""

from __future__ import annotations

import logging
import re
from typing import Any

from sentence_transformers import SentenceTransformer, util  # type: ignore[import]

logger = logging.getLogger(__name__)

_SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_semantic_model: SentenceTransformer | None = None


def _get_semantic_model() -> SentenceTransformer:
    global _semantic_model
    if _semantic_model is None:
        logger.info("Loading semantic matching model '%s'", _SEMANTIC_MODEL_NAME)
        _semantic_model = SentenceTransformer(_SEMANTIC_MODEL_NAME)
    return _semantic_model


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def compute_f1(expected_answer: str, actual_answer: str) -> dict[str, float]:
    """Compute token-level precision/recall/F1."""
    expected_tokens = _tokenize(expected_answer)
    actual_tokens = _tokenize(actual_answer)

    if not expected_tokens and not actual_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not expected_tokens or not actual_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    expected_counts: dict[str, int] = {}
    for token in expected_tokens:
        expected_counts[token] = expected_counts.get(token, 0) + 1

    overlap = 0
    for token in actual_tokens:
        if expected_counts.get(token, 0) > 0:
            overlap += 1
            expected_counts[token] -= 1

    precision = overlap / len(actual_tokens)
    recall = overlap / len(expected_tokens)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_semantic_similarity(expected_answer: str, actual_answer: str) -> float:
    """Compute cosine similarity between expected and actual answers."""
    model = _get_semantic_model()
    embeddings = model.encode([expected_answer, actual_answer], convert_to_tensor=True)
    return float(util.cos_sim(embeddings[0], embeddings[1]).item())


def extract_claims(answer: str) -> list[str]:
    """Split answer into sentence-level claims."""
    claims = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [claim.strip() for claim in claims if claim.strip()]


def compute_hallucination_metrics(
    answer: str,
    retrieved_context: list[str],
    similarity_threshold: float = 0.45,
) -> dict[str, Any]:
    """Compute hallucination and faithfulness from answer claims and retrieved context."""
    claims = extract_claims(answer)
    total_claims = len(claims)

    if total_claims == 0:
        return {
            "hallucination_rate": 0.0,
            "faithfulness": 1.0,
            "total_claims": 0,
            "grounded_claims": 0,
            "ungrounded_claims": 0,
        }

    if not retrieved_context:
        return {
            "hallucination_rate": 1.0,
            "faithfulness": 0.0,
            "total_claims": total_claims,
            "grounded_claims": 0,
            "ungrounded_claims": total_claims,
        }

    model = _get_semantic_model()
    claim_embeddings = model.encode(claims, convert_to_tensor=True)
    context_embeddings = model.encode(retrieved_context, convert_to_tensor=True)
    sims = util.cos_sim(claim_embeddings, context_embeddings)

    grounded = 0
    for row in sims:
        if float(row.max().item()) >= similarity_threshold:
            grounded += 1

    ungrounded = total_claims - grounded
    hallucination_rate = ungrounded / total_claims

    return {
        "hallucination_rate": hallucination_rate,
        "faithfulness": 1.0 - hallucination_rate,
        "total_claims": total_claims,
        "grounded_claims": grounded,
        "ungrounded_claims": ungrounded,
    }
