"""
Main evaluation engine.

Orchestrates evaluation of the pipeline across all modes using test data.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from backend.services import run_pipeline

from backend.evaluation_engine.f1_score import compute_f1_score
from backend.evaluation_engine.hallucination_rate import compute_hallucination_rate
from backend.evaluation_engine.semantic_matching import compute_semantic_similarity

logger = logging.getLogger(__name__)


def _compute_improvement(current: float, baseline: float) -> float | None:
    """Compute improvement over baseline (none mode)."""
    if baseline == 0:
        return None
    return (current - baseline) / baseline


def evaluate_pipeline(test_data_path: str) -> dict[str, Any]:
    """
    Evaluate the pipeline using test data.

    Args:
        test_data_path: Path to test_data.json

    Returns:
        Dict with evaluation results.
    """
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)

    results = {"queries": []}

    modes = ["none", "vectordb", "kg", "hybrid"]
    MAX_ATTEMPTS_PER_MODE = 2

    for item in test_data:
        query = item["query"]
        expected_answer = item["expected_answer"]

        logger.info("Evaluating query: %s", query[:50] + "...")

        query_result = {
            "query": query,
            "expected_answer": expected_answer,
            "modes": {}
        }

        # Store metrics for none mode to compute improvements
        none_metrics = None

        for mode in modes:
            mode_result: dict[str, Any] = {}
            last_error: Exception | None = None

            for attempt in range(1, MAX_ATTEMPTS_PER_MODE + 1):
                try:
                    response = run_pipeline(query=query, test_mode=mode)
                    actual_answer = response.answer

                    # Compute metrics
                    semantic = compute_semantic_similarity(expected_answer, actual_answer)
                    f1 = compute_f1_score(expected_answer, actual_answer)
                    hallucination = compute_hallucination_rate(actual_answer)

                    mode_metrics = {
                        "semantic_score": semantic.score,
                        "f1_score": f1.f1,
                        "f1_precision": f1.precision,
                        "f1_recall": f1.recall,
                        "hallucination_rate": hallucination.rate,
                        "hallucination_total_claims": hallucination.total_claims,
                        "hallucination_verified_claims": hallucination.verified_claims,
                        "hallucination_unverified_claims": hallucination.unverified_claims,
                        "improvement_over_none": None,
                        "attempts": attempt,
                    }

                    if mode == "none":
                        none_metrics = mode_metrics.copy()
                        mode_metrics["improvement_over_none"] = None
                    else:
                        if none_metrics:
                            mode_metrics["improvement_over_none"] = {
                                "semantic": _compute_improvement(mode_metrics["semantic_score"], none_metrics["semantic_score"]),
                                "f1": _compute_improvement(mode_metrics["f1_score"], none_metrics["f1_score"]),
                                "hallucination": _compute_improvement(mode_metrics["hallucination_rate"], none_metrics["hallucination_rate"]),
                            }

                    mode_result = mode_metrics
                    last_error = None
                    break

                except Exception as e:
                    last_error = e
                    logger.exception(
                        "Attempt %s/%s failed for mode %s query %s",
                        attempt,
                        MAX_ATTEMPTS_PER_MODE,
                        mode,
                        query,
                    )

            if last_error is not None:
                mode_result = {
                    "error": str(last_error),
                    "attempts": MAX_ATTEMPTS_PER_MODE,
                }

            query_result["modes"][mode] = mode_result

        results["queries"].append(query_result)

    return results