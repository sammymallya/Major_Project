"""Evaluation engine for multi-mode benchmark runs."""

from __future__ import annotations

import json
import logging
from typing import Any

from backend.prompt_generator import get_prompt_context_budget, set_prompt_context_budget
from backend.services import get_retrieval_limits, run_pipeline_for_evaluation, set_retrieval_limits

from .metrics import compute_f1, compute_hallucination_metrics, compute_semantic_similarity

logger = logging.getLogger(__name__)

MODES = ["none", "vectordb", "kg", "hybrid"]


def _safe_percent_improvement(current: float, baseline: float, higher_is_better: bool = True) -> float | None:
    """Compute percentage improvement and handle zero baselines safely."""
    if abs(baseline) < 1e-12:
        return None
    if higher_is_better:
        return ((current - baseline) / abs(baseline)) * 100.0
    return ((baseline - current) / abs(baseline)) * 100.0


def _load_test_data(test_data_path: str) -> list[dict[str, str]]:
    with open(test_data_path, "r") as f:
        raw_data = json.load(f)

    normalized: list[dict[str, str]] = []
    for item in raw_data:
        query = item["query"]
        # New contract uses `answer`. Keep fallback for old fixtures.
        ground_truth = item.get("answer") or item.get("expected_answer")
        if not ground_truth:
            raise ValueError("Each test item must include 'answer'.")
        normalized.append({"query": query, "answer": ground_truth})
    return normalized


def _make_run_cache_key(query: str, mode: str) -> tuple[str, str, int, int, int, int]:
    limits = get_retrieval_limits()
    return (
        query,
        mode,
        limits["vector_top_k"],
        limits["rerank_top_n"],
        limits["kg_rerank_top_n"],
        get_prompt_context_budget(),
    )


def _evaluate_mode(
    query: str,
    ground_truth: str,
    mode: str,
    run_cache: dict[tuple[str, str, int, int, int, int], Any],
) -> dict[str, Any]:
    cache_key = _make_run_cache_key(query, mode)
    if cache_key not in run_cache:
        run_cache[cache_key] = run_pipeline_for_evaluation(query=query, test_mode=mode)  # type: ignore[arg-type]

    pipeline_result = run_cache[cache_key]
    answer = pipeline_result.answer
    retrieved_context = pipeline_result.retrieved_context

    semantic_similarity = compute_semantic_similarity(ground_truth, answer)
    f1_metrics = compute_f1(ground_truth, answer)
    hallucination_metrics = compute_hallucination_metrics(answer, retrieved_context)

    return {
        "answer": answer,
        "retrieved_context": retrieved_context,
        "retrieved_context_count": len(retrieved_context),
        "semantic_similarity": semantic_similarity,
        "f1_score": f1_metrics["f1"],
        "precision": f1_metrics["precision"],
        "recall": f1_metrics["recall"],
        "hallucination_rate": hallucination_metrics["hallucination_rate"],
        "faithfulness": hallucination_metrics["faithfulness"],
        "total_claims": hallucination_metrics["total_claims"],
        "grounded_claims": hallucination_metrics["grounded_claims"],
        "ungrounded_claims": hallucination_metrics["ungrounded_claims"],
        "improvement_over_none_percent": None,
    }


def _aggregate_mode_records(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {}

    numeric_keys = [
        "semantic_similarity",
        "f1_score",
        "precision",
        "recall",
        "hallucination_rate",
        "faithfulness",
        "total_claims",
        "grounded_claims",
        "ungrounded_claims",
    ]

    summary: dict[str, float] = {}
    for key in numeric_keys:
        summary[f"avg_{key}"] = sum(float(item[key]) for item in records) / len(records)
    summary["count"] = float(len(records))
    return summary


def _score_config(records: list[dict[str, Any]]) -> float:
    if not records:
        return float("-inf")

    hybrid_records = []
    for record in records:
        hybrid = record["modes"].get("hybrid")
        if hybrid and "error" not in hybrid:
            hybrid_records.append(hybrid)

    if not hybrid_records:
        return float("-inf")

    semantic_avg = sum(float(r["semantic_similarity"]) for r in hybrid_records) / len(hybrid_records)
    f1_avg = sum(float(r["f1_score"]) for r in hybrid_records) / len(hybrid_records)
    faithfulness_avg = sum(float(r["faithfulness"]) for r in hybrid_records) / len(hybrid_records)
    return semantic_avg + f1_avg + faithfulness_avg


def _run_once(dataset: list[dict[str, str]], run_cache: dict[tuple[str, str, int, int, int, int], Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for item in dataset:
        query = item["query"]
        ground_truth = item["answer"]

        query_record = {
            "query": query,
            "ground_truth": ground_truth,
            "modes": {},
        }

        none_metrics: dict[str, Any] | None = None

        for mode in MODES:
            try:
                metrics = _evaluate_mode(
                    query=query,
                    ground_truth=ground_truth,
                    mode=mode,
                    run_cache=run_cache,
                )
                query_record["modes"][mode] = metrics
                if mode == "none":
                    none_metrics = metrics
            except Exception as exc:  # pragma: no cover - defensive for runtime services
                logger.exception("Evaluation failed for mode=%s query=%s", mode, query)
                query_record["modes"][mode] = {"error": str(exc)}

        if none_metrics:
            for mode in ["vectordb", "kg", "hybrid"]:
                mode_metrics = query_record["modes"].get(mode)
                if not mode_metrics or "error" in mode_metrics:
                    continue
                mode_metrics["improvement_over_none_percent"] = {
                    "semantic_similarity": _safe_percent_improvement(
                        mode_metrics["semantic_similarity"],
                        none_metrics["semantic_similarity"],
                        higher_is_better=True,
                    ),
                    "f1_score": _safe_percent_improvement(
                        mode_metrics["f1_score"],
                        none_metrics["f1_score"],
                        higher_is_better=True,
                    ),
                    "hallucination_rate": _safe_percent_improvement(
                        mode_metrics["hallucination_rate"],
                        none_metrics["hallucination_rate"],
                        higher_is_better=False,
                    ),
                    "faithfulness": _safe_percent_improvement(
                        mode_metrics["faithfulness"],
                        none_metrics["faithfulness"],
                        higher_is_better=True,
                    ),
                }

        records.append(query_record)

    return records


def _build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    mode_buckets: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for record in records:
        for mode in MODES:
            mode_result = record["modes"].get(mode)
            if mode_result and "error" not in mode_result:
                mode_buckets[mode].append(mode_result)

    summary = {mode: _aggregate_mode_records(mode_buckets[mode]) for mode in MODES}

    none_summary = summary.get("none") or {}
    none_semantic = float(none_summary.get("avg_semantic_similarity", 0.0))
    none_f1 = float(none_summary.get("avg_f1_score", 0.0))
    none_hallucination = float(none_summary.get("avg_hallucination_rate", 0.0))
    none_faithfulness = float(none_summary.get("avg_faithfulness", 0.0))

    for mode in ["vectordb", "kg", "hybrid"]:
        mode_summary = summary.get(mode) or {}
        if not mode_summary:
            continue
        mode_summary["improvement_over_none_percent"] = {
            "semantic_similarity": _safe_percent_improvement(
                float(mode_summary.get("avg_semantic_similarity", 0.0)),
                none_semantic,
                higher_is_better=True,
            ),
            "f1_score": _safe_percent_improvement(
                float(mode_summary.get("avg_f1_score", 0.0)),
                none_f1,
                higher_is_better=True,
            ),
            "hallucination_rate": _safe_percent_improvement(
                float(mode_summary.get("avg_hallucination_rate", 0.0)),
                none_hallucination,
                higher_is_better=False,
            ),
            "faithfulness": _safe_percent_improvement(
                float(mode_summary.get("avg_faithfulness", 0.0)),
                none_faithfulness,
                higher_is_better=True,
            ),
        }

    return summary


def _calibrate(dataset: list[dict[str, str]]) -> dict[str, int]:
    """Run a small bounded sweep for quality-first retrieval settings."""
    candidate_configs = [
        {
            "vector_top_k": 8,
            "rerank_top_n": 1,
            "kg_rerank_top_n": 1,
            "context_budget": 1400,
        },
        {
            "vector_top_k": 12,
            "rerank_top_n": 2,
            "kg_rerank_top_n": 2,
            "context_budget": 1800,
        },
        {
            "vector_top_k": 16,
            "rerank_top_n": 3,
            "kg_rerank_top_n": 3,
            "context_budget": 2200,
        },
    ]

    if not dataset:
        return {
            "vector_top_k": 10,
            "rerank_top_n": 1,
            "kg_rerank_top_n": 1,
            "context_budget": 1600,
        }

    calibration_set = dataset[: min(3, len(dataset))]
    best_config = candidate_configs[0]
    best_score = float("-inf")
    sweep_cache: dict[tuple[str, str, int, int, int, int], Any] = {}

    previous_limits = get_retrieval_limits()
    previous_context_budget = get_prompt_context_budget()

    try:
        for config in candidate_configs:
            set_retrieval_limits(
                vector_top_k=config["vector_top_k"],
                rerank_top_n=config["rerank_top_n"],
                kg_rerank_top_n=config["kg_rerank_top_n"],
            )
            set_prompt_context_budget(config["context_budget"])
            records = _run_once(calibration_set, run_cache=sweep_cache)
            score = _score_config(records)
            if score > best_score:
                best_score = score
                best_config = config
    finally:
        set_retrieval_limits(
            vector_top_k=previous_limits["vector_top_k"],
            rerank_top_n=previous_limits["rerank_top_n"],
            kg_rerank_top_n=previous_limits["kg_rerank_top_n"],
        )
        set_prompt_context_budget(previous_context_budget)

    return best_config


def evaluate_pipeline(test_data_path: str, enable_calibration: bool = False) -> dict[str, Any]:
    """Run benchmark with one execution per query/mode unless calibration is explicitly enabled."""
    dataset = _load_test_data(test_data_path)

    selected_config: dict[str, int]
    if enable_calibration:
        selected_config = _calibrate(dataset)
        set_retrieval_limits(
            vector_top_k=selected_config["vector_top_k"],
            rerank_top_n=selected_config["rerank_top_n"],
            kg_rerank_top_n=selected_config["kg_rerank_top_n"],
        )
        set_prompt_context_budget(selected_config["context_budget"])
    else:
        current_limits = get_retrieval_limits()
        selected_config = {
            "vector_top_k": current_limits["vector_top_k"],
            "rerank_top_n": current_limits["rerank_top_n"],
            "kg_rerank_top_n": current_limits["kg_rerank_top_n"],
            "context_budget": get_prompt_context_budget(),
        }

    run_cache: dict[tuple[str, str, int, int, int, int], Any] = {}
    records = _run_once(dataset, run_cache=run_cache)
    summary = _build_summary(records)

    return {
        "metadata": {
            "modes": MODES,
            "test_data_path": test_data_path,
            "query_count": len(dataset),
            "selected_config": selected_config,
            "single_run_per_mode": not enable_calibration,
            "calibration_enabled": enable_calibration,
        },
        "summary": summary,
        "records": records,
    }