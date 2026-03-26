from types import SimpleNamespace

from backend.evaluation_engine import main as eval_main


def test_run_once_produces_all_modes_and_metrics(monkeypatch) -> None:
    def fake_pipeline(query: str, test_mode: str):
        return SimpleNamespace(
            answer=f"answer-{test_mode}",
            test_mode=test_mode,
            final_prompt="prompt",
            retrieved_context=[f"context-{test_mode}"],
        )

    def fake_semantic(expected: str, actual: str) -> float:
        return 0.7 if "none" in actual else 0.8

    def fake_f1(expected: str, actual: str) -> dict[str, float]:
        return {"precision": 0.5, "recall": 0.5, "f1": 0.5}

    def fake_hallucination(answer: str, context: list[str]) -> dict:
        return {
            "hallucination_rate": 0.2,
            "faithfulness": 0.8,
            "total_claims": 1,
            "grounded_claims": 1,
            "ungrounded_claims": 0,
        }

    monkeypatch.setattr(eval_main, "run_pipeline_for_evaluation", fake_pipeline)
    monkeypatch.setattr(eval_main, "compute_semantic_similarity", fake_semantic)
    monkeypatch.setattr(eval_main, "compute_f1", fake_f1)
    monkeypatch.setattr(eval_main, "compute_hallucination_metrics", fake_hallucination)

    records = eval_main._run_once(
        dataset=[{"query": "beaches", "answer": "Panambur Beach"}],
        run_cache={},
    )

    assert len(records) == 1
    modes = records[0]["modes"]
    assert set(modes.keys()) == {"none", "vectordb", "kg", "hybrid"}

    for mode_name, mode_metrics in modes.items():
        assert "semantic_similarity" in mode_metrics
        assert "f1_score" in mode_metrics
        assert "hallucination_rate" in mode_metrics
        assert "faithfulness" in mode_metrics
        if mode_name != "none":
            assert mode_metrics["improvement_over_none_percent"] is not None


def test_safe_percent_improvement_handles_zero_baseline() -> None:
    assert eval_main._safe_percent_improvement(0.5, 0.0) is None
