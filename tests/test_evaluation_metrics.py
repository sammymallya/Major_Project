from backend.evaluation_engine.metrics import compute_f1, compute_hallucination_metrics, extract_claims


def test_compute_f1_handles_empty_answers() -> None:
    result = compute_f1("", "")
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_extract_claims_splits_sentences() -> None:
    claims = extract_claims("Mangalore has beaches. Udupi has temples! Yes?")
    assert claims == [
        "Mangalore has beaches.",
        "Udupi has temples!",
        "Yes?",
    ]


def test_hallucination_is_full_without_context() -> None:
    result = compute_hallucination_metrics(
        answer="Panambur Beach is in Mangalore. It is popular.",
        retrieved_context=[],
    )
    assert result["total_claims"] == 2
    assert result["grounded_claims"] == 0
    assert result["ungrounded_claims"] == 2
    assert result["hallucination_rate"] == 1.0
    assert result["faithfulness"] == 0.0


def test_hallucination_handles_empty_answer() -> None:
    result = compute_hallucination_metrics(answer="", retrieved_context=["Any context"])
    assert result["total_claims"] == 0
    assert result["hallucination_rate"] == 0.0
    assert result["faithfulness"] == 1.0
