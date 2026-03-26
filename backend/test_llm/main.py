"""
Public interface for the Test LLM component.

This module provides a simple LLM client using FLAN-T5-large for generating answers
from prompts. Used by the orchestration layer for testing/benchmarking.
"""

from __future__ import annotations

import logging
from typing import Optional

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore[import]

from .config import TestLLMSettings, get_test_llm_settings

logger = logging.getLogger(__name__)

# Global model and tokenizer (lazy load)
_model: Optional[AutoModelForSeq2SeqLM] = None
_tokenizer: Optional[AutoTokenizer] = None


def _load_model(settings: TestLLMSettings) -> None:
    """Lazy load the FLAN-T5 model and tokenizer."""
    global _model, _tokenizer
    if _model is None:
        logger.info("Loading FLAN model '%s' on device '%s'", settings.model_name, settings.device)
        _tokenizer = AutoTokenizer.from_pretrained(settings.model_name, token=settings.hf_token)
        _model = AutoModelForSeq2SeqLM.from_pretrained(settings.model_name, token=settings.hf_token)
        if settings.device == "cuda":
            _model.to("cuda")
        logger.info("Model loaded successfully")


def generate_answer(prompt: str) -> str:
    """
    Generate an answer from the given prompt using FLAN-T5-large.

    Args:
        prompt: The input prompt string in instruction format.

    Returns:
        Generated answer text.
    """
    settings = get_test_llm_settings()
    _load_model(settings)

    logger.debug("Generating answer for prompt length %d", len(prompt))

    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=settings.max_input_tokens,
    )
    if settings.device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    outputs = _model.generate(
        **inputs,
        max_new_tokens=settings.max_new_tokens,
        num_beams=settings.num_beams,
        repetition_penalty=settings.repetition_penalty,
        no_repeat_ngram_size=settings.no_repeat_ngram_size,
        do_sample=False,
        pad_token_id=_tokenizer.pad_token_id,
    )

    answer = _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    logger.debug("Generated answer length %d", len(answer))
    return answer