"""
Public interface for the Test LLM component.

This module provides a simple LLM client using FLAN-T5 for generating answers
from prompts. Used by the orchestration layer for testing/benchmarking.
"""

from __future__ import annotations

import logging
from typing import Optional

from transformers import T5Tokenizer, T5ForConditionalGeneration  # type: ignore[import]

from .config import TestLLMSettings, get_test_llm_settings

logger = logging.getLogger(__name__)

# Global model and tokenizer (lazy load)
_model: Optional[T5ForConditionalGeneration] = None
_tokenizer: Optional[T5Tokenizer] = None


def _load_model(settings: TestLLMSettings) -> None:
    """Lazy load the FLAN-T5 model and tokenizer."""
    global _model, _tokenizer
    if _model is None:
        logger.info("Loading FLAN-T5 model '%s' on device '%s'", settings.model_name, settings.device)
        _tokenizer = T5Tokenizer.from_pretrained(settings.model_name)
        _model = T5ForConditionalGeneration.from_pretrained(settings.model_name)
        if settings.device == "cuda":
            _model.to("cuda")
        logger.info("Model loaded successfully")


def generate_answer(prompt: str) -> str:
    """
    Generate an answer from the given prompt using FLAN-T5.

    Args:
        prompt: The input prompt string.

    Returns:
        Generated answer text.
    """
    settings = get_test_llm_settings()
    _load_model(settings)

    logger.debug("Generating answer for prompt length %d", len(prompt))

    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if settings.device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    outputs = _model.generate(
        **inputs,
        max_length=settings.max_length,
        temperature=settings.temperature,
        do_sample=settings.temperature > 0,
        num_beams=1 if settings.temperature == 0 else 4,
    )

    answer = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.debug("Generated answer length %d", len(answer))
    return answer