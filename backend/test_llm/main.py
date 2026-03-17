"""
Public interface for the Test LLM component.

This module provides a simple LLM client using FLAN-T5 for generating answers
from prompts. Used by the orchestration layer for testing/benchmarking.
"""

from __future__ import annotations

import logging
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]

from .config import TestLLMSettings, get_test_llm_settings

logger = logging.getLogger(__name__)

# Global model and tokenizer (lazy load)
_model: Optional[AutoModelForCausalLM] = None
_tokenizer: Optional[AutoTokenizer] = None


def _load_model(settings: TestLLMSettings) -> None:
    """Lazy load the TinyLlama model and tokenizer."""
    global _model, _tokenizer
    if _model is None:
        logger.info("Loading TinyLlama model '%s' on device '%s'", settings.model_name, settings.device)
        _tokenizer = AutoTokenizer.from_pretrained(settings.model_name, token=settings.hf_token)
        _model = AutoModelForCausalLM.from_pretrained(settings.model_name, token=settings.hf_token)
        if settings.device == "cuda":
            _model.to("cuda")
        logger.info("Model loaded successfully")


def generate_answer(prompt: str) -> str:
    """
    Generate an answer from the given prompt using TinyLlama.

    Args:
        prompt: The input prompt string (assumed to be in chat format).

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
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        top_k=settings.top_k,
        top_p=settings.top_p,
        repetition_penalty=settings.repetition_penalty,
        no_repeat_ngram_size=settings.no_repeat_ngram_size,
        do_sample=True,
        pad_token_id=_tokenizer.eos_token_id,
    )

    full_output = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the input prompt from the output
    answer = full_output[len(prompt):].strip()
    logger.debug("Generated answer length %d", len(answer))
    return answer