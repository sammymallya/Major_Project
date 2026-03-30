"""
Public interface for the Test LLM component.

This module provides a simple LLM client using FLAN-T5-large or causal LMs for generating answers
from prompts. Used by the orchestration layer for testing/benchmarking.
Supports both Seq2Seq models (FLAN-T5) and causal language models (Llama, Mistral, etc).
"""

from __future__ import annotations

import logging
from typing import Optional

from transformers import (  # type: ignore[import]
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)

from .config import TestLLMSettings, get_test_llm_settings

logger = logging.getLogger(__name__)

# Global model and tokenizer (lazy load)
_model: Optional[PreTrainedModel] = None
_tokenizer: Optional[AutoTokenizer] = None
_is_seq2seq: bool = False


def _load_model(settings: TestLLMSettings) -> None:
    """Lazy load the model (Seq2Seq or Causal LM) and tokenizer."""
    global _model, _tokenizer, _is_seq2seq
    if _model is None:
        logger.info("Loading model '%s' on device '%s'", settings.model_name, settings.device)
        _tokenizer = AutoTokenizer.from_pretrained(settings.model_name, token=settings.hf_token)
        
        # Ensure tokenizer has pad_token set (important for some models)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        
        # Try loading as Seq2Seq first (FLAN-T5, etc)
        try:
            _model = AutoModelForSeq2SeqLM.from_pretrained(settings.model_name, token=settings.hf_token)
            _is_seq2seq = True
            logger.info("Loaded as Seq2Seq model")
        except ValueError as e:
            # Fall back to causal LM (Llama, Mistral, etc)
            if "not compatible with AutoModelForSeq2SeqLM" in str(e) or "Unrecognized configuration class" in str(e):
                logger.info("Model not Seq2Seq, trying Causal LM...")
                _model = AutoModelForCausalLM.from_pretrained(settings.model_name, token=settings.hf_token)
                _is_seq2seq = False
                logger.info("Loaded as Causal LM model")
            else:
                raise
        
        if settings.device == "cuda":
            _model.to("cuda")
        elif settings.device == "mps":  # For Apple Silicon
            _model.to("mps")
        logger.info("Model loaded successfully")


def generate_answer(prompt: str) -> str:
    """
    Generate an answer from the given prompt using a Seq2Seq or Causal LM.

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
    
    # Move inputs to the same device as the model
    device = _model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Configure generation parameters based on model type
    gen_kwargs = {
        "max_new_tokens": settings.max_new_tokens,
        "do_sample": False,
    }
    
    if _is_seq2seq:
        # Seq2Seq models: use num_beams, repetition penalty, no_repeat_ngram
        gen_kwargs.update({
            "num_beams": settings.num_beams,
            "repetition_penalty": settings.repetition_penalty,
            "no_repeat_ngram_size": settings.no_repeat_ngram_size,
            "pad_token_id": _tokenizer.pad_token_id,
        })
    else:
        # Causal LM: simpler generation parameters
        gen_kwargs.update({
            "eos_token_id": _tokenizer.eos_token_id,
            "pad_token_id": _tokenizer.eos_token_id,
        })

    outputs = _model.generate(**inputs, **gen_kwargs)

    answer = _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    logger.debug("Generated answer length %d", len(answer))
    return answer