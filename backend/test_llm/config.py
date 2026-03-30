"""
Configuration for the Test LLM component.

Uses Pydantic settings for model parameters.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class TestLLMSettings(BaseSettings):
    """
    Settings for the Test LLM (FLAN-T5-large).

    Optimized to generate 2-4 sentence elaborated answers with natural language structure.
    The key parameters for longer outputs are length_penalty and min_length.
    """

    model_name: str = Field(default="google/flan-t5-large", description="Hugging Face model name")
    device: str = Field(default="cpu", description="Device: cpu, cuda, or mps (Apple Silicon)")
    max_input_tokens: int = Field(default=1024, description="Max input tokens")
    max_new_tokens: int = Field(default=150, description="Max output tokens (tuned for 2-4 sentences)")
    num_beams: int = Field(default=4, description="Beam search width for balanced quality/speed")
    repetition_penalty: float = Field(default=1.2, description="Penalize repetitive tokens")
    no_repeat_ngram_size: int = Field(default=3, description="Prevent repeated n-grams")
    length_penalty: float = Field(default=2.0, description="CRITICAL: Strongly encourage longer sequences (solves early stopping)")
    min_length: int = Field(default=50, description="CRITICAL: Force minimum output length (prevents truncation)")
    hf_token: str | None = Field(default=None, description="Hugging Face token for authenticated requests")

    class Config:
        env_prefix = "TEST_LLM_"


def get_test_llm_settings() -> TestLLMSettings:
    """Load settings from environment."""
    return TestLLMSettings()