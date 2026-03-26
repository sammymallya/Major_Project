"""
Configuration for the Test LLM component.

Uses Pydantic settings for model parameters.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class TestLLMSettings(BaseSettings):
    """
    Settings for the Test LLM (FLAN-T5-large).

    Configurable model parameters.
    """

    model_name: str = Field(default="google/flan-t5-large", description="Hugging Face model name")
    device: str = Field(default="cpu", description="Device: cpu or cuda")
    max_input_tokens: int = Field(default=1024, description="Max input tokens")
    max_new_tokens: int = Field(default=220, description="Max new tokens to generate")
    num_beams: int = Field(default=4, description="Beam width for deterministic decoding")
    repetition_penalty: float = Field(default=1.1, description="Repetition penalty")
    no_repeat_ngram_size: int = Field(default=3, description="No repeat n-gram size")
    hf_token: str | None = Field(default=None, description="Hugging Face token for authenticated requests")

    class Config:
        env_prefix = "TEST_LLM_"


def get_test_llm_settings() -> TestLLMSettings:
    """Load settings from environment."""
    return TestLLMSettings()