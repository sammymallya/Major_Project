"""
Configuration for the Test LLM component.

Uses Pydantic settings for model parameters.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class TestLLMSettings(BaseSettings):
    """
    Settings for the Test LLM (TinyLlama).

    Configurable model parameters.
    """

    model_name: str = Field(default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", description="Hugging Face model name")
    device: str = Field(default="cpu", description="Device: cpu or cuda")
    max_new_tokens: int = Field(default=512, description="Max new tokens to generate")
    temperature: float = Field(default=0.5, description="Generation temperature")
    top_k: int = Field(default=50, description="Top-k sampling")
    top_p: float = Field(default=0.95, description="Top-p sampling")
    repetition_penalty: float = Field(default=1.2, description="Repetition penalty")
    no_repeat_ngram_size: int = Field(default=3, description="No repeat n-gram size")
    hf_token: str | None = Field(default=None, description="Hugging Face token for authenticated requests")

    class Config:
        env_prefix = "TEST_LLM_"


def get_test_llm_settings() -> TestLLMSettings:
    """Load settings from environment."""
    return TestLLMSettings()