"""
Configuration for the Test LLM component.

Uses Pydantic settings for model parameters.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class TestLLMSettings(BaseSettings):
    """
    Settings for the Test LLM (FLAN-T5).

    Configurable model name, device, max length, etc.
    """

    model_name: str = Field(default="google/flan-t5-large", description="Hugging Face model name")
    device: str = Field(default="cpu", description="Device: cpu or cuda")
    max_length: int = Field(default=512, description="Max generation length")
    temperature: float = Field(default=0.0, description="Generation temperature (0 for deterministic)")

    class Config:
        env_prefix = "TEST_LLM_"


def get_test_llm_settings() -> TestLLMSettings:
    """Load settings from environment."""
    return TestLLMSettings()