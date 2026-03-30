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
    max_new_tokens: int = Field(default=500, description="Max new tokens to generate for longer answers")
    num_beams: int = Field(default=5, description="Beam width for better quality")
    repetition_penalty: float = Field(default=1.05, description="Lower penalty for more natural speech")
    no_repeat_ngram_size: int = Field(default=2, description="Allow more word repetition for natural answers")
    hf_token: str | None = Field(default=None, description="Hugging Face token for authenticated requests")

    class Config:
        env_prefix = "TEST_LLM_"


def get_test_llm_settings() -> TestLLMSettings:
    """Load settings from environment."""
    return TestLLMSettings()