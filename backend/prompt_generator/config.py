"""
Configuration for the Prompt Generator component.

Uses Pydantic settings for any configurable parameters.
"""

from pydantic_settings import BaseSettings


class PromptGeneratorSettings(BaseSettings):
    """
    Settings for prompt generation.

    Currently minimal; can add template paths, max lengths, etc. later.
    """

    class Config:
        env_prefix = "PROMPT_GENERATOR_"


def get_prompt_generator_settings() -> PromptGeneratorSettings:
    """Load settings from environment."""
    return PromptGeneratorSettings()