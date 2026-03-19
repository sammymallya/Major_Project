"""
Configuration for the F1 Score module.

Uses Pydantic settings for ROUGE configuration.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class F1ScoreSettings(BaseSettings):
    """
    Settings for the F1 Score component.

    Configurable ROUGE parameters.
    """

    rouge_type: str = Field(default="rouge1", description="ROUGE type: rouge1, rouge2, rougeL")
    use_stemmer: bool = Field(default=True, description="Whether to use stemming in ROUGE")

    model_config = SettingsConfigDict(
        env_file=("backend/.env",),
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )


def get_f1_score_settings() -> F1ScoreSettings:
    """Load settings from environment."""
    return F1ScoreSettings()