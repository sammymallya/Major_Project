"""
Configuration for the Hallucination Rate module.

Uses Pydantic settings for hallucination detection parameters.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class HallucinationRateSettings(BaseSettings):
    """
    Settings for the Hallucination Rate component.

    Configurable parameters for claim verification.
    """

    similarity_threshold: float = Field(default=0.7, description="Similarity threshold for considering a claim verified")
    embedding_model_name: str = Field(default="all-roberta-large-v1", description="Embedding model for claim verification")
    vector_top_k: int = Field(default=5, description="Top K results from vector DB for verification")
    kg_top_k: int = Field(default=5, description="Top K triples from KG for verification")

    model_config = SettingsConfigDict(
        env_file=("backend/.env",),
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )


def get_hallucination_rate_settings() -> HallucinationRateSettings:
    """Load settings from environment."""
    return HallucinationRateSettings()