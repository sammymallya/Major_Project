"""
Configuration for the Semantic Matching module.

Uses Pydantic settings for embedding model configuration.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SemanticMatchingSettings(BaseSettings):
    """
    Settings for the Semantic Matching component.

    Configurable embedding model for computing similarities.
    """

    embedding_model_name: str = Field(default="all-roberta-large-v1", description="Sentence transformer model for embeddings")

    model_config = SettingsConfigDict(
        env_file=("backend/.env",),
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )


def get_semantic_matching_settings() -> SemanticMatchingSettings:
    """Load settings from environment."""
    return SemanticMatchingSettings()