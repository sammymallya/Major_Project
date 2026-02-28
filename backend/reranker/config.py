"""
Configuration for the cross-encoder re-ranking component.

This module defines strongly-typed settings for loading a CrossEncoder model
used to re-score candidate passages retrieved from the vector database.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RerankerSettings(BaseSettings):
    """
    Settings for the cross-encoder re-ranking component.

    Environment variables are prefixed with `RERANKER_` to keep configuration
    isolated and easy to swap without touching other parts of the system.
    """

    cross_encoder_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        alias="RERANKER_CROSS_ENCODER_MODEL_NAME",
    )

    # Optional explicit device selection (e.g., "cpu", "cuda", "mps").
    # If unset, sentence-transformers will choose an appropriate default.
    device: str | None = Field(default=None, alias="RERANKER_DEVICE")

    # Batch size for scoring (trade-off between speed and memory usage).
    batch_size: int = Field(default=16, alias="RERANKER_BATCH_SIZE")

    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env"),
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )


def get_reranker_settings() -> RerankerSettings:
    """
    Return a configured instance of `RerankerSettings`.
    """

    return RerankerSettings()

