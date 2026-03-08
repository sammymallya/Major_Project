"""
Configuration for the KG reranker component.

Settings for the cross-encoder used to re-rank KG triples.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class KgRerankerSettings(BaseSettings):
    """
    Settings for the KG reranker (cross-encoder).

    Uses KG_RERANKER_* prefixed env vars for settings.
    """

    cross_encoder_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        alias="KG_RERANKER_MODEL_NAME",
        description="Cross-encoder model name from HuggingFace.",
    )
    device: str | None = Field(
        default=None,
        alias="KG_RERANKER_DEVICE",
        description="Device: 'cpu', 'cuda', or None for auto-detect",
    )
    batch_size: int = Field(
        default=16,
        alias="KG_RERANKER_BATCH_SIZE",
        description="Batch size for cross-encoder predictions.",
    )

    model_config = SettingsConfigDict(
        env_file=("backend/.env",),
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )


def get_kg_reranker_settings() -> KgRerankerSettings:
    """Return a configured instance of KgRerankerSettings."""
    return KgRerankerSettings()
