"""
Configuration for the Query Structurer (LLM-powered) component.

Settings for the Gemini API used to produce structured semantic and Cypher queries.
API key and model name can be changed here without affecting other components.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class QueryStructurerSettings(BaseSettings):
    """
    Settings for the Query Structurer LLM (Gemini API).

    Uses GEMINI_API_KEY and model name (e.g. gemini-1.5-flash). Change these
    in .env to switch models without touching the rest of the system.
    """

    api_key: str = Field(..., alias="GEMINI_API_KEY")
    model_name: str = Field(
        default="gemini-2.5-flash",
        alias="GEMINI_MODEL_NAME",
    )

    model_config = SettingsConfigDict(
        env_file=("backend/.env",),
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )


def get_query_structurer_settings() -> QueryStructurerSettings:
    """Return a configured instance of QueryStructurerSettings."""
    settings = QueryStructurerSettings()
    # Backwards-compatible env var name: allow GEMINI_MODEL to override
    # (some users set GEMINI_MODEL in backend/.env). Respect it if present.
    import os

    gemini_model = os.environ.get("GEMINI_MODEL")
    if gemini_model:
        settings.model_name = gemini_model
    return settings
