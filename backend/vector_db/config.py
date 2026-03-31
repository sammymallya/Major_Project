"""
Configuration for the vector database (Pinecone) component.

This module defines strongly-typed settings for connecting to Pinecone
and for controlling the local embedding model used to encode queries.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VectorDBSettings(BaseSettings):
    """
    Settings for the vector database component.

    Values are loaded from environment variables, with optional support
    for a `.env` file in the project root.
    
    NOTE: Query embedding is handled by Pinecone's serverless embedding
    (llama-text-embed-v2), not by a local embedding model. This ensures
    consistency with the data embeddings used during upload.
    """

    pinecone_api_key: str = Field(..., alias="VECTORDB_PINECONE_API_KEY")
    pinecone_environment: str = Field(..., alias="VECTORDB_PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field(..., alias="VECTORDB_PINECONE_INDEX_NAME")

    # Optional namespace support for isolating experiments within an index
    # Default to 'tourism' to target the correct dataset namespace.
    pinecone_namespace: str | None = Field(
        default="tourism", alias="VECTORDB_PINECONE_NAMESPACE"
    )

    # pydantic-settings configuration
    model_config = SettingsConfigDict(
        # Look for both a project-root .env and the existing backend/.env
        env_file=(".env", "backend/.env"),
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )


def get_vectordb_settings() -> VectorDBSettings:
    """
    Return a configured instance of `VectorDBSettings`.

    This helper exists so that other modules can import a single function
    instead of depending directly on `BaseSettings` internals.
    """

    return VectorDBSettings()

