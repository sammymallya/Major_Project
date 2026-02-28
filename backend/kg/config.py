"""
Configuration for the knowledge graph (Neo4j) component.

Defines settings for connecting to Neo4j Aura. Used by the KG client;
in stub mode the client does not perform real connections.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class KGSettings(BaseSettings):
    """
    Settings for the KG component.

    Environment variables are prefixed with KG_ so they can be overridden
    independently. In stub mode, connectivity is not required.
    """

    neo4j_uri: str = Field(
        default="neo4j+s://localhost",
        alias="KG_NEO4J_URI",
        description="Neo4j connection URI (e.g. neo4j+s://xxx.databases.neo4j.io)",
    )
    neo4j_username: str = Field(default="neo4j", alias="KG_NEO4J_USERNAME")
    neo4j_password: str = Field(default="", alias="KG_NEO4J_PASSWORD")

    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env"),
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )


def get_kg_settings() -> KGSettings:
    """Return a configured instance of KGSettings."""
    return KGSettings()
