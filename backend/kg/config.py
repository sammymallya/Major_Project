"""
Configuration for the knowledge graph (Neo4j) component.

Defines settings for connecting to Neo4j Aura. Used by the KG client;
in stub mode the client does not perform real connections.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import os


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
        env_file=("backend/.env",),
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )


def get_kg_settings() -> KGSettings:
    """Return a configured instance of KGSettings."""
    # Ensure backend/.env is loaded (helps when imports happen before test scripts)
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(dotenv_path=env_path, override=False)

    settings = KGSettings()

    # Backwards-compatible env var names: allow NEO4J_URI/USERNAME/PASSWORD
    # in addition to KG_NEO4J_* aliases.
    uri = os.environ.get("KG_NEO4J_URI") or os.environ.get("NEO4J_URI") or os.environ.get("KG_URI")
    user = os.environ.get("KG_NEO4J_USERNAME") or os.environ.get("NEO4J_USERNAME") or os.environ.get("NEO4J_USER")
    pwd = os.environ.get("KG_NEO4J_PASSWORD") or os.environ.get("NEO4J_PASSWORD") or os.environ.get("KG_NEO4J_PWD")

    if uri:
        settings.neo4j_uri = uri
    if user:
        settings.neo4j_username = user
    if pwd:
        settings.neo4j_password = pwd

    return settings
