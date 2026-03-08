"""
CLI test for the knowledge graph (Neo4j) component.

Run from project root:
    python -m backend.scripts.test_kg --question "Beaches in Mangalore"

The script loads `backend/.env` and attempts to connect to Neo4j using
`KG_NEO4J_URI`, `KG_NEO4J_USERNAME`, `KG_NEO4J_PASSWORD` (or the defaults).
It exercises `answer_question()` to return a human-readable result.
"""

from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv

# Silence noisy loggers
for _name in ("httpx", "huggingface_hub", "transformers", "urllib3"):
    logging.getLogger(_name).setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

from backend.kg import answer_question, _get_driver, get_kg_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test KG connectivity and a sample query.")
    parser.add_argument("--question", type=str, default="Beaches in Mangalore", help="Natural language question to run against the KG.")
    args = parser.parse_args(argv)

    # Load only backend/.env
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    logger.info("Loading environment from %s", env_path)
    load_dotenv(dotenv_path=env_path, override=False)

    # Quick settings print
    try:
        settings = get_kg_settings()
        logger.info("KG settings: uri=%s user=%s", settings.neo4j_uri, settings.neo4j_username)
    except Exception as e:
        logger.exception("Failed to load KG settings: %s", e)
        return 2

    # Try driver
    try:
        d = _get_driver()
        logger.info("Connected to Neo4j driver: %s", d)
    except Exception as e:
        logger.exception("Failed to create Neo4j driver: %s", e)
        return 3

    try:
        answer = answer_question(args.question)
        print("\n=== KG Answer ===\n")
        print(answer)
        return 0
    except Exception as e:
        logger.exception("KG query failed: %s", e)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
