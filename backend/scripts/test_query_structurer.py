"""
Simple CLI script to test the Query Structurer (Gemini API).

Verifies that the Gemini API returns a response and that structure_query
produces a StructuredQuery. Run from the project root:

    python -m backend.scripts.test_query_structurer
    python -m backend.scripts.test_query_structurer --query "Where is Panambur Beach?"
"""

from __future__ import annotations

import argparse
import logging
import os
from dotenv import load_dotenv

# Silence noisy loggers so output is readable
for _name in ("httpx", "huggingface_hub", "sentence_transformers", "transformers", "urllib3"):
    logging.getLogger(_name).setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

from backend.query_structurer import structure_query, StructuredQuery


def _print_result(result: StructuredQuery) -> None:
    print(result)
    print("\n=== Query Structurer Result (Gemini) ===")
    print("semantic_search_query:", result.semantic_search_query)
    print("cypher_query:", result.cypher_query)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test Query Structurer (Gemini API).")
    parser.add_argument(
        "--query",
        type=str,
        default="Where is Thannirbhavi Beach located?",
        help="User question to structure (default: Karnataka tourism example).",
    )
    parser.add_argument(
        "--output-kind",
        type=str,
        choices=["vector_only", "kg_only", "both"],
        default="both",
        help="What to ask the structurer to output.",
    )
    args = parser.parse_args(argv)

    # Load only backend/.env as requested (do not read project root .env)
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    logging.info("Loading environment from %s", env_path)
    load_dotenv(dotenv_path=env_path, override=False)

    if not os.environ.get("GEMINI_API_KEY"):
        logging.error("GEMINI_API_KEY is not set. Set it in backend/.env or the environment.")
        return 1

    logging.info("Calling Gemini via structure_query(query=%r, output_kind=%s)", args.query, args.output_kind)
    try:
        result = structure_query(args.query, args.output_kind)
        _print_result(result)
        return 0
    except Exception as e:
        logging.exception("Query structurer test failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
