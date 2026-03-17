"""
CLI test for the Test LLM component.

Run from project root:
    python -m backend.scripts.test_test_llm --prompt "What is the capital of France?"
"""

from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv

# Silence noisy loggers
for _name in ("httpx", "huggingface_hub", "sentence_transformers", "transformers", "urllib3"):
    logging.getLogger(_name).setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

from backend.test_llm import generate_answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test Test LLM component.")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?", help="Prompt to generate answer for.")
    args = parser.parse_args(argv)

    # Load env
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(dotenv_path=env_path, override=False)

    logger.info("Generating answer for prompt: %s", args.prompt[:50] + "...")

    try:
        answer = generate_answer(args.prompt)
        print("\n=== LLM Answer ===")
        print(answer)
        return 0
    except Exception as e:
        logger.exception("Test LLM failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())