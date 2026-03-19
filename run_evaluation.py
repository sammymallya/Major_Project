"""
Run the evaluation engine.

Loads test_data.json, runs evaluation, saves results to evaluation_results.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from backend.evaluation_engine import evaluate_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the evaluation engine.")
    parser.add_argument(
        "--test-data",
        type=str,
        default="test_data.json",
        help="Path to test data JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Path to output results JSON file.",
    )

    args = parser.parse_args(argv)

    if not os.path.exists(args.test_data):
        logger.error("Test data file not found: %s", args.test_data)
        return 1

    logger.info("Starting evaluation with test data: %s", args.test_data)

    try:
        results = evaluate_pipeline(args.test_data)

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info("Evaluation complete. Results saved to: %s", args.output)
        return 0

    except Exception as e:
        logger.exception("Evaluation failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())