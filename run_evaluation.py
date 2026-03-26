"""
Run the evaluation engine.

Loads test_data.json, runs evaluation, saves results to evaluation_results.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from backend.evaluation_engine import evaluate_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _write_csv_outputs(results: dict, csv_prefix: str) -> None:
    """Write per-query-per-mode and aggregate summary CSV files."""
    detail_path = f"{csv_prefix}_detail.csv"
    summary_path = f"{csv_prefix}_summary.csv"

    with open(detail_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "query",
                "mode",
                "semantic_similarity",
                "f1_score",
                "precision",
                "recall",
                "hallucination_rate",
                "faithfulness",
                "total_claims",
                "grounded_claims",
                "ungrounded_claims",
            ]
        )
        for record in results.get("records", []):
            query = record.get("query", "")
            modes = record.get("modes", {})
            for mode, values in modes.items():
                if "error" in values:
                    continue
                writer.writerow(
                    [
                        query,
                        mode,
                        values.get("semantic_similarity"),
                        values.get("f1_score"),
                        values.get("precision"),
                        values.get("recall"),
                        values.get("hallucination_rate"),
                        values.get("faithfulness"),
                        values.get("total_claims"),
                        values.get("grounded_claims"),
                        values.get("ungrounded_claims"),
                    ]
                )

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mode",
                "avg_semantic_similarity",
                "avg_f1_score",
                "avg_hallucination_rate",
                "avg_faithfulness",
                "improvement_semantic_pct",
                "improvement_f1_pct",
                "improvement_hallucination_pct",
                "improvement_faithfulness_pct",
            ]
        )
        summary = results.get("summary", {})
        for mode, values in summary.items():
            improvement = values.get("improvement_over_none_percent", {}) if isinstance(values, dict) else {}
            writer.writerow(
                [
                    mode,
                    values.get("avg_semantic_similarity") if isinstance(values, dict) else None,
                    values.get("avg_f1_score") if isinstance(values, dict) else None,
                    values.get("avg_hallucination_rate") if isinstance(values, dict) else None,
                    values.get("avg_faithfulness") if isinstance(values, dict) else None,
                    improvement.get("semantic_similarity") if isinstance(improvement, dict) else None,
                    improvement.get("f1_score") if isinstance(improvement, dict) else None,
                    improvement.get("hallucination_rate") if isinstance(improvement, dict) else None,
                    improvement.get("faithfulness") if isinstance(improvement, dict) else None,
                ]
            )

    logger.info("CSV outputs written: %s, %s", detail_path, summary_path)


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
    parser.add_argument(
        "--csv-prefix",
        type=str,
        default=None,
        help="Optional prefix to emit CSV mirror files.",
    )
    parser.add_argument(
        "--enable-calibration",
        action="store_true",
        help="Enable bounded pre-benchmark calibration sweep (disabled by default).",
    )

    args = parser.parse_args(argv)

    if not os.path.exists(args.test_data):
        logger.error("Test data file not found: %s", args.test_data)
        return 1

    logger.info("Starting evaluation with test data: %s", args.test_data)

    try:
        results = evaluate_pipeline(args.test_data, enable_calibration=args.enable_calibration)

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("Evaluation complete. Results saved to: %s", args.output)
        if args.csv_prefix:
            _write_csv_outputs(results, args.csv_prefix)
        return 0

    except Exception as e:
        logger.exception("Evaluation failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())