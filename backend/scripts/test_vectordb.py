"""
Simple CLI script to exercise the vector database component end-to-end.

Run from the project root, for example:

    python -m backend.scripts.test_vectordb --query "best beaches near Mangalore" --top-k 5
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings

from backend.vector_db import VectorResult, VectorQueryDebugInfo, fetch_top_vectordb


def _configure_logging(verbose: bool) -> None:
    """
    Configure basic logging for CLI usage.
    """

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Keep CLI output readable by silencing very chatty dependency loggers.
    # When `--verbose` is passed, we leave everything enabled for debugging.
    if not verbose:
        noisy_loggers = (
            "httpx",
            "huggingface_hub",
            "sentence_transformers",
            "transformers",
            "urllib3",
        )
        for name in noisy_loggers:
            logging.getLogger(name).setLevel(logging.WARNING)

        # Disable progress bars and suppress the common HF unauthenticated warning
        # (it does not affect correctness; it only impacts rate limits/download speed).
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        warnings.filterwarnings(
            "ignore",
            message=r"You are sending unauthenticated requests to the HF Hub\..*",
        )


def _print_results(results: list[VectorResult], debug: VectorQueryDebugInfo | None) -> None:
    """
    Print retrieved results and optional debug information to stdout.
    """

    print("\n=== Vector DB Results ===")
    if not results:
        print("No results returned.")
    for i, r in enumerate(results, start=1):
        print(f"\nResult #{i}")
        print(f"  id: {r.id}")
        print(f"  score: {r.score:.4f}")
        print(f"  text: {r.text!r}")
        if r.metadata:
            print(f"  metadata keys: {list(r.metadata.keys())}")

    if debug is not None:
        print("\n=== Debug Info ===")
        print(f"Embedding model : {debug.model_name}")
        print(f"Namespace       : {debug.namespace}")
        if debug.top_scores:
            print(f"Top scores      : {[round(s, 4) for s in debug.top_scores]}")


def main(argv: list[str] | None = None) -> int:
    """
    Entry point for the test script.
    """

    parser = argparse.ArgumentParser(
        description="Test the vector_db component using a sample query.",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Natural language query to send to the vector DB.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to retrieve.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG-level) logging.",
    )

    args = parser.parse_args(argv)

    _configure_logging(verbose=args.verbose)

    logging.getLogger(__name__).info(
        "Running vector DB test with query=%r, top_k=%d", args.query, args.top_k
    )

    try:
        results, debug = fetch_top_vectordb(
            n=args.top_k,
            query=args.query,
            include_debug=False,
        )
    except Exception as exc:  # pragma: no cover - simple CLI error reporting
        logging.getLogger(__name__).error("Vector DB test failed: %s", exc, exc_info=True)
        return 1

    _print_results(results, debug)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

