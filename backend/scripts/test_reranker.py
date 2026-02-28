"""
Simple CLI script to exercise vector retrieval + cross-encoder reranking.

Run from the project root, for example:

    python -m backend.scripts.test_reranker --query "best beaches near Mangalore" --top-k 10 --top-n 1
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings

from backend.reranker import RerankedVectorResult, rerank_top_cross_encoder
from backend.vector_db import fetch_top_vectordb


def _configure_logging(verbose: bool) -> None:
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

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        warnings.filterwarnings(
            "ignore",
            message=r"You are sending unauthenticated requests to the HF Hub\..*",
        )


def _print_top(r: RerankedVectorResult | None) -> None:
    print("\n=== Cross-Encoder Top Match ===")
    if r is None:
        print("No match selected.")
        return
    print(f"  id          : {r.id}")
    print(f"  vector_score: {r.vector_score:.4f}")
    print(f"  cross_score : {r.cross_score:.4f}")
    print(f"  text        : {r.text!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Test cross-encoder reranking over vector DB candidates.",
    )
    parser.add_argument("--query", type=str, required=True, help="Natural language query.")
    parser.add_argument("--top-k", type=int, default=10, help="Candidates to fetch from Pinecone.")
    parser.add_argument("--top-n", type=int, default=1, help="Results to keep after reranking.")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG-level logging.")

    args = parser.parse_args(argv)
    _configure_logging(verbose=args.verbose)

    logging.getLogger(__name__).info(
        "Fetching vector candidates: query=%r top_k=%d", args.query, args.top_k
    )
    candidates, _ = fetch_top_vectordb(n=args.top_k, query=args.query, include_debug=False)

    logging.getLogger(__name__).info(
        "Reranking %d candidate(s) with cross-encoder (top_n=%d)",
        len(candidates),
        args.top_n,
    )
    reranked, _ = rerank_top_cross_encoder(
        query=args.query,
        candidates=candidates,
        top_n=args.top_n,
        include_debug=True,
    )

    _print_top(reranked[0] if reranked else None)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

