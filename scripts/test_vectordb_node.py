"""Quick diagnostic for vector DB node connectivity and retrieval.

Run from project root with the root venv, for example:

    ./venv/bin/python scripts/test_vectordb_node.py \
        --query "beaches in Mangalore" \
        --top-k 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on import path when script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.vector_db.main import VectorDBClient


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check vector DB contactability and retrieval via vector_db node.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="beaches in Mangalore",
        help="Query string used for retrieval test.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to request.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    print("[1/2] Contactability check: initializing VectorDBClient...")
    try:
        client = VectorDBClient()
        print("  OK: VectorDBClient initialized.")
        print(f"  Index: {client.settings.pinecone_index_name}")
        print(f"  Namespace: {client.settings.pinecone_namespace}")
        print(f"  Embedding: Pinecone inference API (llama-text-embed-v2)")
    except Exception as exc:
        print(f"  FAIL: Unable to initialize vector DB client: {exc}")
        return 1

    print("\n[2/2] Retrieval check: querying vector_db node...")
    try:
        results, debug = client.fetch_top(args.top_k, args.query, include_debug=True)
    except Exception as exc:
        print(f"  FAIL: Query request failed: {exc}")
        return 2

    print("  OK: Query executed.")
    print(f"  Requested top_k: {args.top_k}")
    print(f"  Returned results: {len(results)}")

    if debug is not None:
        print(f"  Debug model: {debug.model_name}")
        print(f"  Debug namespace: {debug.namespace}")
        if debug.top_scores:
            print(f"  Top scores: {[round(score, 4) for score in debug.top_scores]}")

    if not results:
        print("  WARNING: No retrieval results were returned.")
        print("  This still confirms the node is reachable, but your index/namespace may be empty or mismatched.")
        return 0

    print("\nTop results:")
    for i, item in enumerate(results, start=1):
        snippet = (item.text or "").strip().replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        print(f"  {i}. id={item.id} score={item.score:.4f} text={snippet}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
