"""
CLI test for the KG reranker component.

Run from project root:
    python -m backend.scripts.test_kg_reranker --question "Beaches in Mangalore"

The script loads `backend/.env`, fetches KG triples, and re-ranks them using
the cross-encoder to identify the top result.
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

from backend.kg import extract_entities, build_query, run_query
from backend.kg.types import KgTriple
from backend.kg_reranker import rerank_kg_triples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test KG reranker component.")
    parser.add_argument("--question", type=str, default="Beaches in Mangalore", help="Natural language question.")
    args = parser.parse_args(argv)

    # Load only backend/.env
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    logger.info("Loading environment from %s", env_path)
    load_dotenv(dotenv_path=env_path, override=False)

    try:
        # Extract entities and build query
        intent = extract_entities(args.question)
        logger.info("Extracted intent: %s", intent)

        cypher_query, params = build_query(intent)
        logger.debug("Built Cypher query: %s", cypher_query[:200])

        # Fetch raw KG triples
        raw_triples = run_query(cypher_query, params)
        logger.info("Fetched %d raw triples from KG", len(raw_triples))

        if not raw_triples:
            print("No triples returned from KG.")
            return 0

        # Print raw results
        print("\n=== Raw KG Results ===")
        for i, triple_dict in enumerate(raw_triples[:5], start=1):
            print(f"{i}. {triple_dict}")

        print("\n=== Testing KG Reranker ===")
        
        # For demo purposes, create simple KgTriple objects from the raw results
        kg_triples = []
        for result_dict in raw_triples:
            # Try to construct triples from the query results
            # Since the query returns {name, category, city, state, tags}
            # we'll create synthetic triples for demonstration
            name = result_dict.get('name', 'unknown')
            category = result_dict.get('category', '')
            city = result_dict.get('city', '')
            tags = result_dict.get('tags', [])
            
            # Create primary triple with category info
            kg_triples.append(KgTriple(
                subject=name,
                predicate="has_category",
                object=category if category else "uncategorized"
            ))
            
            # Add location triple if city exists
            if city:
                kg_triples.append(KgTriple(
                    subject=name,
                    predicate="located_in",
                    object=city
                ))
            
            # Add tags as triples if they exist
            if tags and isinstance(tags, list):
                for tag in tags[:3]:  # Limit to top 3 tags
                    kg_triples.append(KgTriple(
                        subject=name,
                        predicate="has_tag",
                        object=tag
                    ))

        logger.info("Created %d KgTriple objects for reranking", len(kg_triples))

        # Rerank the triples
        reranked, debug = rerank_kg_triples(
            query=args.question,
            triples=kg_triples,
            top_n=1,
            include_debug=True,
        )

        print("\n=== Reranked Results (Top 1) ===")
        for i, result in enumerate(reranked, start=1):
            print(f"{i}. Subject: {result.subject}")
            print(f"   Predicate: {result.predicate}")
            print(f"   Object: {result.object[:100]}")
            print(f"   Cross-encoder score: {result.cross_score:.4f}\n")

        if debug:
            print("=== Debug Info ===")
            print(f"Model: {debug.model_name}")
            print(f"Device: {debug.device}")
            print(f"Input count: {debug.input_count}")
            print(f"Total top scores: {debug.top_cross_scores}")
            print(f"Selected subject: {debug.selected_subject}")

        return 0

    except Exception as e:
        logger.exception("KG reranker test failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
