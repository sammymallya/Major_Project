"""Test script for the Prompt Generator component.

This script demonstrates how a final prompt is constructed from the query,
vector retrieval context, and knowledge graph triples.

Usage examples:
  python -m backend.scripts.test_prompt_generator --query "Beaches in Mangalore"

  python -m backend.scripts.test_prompt_generator \
    --query "Beaches in Mangalore" \
    --vector "Multiple guides mention Panambur Beach..." \
    --kg "Mangalore --has_beach--> Panambur Beach"

This script is intentionally lightweight and does not require any external services.
It is a good way to verify prompt formatting before wiring it into the full pipeline.
"""

from __future__ import annotations

import argparse

from backend.prompt_generator import build_prompt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test the prompt generator component.")
    parser.add_argument(
        "--query",
        type=str,
        default="Beaches in Mangalore",
        help="Natural language query to build a prompt for.",
    )
    parser.add_argument(
        "--vector",
        type=str,
        default="",
        help="Optional vector retrieval snippet (e.g., from a vector DB).",
    )
    parser.add_argument(
        "--kg",
        type=str,
        action="append",
        default=[],
        help="Optional KG triple(s) in the format 'subject --predicate--> object'. Can be passed multiple times.",
    )

    args = parser.parse_args(argv)

    # Convert KG triple strings into a minimal object compatible with KgTriple
    # (either a tuple or mapping with subject/predicate/object). We keep it simple.
    kg_triples = []
    for triple_text in args.kg:
        if "--" in triple_text and "-->" in triple_text:
            # Expect the format "subject --predicate--> object"
            subject, rest = triple_text.split("--", 1)
            predicate, object_part = rest.split("-->", 1)
            kg_triples.append({
                "subject": subject.strip(),
                "predicate": predicate.strip(),
                "object": object_part.strip(),
            })

    # Build the prompt
    prompt = build_prompt(args.query, args.vector or None, kg_triples or None)

    print("\n=== Generated Prompt ===\n")
    print(prompt)
    print("\n=== End Prompt ===\n")

    print("In the full pipeline (FastAPI -> orchestration), this prompt becomes the `final_prompt` field returned in the response.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
