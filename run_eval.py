"""
run_eval.py — Karnataka Tourism Pipeline Evaluation Runner
===========================================================
Place in: C:\\Users\\shriy\\Major_Project\\

Run:
    python run_eval.py

Requires:
    - backend/.env with all credentials
    - pip install sentence-transformers matplotlib tabulate
"""

from __future__ import annotations

# ── Load .env BEFORE any backend imports ──────────────────
import os
import sys
import time
from dotenv import load_dotenv
load_dotenv()                      # root .env
load_dotenv("backend/.env")        # backend/.env (overrides if set)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Backend imports ────────────────────────────────────────
from backend.kg.main          import run_query
from backend.vector_db        import fetch_top_vectordb
from backend.reranker         import rerank_top_cross_encoder
from backend.services.orchestration import run_pipeline

# ── Evaluator ─────────────────────────────────────────────
from evaluator import Evaluator


# ─────────────────────────────────────────────────────────
# KG PIPELINE FUNCTION
# Directly queries Neo4j using Cypher built from query keywords
# Bypasses extract_entities() which doesn't detect types well
# ─────────────────────────────────────────────────────────
TYPE_MAP = {
    "beach":         ["Beach", "Beach and Temple"],
    "beaches":       ["Beach", "Beach and Temple"],
    "temple":        ["Temple", "Beach and Temple"],
    "temples":       ["Temple", "Beach and Temple"],
    "waterfall":     ["Waterfall"],
    "waterfalls":    ["Waterfall"],
    "trek":          ["Trekking Destination", "Hill Station"],
    "trekking":      ["Trekking Destination", "Hill Station"],
    "restaurant":    ["Restaurant", "Seafood Restaurants", "Cafe", "Dessert Shop", "Dessert Restaurant"],
    "restaurants":   ["Restaurant", "Seafood Restaurants", "Cafe", "Dessert Shop", "Dessert Restaurant"],
    "historical":    ["Historical Site", "Historical Fort", "Historical Monument", "Heritage Site", "Monument", "Palace"],
    "national park": ["National Park"],
    "national parks":["National Park"],
    "church":        ["Church"],
    "churches":      ["Church"],
    "shopping":      ["Shopping Mall", "Shopping Area"],
    "park":          ["Park", "National Park", "Nature Park"],
    "parks":         ["Park", "National Park", "Nature Park"],
    "waterpark":     ["Water Park"],
    "lake":          ["Lake"],
    "viewpoint":     ["Viewpoint", "Hill Viewpoint"],
    "viewpoints":    ["Viewpoint", "Hill Viewpoint"],
}

CITY_MAP = {
    "mangalore":      "Mangalore",
    "udupi":          "Udupi",
    "hampi":          "Hampi",
    "madikeri":       "Madikeri",
    "chikkamagaluru": "Chikkamagaluru",
    "mysuru":         "Mysuru",
    "mysore":         "Mysore",
    "gokarna":        "Gokarna",
    "karwar":         "Karwar",
    "karkala":        "Karkala",
    "kundapura":      "Kundapura",
}

STATE_MAP = {
    "karnataka": "Karnataka",
}


def kg_pipeline_fn(query: str) -> list[dict]:
    """
    Build and run Cypher query for given natural language query.
    Returns list of dicts with name, type, city, description.
    """
    q = query.lower()

    # Detect types
    detected_types = []
    for keyword, types in TYPE_MAP.items():
        if keyword in q:
            detected_types = types
            break

    # Detect city
    detected_city = None
    for keyword, city in CITY_MAP.items():
        if keyword in q:
            detected_city = city
            break

    # Detect state
    detected_state = None
    for keyword, state in STATE_MAP.items():
        if keyword in q:
            detected_state = state
            break

    if not detected_types:
        return []

    # Build Cypher
    if detected_city:
        cypher = """
        MATCH (p:Place)-[:LOCATED_IN]->(c:City)
        WHERE p.type IN $types AND c.name = $location
        RETURN p.name AS name, p.type AS type, c.name AS city,
               p.description AS description,
               p.best_time AS best_time,
               p.entry_fee AS entry_fee
        ORDER BY p.name LIMIT 20
        """
        params = {"types": detected_types, "location": detected_city}

    elif detected_state:
        # State-level: Place -> LOCATED_IN -> City -> IN_DISTRICT -> District -> IN_STATE -> State
        cypher = """
        MATCH (p:Place)-[:LOCATED_IN]->(c:City)
        OPTIONAL MATCH (c)-[:IN_DISTRICT]->(d:District)
        OPTIONAL MATCH (d)-[:IN_STATE]->(s:State)
        WHERE p.type IN $types AND s.name = $location
        RETURN p.name AS name, p.type AS type, c.name AS city,
               p.description AS description,
               p.best_time AS best_time,
               p.entry_fee AS entry_fee
        ORDER BY p.name LIMIT 20
        """
        params = {"types": detected_types, "location": detected_state}

    else:
        cypher = """
        MATCH (p:Place)
        WHERE p.type IN $types
        RETURN p.name AS name, p.type AS type,
               p.description AS description,
               p.best_time AS best_time,
               p.entry_fee AS entry_fee
        ORDER BY p.name LIMIT 20
        """
        params = {"types": detected_types}

    try:
        return run_query(cypher, params)
    except Exception as e:
        print(f"  KG query error: {e}")
        return []


# ─────────────────────────────────────────────────────────
# VECTOR PIPELINE FUNCTION
# Fetches from Pinecone and returns top snippet after reranking
# ─────────────────────────────────────────────────────────
def vector_pipeline_fn(query: str) -> str | None:
    """
    Fetch from Pinecone and return top text snippet after reranking.
    """
    try:
        candidates, _ = fetch_top_vectordb(n=15, query=query, include_debug=False)
        if not candidates:
            return None
        reranked, _ = rerank_top_cross_encoder(
            query=query,
            candidates=candidates,
            top_n=5,
            include_debug=False
        )
        if reranked:
            # Return top 5 snippets combined for better coverage
            return " ".join(r.text for r in reranked if r.text)
        return None
    except Exception as e:
        print(f"  Vector query error: {e}")
        return None


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "=" * 62)
    print("  Karnataka Tourism — Pipeline Evaluation")
    print("=" * 62)
    print("\nWhat do you want to test?")
    print("  1 — KG only benchmark     (Precision, Recall, Grounding)")
    print("  2 — Vector DB benchmark   (Relevance, Semantic Coverage)")
    print("  3 — Both KG + Vector      (run both)")
    print("  4 — Single query test     (quick test one query)")

    choice = input("\nEnter choice (1/2/3/4): ").strip()

    evaluator = Evaluator()

    if choice == "1":
        print("\n🔬 Running KG benchmark...")
        results = evaluator.run_kg_benchmark(
            kg_pipeline_fn=kg_pipeline_fn,
            save_json="kg_eval_results.json",
            save_csv="kg_eval_results.csv"
        )
        evaluator.plot_kg_results(results, save_path="kg_eval_chart.png")
        print("\n✅ Done! Files: kg_eval_results.csv, kg_eval_chart.png")

    elif choice == "2":
        print("\n🔬 Running Vector DB benchmark...")
        results = evaluator.run_vector_benchmark(
            vector_pipeline_fn=vector_pipeline_fn,
            save_json="vector_eval_results.json",
            save_csv="vector_eval_results.csv"
        )
        evaluator.plot_vector_results(results, save_path="vector_eval_chart.png")
        print("\n✅ Done! Files: vector_eval_results.csv, vector_eval_chart.png")

    elif choice == "3":
        print("\n🔬 Running KG benchmark...")
        kg_results = evaluator.run_kg_benchmark(
            kg_pipeline_fn=kg_pipeline_fn,
            save_json="kg_eval_results.json",
            save_csv="kg_eval_results.csv"
        )
        evaluator.plot_kg_results(kg_results, save_path="kg_eval_chart.png")

        print("\n🔬 Running Vector DB benchmark...")
        vec_results = evaluator.run_vector_benchmark(
            vector_pipeline_fn=vector_pipeline_fn,
            save_json="vector_eval_results.json",
            save_csv="vector_eval_results.csv"
        )
        evaluator.plot_vector_results(vec_results, save_path="vector_eval_chart.png")
        print("\n✅ Done! Check all CSV and chart files.")

    elif choice == "4":
        print("\nQuick single query test")
        query  = input("Enter query: ").strip()
        source = input("Test (kg/vector/both): ").strip().lower()

        if source in ("kg", "both"):
            print("\n── KG Results ──")
            start = time.time()
            kg_res = kg_pipeline_fn(query)
            elapsed = (time.time() - start) * 1000
            print(f"Returned {len(kg_res)} results in {elapsed:.0f}ms")
            for r in kg_res[:5]:
                print(f"  {r.get('name')} | {r.get('type')} | {r.get('city','')}")
            m = evaluator.evaluate_kg(query, kg_res, elapsed)
            evaluator.print_kg_metrics(m)

        if source in ("vector", "both"):
            print("\n── Vector Results ──")
            start = time.time()
            snippet = vector_pipeline_fn(query)
            elapsed = (time.time() - start) * 1000
            print(f"Snippet in {elapsed:.0f}ms:")
            print(f"  {snippet[:200] if snippet else 'None'}...")
            m = evaluator.evaluate_vector(query, snippet, elapsed)
            evaluator.print_vector_metrics(m)

    else:
        print("Invalid choice.")
