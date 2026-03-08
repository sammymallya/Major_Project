"""
run_eval.py
===========
Connects your evaluator to the existing orchestration pipeline.
Place this file in: C:\\Users\\shriy\\Major_Project\\

Run:
    python run_eval.py

Requirements:
    pip install transformers torch sentence-transformers tabulate matplotlib
"""

from __future__ import annotations

import sys
import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ── Make sure backend is importable ───────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.services.orchestration import run_pipeline
from backend.kg.main import run_query, extract_entities, build_query
from evaluator import Evaluator, TEST_SET


# ─────────────────────────────────────────────────────────
# FORMATTER — flan-t5-base
# Small model: formats KG/vector results into readable answer
# Cannot answer on its own — needs data from pipeline
# ─────────────────────────────────────────────────────────
class Formatter:
    MODEL_NAME = "google/flan-t5-base"

    def __init__(self):
        print(f"  Loading formatter ({self.MODEL_NAME})...")
        self.tokenizer = T5Tokenizer.from_pretrained(self.MODEL_NAME)
        self.model     = T5ForConditionalGeneration.from_pretrained(self.MODEL_NAME)
        self.model.eval()
        print("  Formatter ready.")

    def format(self, query: str, vector_snippet: str | None, kg_triples: list) -> str:
        """Format KG triples + vector snippet into natural language answer."""

        if not vector_snippet and not kg_triples:
            return "I could not find any matching places for your query."

        parts = []

        # KG triples → structured facts
        if kg_triples:
            facts = "\n".join(
                f"{t.subject} → {t.predicate} → {t.object}"
                for t in kg_triples[:5]
            )
            parts.append(f"Structured facts:\n{facts}")

        # Vector snippet → semantic context
        if vector_snippet:
            parts.append(f"Additional context:\n{vector_snippet[:300]}")

        context = "\n\n".join(parts)

        prompt = (
            f"Using only the following data, write a short helpful travel "
            f"response for the query: '{query}'\n\n"
            f"{context}\n"
            f"Response:"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────
# PIPELINE WRAPPER
# Converts orchestration output → list[dict] for evaluator
# ─────────────────────────────────────────────────────────
def pipeline_fn(query: str, mode: str) -> list[dict]:
    """
    Wrapper that connects evaluator to orchestration pipeline.

    Calls run_pipeline() and converts results to list[dict]
    format that evaluator.py understands.

    mode mapping:
        "kg"     → "kg"
        "vector" → "vectordb"
        "hybrid" → "hybrid"
        "none"   → "none"
    """
    # Map evaluator mode names to orchestration TestMode names
    mode_map = {
        "kg":     "kg",
        "vector": "vectordb",
        "hybrid": "hybrid",
        "none":   "none",
    }
    test_mode = mode_map.get(mode, "hybrid")

    try:
        response = run_pipeline(query=query, test_mode=test_mode)
    except Exception as e:
        print(f"  Pipeline error: {e}")
        return []

    results = []

    # ── Extract from KG triples ────────────────────────
    # KG returns triples (subject→predicate→object)
    # We convert them back to result dicts for evaluation
    if response.context_used and response.context_used.kg_triples_count:
        # Re-run KG query directly to get full structured results
        try:
            intent = extract_entities(query)
            cypher, params = build_query(intent)
            kg_data = run_query(cypher, params)
            for r in kg_data:
                results.append({
                    "name":        r.get("name", ""),
                    "type":        r.get("type", "Place"),
                    "city":        r.get("city", ""),
                    "description": r.get("description", ""),
                    "best_time":   r.get("best_time", ""),
                    "entry_fee":   r.get("entry_fee", ""),
                    "source":      "kg"
                })
        except Exception as e:
            print(f"  KG extraction error: {e}")

    # ── Extract from vector snippet ────────────────────
    if response.context_used and response.context_used.vector_snippet:
        snippet = response.context_used.vector_snippet
        # Vector returns text snippet — wrap as a result dict
        results.append({
            "name":        query,        # best we can do without metadata
            "type":        "",
            "city":        "",
            "description": snippet,
            "source":      "vector"
        })

    return results


# ─────────────────────────────────────────────────────────
# INTERACTIVE MODE — ask questions, see formatted answers
# ─────────────────────────────────────────────────────────
def run_interactive(formatter: Formatter):
    """Interactive session with mode selection and formatted answers."""

    MODES = {"1": "kg", "2": "vectordb", "3": "hybrid", "4": "none"}

    print("\n" + "=" * 58)
    print("  Karnataka Tourism — Interactive Query")
    print("=" * 58)
    print("\nSelect mode:")
    print("  1 — KG only     (Knowledge Graph)")
    print("  2 — Vector only (Pinecone)")
    print("  3 — Hybrid      (KG + Vector)  [default]")
    print("  4 — None        (no retrieval)")

    choice = input("\nEnter choice (1/2/3/4) or Enter for hybrid: ").strip()
    mode   = MODES.get(choice, "hybrid")
    print(f"\n✅ Mode: {mode.upper()}")
    print("   Type 'mode' to switch | 'eval' to run benchmark | 'quit' to exit")

    evaluator = Evaluator()

    while True:
        try:
            user_input = input("\n💬 Ask: ").strip()

            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            # Switch mode
            if user_input.lower() == "mode":
                print("\n  1 — KG | 2 — Vector | 3 — Hybrid | 4 — None")
                c    = input("  Choice: ").strip()
                mode = MODES.get(c, mode)
                print(f"  ✅ Switched to {mode.upper()}")
                continue

            # Run full benchmark
            if user_input.lower() == "eval":
                print("\n🔬 Running full benchmark...")
                results = evaluator.run_benchmark(
                    pipeline_fn=pipeline_fn,
                    modes=["kg", "vector", "hybrid", "none"],
                    save_json="eval_results.json",
                    save_csv="eval_results.csv"
                )
                evaluator.plot_results(results, save_path="eval_chart.png")
                continue

            # Run query through orchestration
            response = run_pipeline(query=user_input, test_mode=mode)

            # Format answer with flan-t5
            kg_triples    = []
            vector_snippet = None

            if response.context_used:
                vector_snippet = response.context_used.vector_snippet
                # Re-fetch KG triples for formatter
                if response.context_used.kg_triples_count:
                    from backend.kg.main import fetch_kg
                    from backend.query_structurer import structure_query
                    structured = structure_query(user_input, "kg_only")
                    if structured.cypher_query:
                        kg_triples = fetch_kg(structured.cypher_query)

            answer = formatter.format(user_input, vector_snippet, kg_triples)

            print(f"\n{'─'*58}")
            print(f"  ✅ ANSWER  [{mode.upper()}]")
            print(f"{'─'*58}")
            print(f"  {answer}")

            # Show quick metrics for this query (dev mode)
            raw_results = pipeline_fn(user_input, mode)
            if raw_results:
                metrics = evaluator.evaluate(
                    query=user_input,
                    results=raw_results,
                    mode=mode
                )
                evaluator.print_metrics(metrics)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "=" * 58)
    print("  Karnataka Tourism — Evaluation Runner")
    print("=" * 58)
    print("\nWhat do you want to do?")
    print("  1 — Interactive mode (ask questions)")
    print("  2 — Run full benchmark (all test queries, all modes)")
    print("  3 — Quick single query test")

    choice = input("\nEnter choice (1/2/3): ").strip()

    # Load formatter once
    formatter = Formatter()
    evaluator = Evaluator()

    if choice == "1":
        run_interactive(formatter)

    elif choice == "2":
        print("\n🔬 Running full benchmark across all modes...")
        results = evaluator.run_benchmark(
            pipeline_fn=pipeline_fn,
            modes=["kg", "vector", "hybrid", "none"],
            save_json="eval_results.json",
            save_csv="eval_results.csv"
        )
        evaluator.plot_results(results, save_path="eval_chart.png")
        print("\n✅ Done! Check eval_results.csv and eval_chart.png")

    elif choice == "3":
        query = input("\nEnter your query: ").strip()
        mode  = input("Mode (kg/vector/hybrid/none): ").strip() or "hybrid"

        mode_map = {"kg": "kg", "vector": "vectordb", "hybrid": "hybrid", "none": "none"}
        response = run_pipeline(query=query, test_mode=mode_map.get(mode, "hybrid"))

        print(f"\nRaw pipeline answer: {response.answer}")

        raw_results = pipeline_fn(query, mode)
        metrics = evaluator.evaluate(query=query, results=raw_results, mode=mode)
        evaluator.print_metrics(metrics)

    else:
        print("Invalid choice. Run again and enter 1, 2, or 3.")
