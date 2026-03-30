"""
End-to-end test with the updated configuration.
Tests the full pipeline with the new length_penalty and min_length settings.
"""

import os
import sys
from dotenv import load_dotenv

# Load env
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=env_path, override=False)

from backend.prompt_generator import build_prompt
from backend.test_llm import generate_answer
from backend.kg.types import KgTriple

# Clear the global model cache to reload with new settings
import backend.test_llm.main as llm_main
llm_main._model = None
llm_main._tokenizer = None

print("="*80)
print("FULL PIPELINE TEST WITH OPTIMIZED CONFIGURATION")
print("="*80)

test_cases = [
    {
        "name": "Test 1: Vector Context - Beach Query",
        "query": "Tell me about beaches in Mangalore",
        "vector_context": "Uchila Beach is a quiet coastal destination located between Mangalore and Udupi with scenic shoreline.\nCoastal promenade area popular for evening walks.\nCliffside walking path overlooking the Arabian Sea.",
        "kg_triples": None,
    },
    {
        "name": "Test 2: KG Context - Temple Query",
        "query": "Tell me about famous temples in Mangalore",
        "vector_context": None,
        "kg_triples": [
            KgTriple(subject="Kadri Manjunath Temple", predicate="location", object="Mangalore"),
            KgTriple(subject="Kadri Manjunath Temple", predicate="known_for", object="intricate carvings"),
        ],
    },
    {
        "name": "Test 3: Hybrid - Water Sports Query",
        "query": "Which beaches in Mangalore are known for water sports?",
        "vector_context": "Panambur Beach offers jet skiing and parasailing activities.",
        "kg_triples": [
            KgTriple(subject="Panambur Beach", predicate="location", object="Mangalore"),
            KgTriple(subject="Panambur Beach", predicate="activity", object="water sports"),
        ],
    },
]

results = []

for test in test_cases:
    print(f"\n{'-'*80}")
    print(test["name"])
    print(f"Query: {test['query']}")
    
    # Build prompt
    prompt = build_prompt(test["query"], test["vector_context"], test["kg_triples"])
    
    # Generate answer
    answer = generate_answer(prompt)
    
    # Calculate stats
    char_count = len(answer)
    sentence_count = len([s for s in answer.split('.') if s.strip()])
    
    print(f"\nAnswer ({char_count} chars, ~{sentence_count} sentences):")
    print(answer)
    
    results.append({
        "test": test["name"],
        "answer": answer,
        "chars": char_count,
        "sentences": sentence_count,
    })

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

for r in results:
    status = "✓ GOOD" if r["chars"] > 150 and r["sentences"] > 1 else "✗ SHORT"
    print(f"\n{status} | {r['test']}")
    print(f"  {r['chars']} chars, ~{r['sentences']} sentences")

avg_chars = sum(r["chars"] for r in results) / len(results)
avg_sentences = sum(r["sentences"] for r in results) / len(results)

print(f"\nAverage: {avg_chars:.0f} chars, {avg_sentences:.1f} sentences")
print(f"Target: >150 chars, >2 sentences per answer ✓" if avg_chars > 150 else "Target: >150 chars, >2 sentences per answer ✗")
