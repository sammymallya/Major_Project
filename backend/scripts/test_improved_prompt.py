"""
Test script for improved prompt generation and answer quality.
"""

from backend.prompt_generator import build_prompt
from backend.test_llm import generate_answer
from backend.kg.types import KgTriple


def test_vector_context_only():
    """Test with vector context only."""
    query = "What is special about Maravanthe Beach?"
    vector_context = "Coastal promenade area popular for evening walks."
    prompt = build_prompt(query, vector_context, None)
    
    print("=== TEST 1: Vector Context Only ===")
    print(f"Query: {query}")
    print(f"Context: {vector_context}")
    print("\nGenerated Answer:")
    answer = generate_answer(prompt)
    print(answer)
    print("\n" + "="*70 + "\n")
    return answer


def test_kg_context_only():
    """Test with KG context only."""
    query = "Which beaches in Mangalore are known for water sports?"
    kg_triples = [
        KgTriple(subject="Panambur Beach", predicate="known_for", object="water sports"),
        KgTriple(subject="Panambur Beach", predicate="location", object="Mangalore"),
    ]
    prompt = build_prompt(query, None, kg_triples)
    
    print("=== TEST 2: KG Context Only ===")
    print(f"Query: {query}")
    print(f"KG Triples: {[(t.subject, t.predicate, t.object) for t in kg_triples]}")
    print("\nGenerated Answer:")
    answer = generate_answer(prompt)
    print(answer)
    print("\n" + "="*70 + "\n")
    return answer


def test_hybrid_context():
    """Test with both vector and KG context."""
    query = "Tell me about famous temples in Mangalore."
    vector_context = "Kadri Manjunath Temple is a historic temple with intricate architecture."
    kg_triples = [
        KgTriple(subject="Kadri Manjunath Temple", predicate="location", object="Mangalore"),
        KgTriple(subject="Kadri Manjunath Temple", predicate="known_for", object="ancient carvings"),
    ]
    prompt = build_prompt(query, vector_context, kg_triples)
    
    print("=== TEST 3: Hybrid Context (Vector + KG) ===")
    print(f"Query: {query}")
    print(f"Vector Context: {vector_context}")
    print(f"KG Triples: {[(t.subject, t.predicate, t.object) for t in kg_triples]}")
    print("\nGenerated Answer:")
    answer = generate_answer(prompt)
    print(answer)
    print("\n" + "="*70 + "\n")
    return answer


if __name__ == "__main__":
    try:
        ans1 = test_vector_context_only()
        ans2 = test_kg_context_only()
        ans3 = test_hybrid_context()
        
        print("\n=== SUMMARY ===")
        print(f"Test 1 answer length: {len(ans1)} chars, {len(ans1.split('.'))} sentences")
        print(f"Test 2 answer length: {len(ans2)} chars, {len(ans2.split('.'))} sentences")
        print(f"Test 3 answer length: {len(ans3)} chars, {len(ans3.split('.'))} sentences")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
