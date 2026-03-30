"""
Final test to verify the optimal configuration works.
Tests Config 3 + Config 2 combination for best results.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-large"
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

test_prompt = """Instruction:
Answer the query using the supplied context below.
Write your answer in 2-4 complete, well-structured sentences that flow naturally.
Start by directly addressing the query with the main information.
Then elaborate with specific details and descriptive context from the provided information.
Construct a coherent narrative response - do not simply repeat the context verbatim.
Use descriptive language and ensure the answer is informative and complete.

Query:
Tell me about beaches in Mangalore

Context:
Vector context:
Uchila Beach is a quiet coastal destination located between Mangalore and Udupi with scenic shoreline.
Coastal promenade area popular for evening walks.
Cliffside walking path overlooking the Arabian Sea.

Answer:
"""

inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=1024)

print("\nTESTING OPTIMAL CONFIGURATION:")
print("="*80)

# Optimal combination: min_length + length_penalty for best elaboration
optimal_config = {
    "max_new_tokens": 150,
    "num_beams": 4,
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 3,
    "length_penalty": 2.0,  # Encourage longer sequences
    "min_length": 50,       # Force minimum output length
    "do_sample": False,
}

print(f"Configuration: {optimal_config}\n")

outputs = model.generate(**inputs, **optimal_config)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

print(f"Answer:\n{answer}\n")
print(f"Length: {len(answer)} chars")
print(f"Sentences: ~{len([s for s in answer.split('.') if s.strip()])}")
