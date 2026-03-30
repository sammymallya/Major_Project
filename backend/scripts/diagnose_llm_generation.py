"""
Diagnostic script to investigate LLM answer length issues.
Tests various generation parameters to identify the best configuration.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model setup
MODEL_NAME = "google/flan-t5-large"
logger.info(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Test prompt with context
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

print("="*80)
print("DIAGNOSTIC: Testing LLM Generation Parameters")
print("="*80)

# Tokenize input
inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=1024)
print(f"\nInput prompt tokens: {inputs['input_ids'].shape[1]}")
print(f"Input prompt length (chars): {len(test_prompt)}")

# Test different configurations
configs = [
    {
        "name": "Config 1: Current Modified (do_sample=False)",
        "params": {
            "max_new_tokens": 150,
            "num_beams": 4,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "do_sample": False,
        }
    },
    {
        "name": "Config 2: With length_penalty (more tokens)",
        "params": {
            "max_new_tokens": 150,
            "num_beams": 4,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "length_penalty": 2.0,  # Encourage longer outputs
            "do_sample": False,
        }
    },
    {
        "name": "Config 3: With min_length",
        "params": {
            "max_new_tokens": 150,
            "num_beams": 4,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "min_length": 50,  # Force at least 50 tokens
            "do_sample": False,
        }
    },
    {
        "name": "Config 4: With sampling & temperature",
        "params": {
            "max_new_tokens": 150,
            "repetition_penalty": 1.2,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
        }
    },
    {
        "name": "Config 5: High max_new_tokens (original default)",
        "params": {
            "max_new_tokens": 250,
            "num_beams": 4,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "do_sample": False,
        }
    },
    {
        "name": "Config 6: Early stopping disabled",
        "params": {
            "max_new_tokens": 150,
            "num_beams": 4,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "early_stopping": False,
            "do_sample": False,
        }
    },
]

results = []

for config in configs:
    print(f"\n{'-'*80}")
    print(f"Testing: {config['name']}")
    print(f"Parameters: {config['params']}")
    
    try:
        # Generate
        outputs = model.generate(**inputs, **config['params'])
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Stats
        output_tokens = outputs[0].shape[0] - inputs['input_ids'].shape[1]
        answer_length = len(answer)
        sentence_count = len([s for s in answer.split('.') if s.strip()])
        
        print(f"\nOutput tokens generated: {output_tokens}")
        print(f"Output length (chars): {answer_length}")
        print(f"Approx sentence count: {sentence_count}")
        print(f"\nAnswer:\n{answer}\n")
        
        results.append({
            "config": config['name'],
            "output_tokens": output_tokens,
            "answer_length": answer_length,
            "sentences": sentence_count,
            "answer": answer,
        })
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

for r in results:
    print(f"\n{r['config']}")
    print(f"  Output tokens: {r['output_tokens']}, Chars: {r['answer_length']}, Sentences: ~{r['sentences']}")

print("\n" + "="*80)
print("BEST CONFIGURATION FOR 2-4 SENTENCES:")
print("="*80)

best = max(results, key=lambda x: x['answer_length'])
print(f"\n{best['config']} produced the longest answer ({best['answer_length']} chars, ~{best['sentences']} sentences)")
print(f"\nRecommended config: {best['config']}")
print(f"Full Answer:\n{best['answer']}")
