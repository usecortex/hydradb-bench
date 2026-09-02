import os
import json
import time
import argparse
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("GENERATION_MODEL", "llama-3.3-70b-versatile")

if not api_key:
    raise ValueError("Missing API key. Set GROQ_API_KEY or OPENAI_API_KEY in .env")

client = OpenAI(api_key=api_key, base_url=base_url)


def generate_multihop_scenario(domain: str, max_retries: int = 3) -> Optional[dict]:
    prompt = f"""Generate a realistic multi-hop reasoning scenario for a Graph RAG benchmark in the domain of '{domain}'.
The scenario must require at least 2 logical hops to answer.

Output ONLY a valid JSON object with the following exact schema:
{{
  "id": "unique-uuid-string",
  "question": "The multi-hop question to be asked",
  "reference_answer": "The final correct answer",
  "intermediate_steps": ["Fact 1 required to bridge the gap", "Fact 2 required to bridge the gap"],
  "reference_contexts": ["Document chunk 1 containing Fact 1", "Document chunk 2 containing Fact 2", "Document chunk 3 containing the final answer"]
}}
"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            content = response.choices[0].message.content
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            print(f"  [Attempt {attempt+1}] JSON parse error: {e}")
        except Exception as e:
            print(f"  [Attempt {attempt+1}] API error: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"  Retrying in {wait_time}s...")
                time.sleep(wait_time)
    
    print(f"  FAILED after {max_retries} attempts. Skipping this sample.")
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate multi-hop benchmark dataset")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--domain", type=str, default="university research")
    parser.add_argument("--output", type=str, default="data/multihop_qa.json")
    args = parser.parse_args()

    print(f"Generating {args.num_samples} samples for domain: '{args.domain}'...")
    dataset = []
    
    for i in range(args.num_samples):
        print(f"Sample {i+1}/{args.num_samples}...")
        scenario = generate_multihop_scenario(args.domain)
        if scenario:
            dataset.append(scenario)
        # Small delay to avoid rate limits even on success
        time.sleep(0.5)
            
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"\nSuccessfully generated {len(dataset)}/{args.num_samples} samples.")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()