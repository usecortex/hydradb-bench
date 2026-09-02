import os
import json
import argparse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client (supports OPENAI_API_KEY or GROQ_API_KEY via base_url)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

MODEL_NAME = os.getenv("GENERATION_MODEL", "gpt-4o") # Or "llama3-70b-8192" for Groq

def generate_multihop_scenario(domain: str) -> dict:
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
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7
    )
    
    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("Error: LLM did not return valid JSON.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate multi-hop benchmark dataset")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of samples to generate")
    parser.add_argument("--domain", type=str, default="corporate acquisitions and tech startups", help="Domain for synthetic data")
    parser.add_argument("--output", type=str, default="data/multihop_qa.json", help="Output JSON file path")
    args = parser.parse_args()

    print(f"Generating {args.num_samples} multi-hop samples for domain: '{args.domain}'...")
    
    dataset = []
    for i in range(args.num_samples):
        print(f"Generating sample {i+1}/{args.num_samples}...")
        scenario = generate_multihop_scenario(args.domain)
        if scenario:
            dataset.append(scenario)
            
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Successfully generated {len(dataset)} samples. Saved to {args.output}")

if __name__ == "__main__":
    main()