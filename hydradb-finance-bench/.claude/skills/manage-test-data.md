# Skill: Manage Test Data

## Description
Create, edit, validate, and manage benchmark test datasets and source documents.

## When to Use
When the user wants to add test samples, modify the evaluation dataset, add source documents for ingestion, or validate dataset format.

## Test Dataset Format
Located at `evaluation.test_dataset_path` (default: `data/privacy_qa.json`).

Must be a JSON array of objects:
```json
[
  {
    "id": "unique-sample-id",
    "question": "What is the data retention policy for user accounts?",
    "reference_answer": "User account data is retained for 3 years after account deletion...",
    "reference_contexts": [
      "Section 4.2: Data retention policy states that all user account data...",
      "Appendix B: The retention schedule for personal data includes..."
    ]
  }
]
```

### Required Fields
| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier for the sample |
| `question` | string | The query to send to the retrieval system |
| `reference_answer` | string | Expected/gold answer for accuracy comparison |
| `reference_contexts` | string[] | Reference context chunks for recall/precision metrics |

## Adding Samples Manually
```python
import json, uuid

# Load existing dataset
with open("data/privacy_qa.json") as f:
    dataset = json.load(f)

# Add new sample
dataset.append({
    "id": str(uuid.uuid4()),
    "question": "Your question here?",
    "reference_answer": "The expected answer...",
    "reference_contexts": ["Relevant context chunk 1", "Relevant context chunk 2"]
})

# Save
with open("data/privacy_qa.json", "w") as f:
    json.dump(dataset, f, indent=2)
```

## Validating the Dataset
```python
import json

with open("data/privacy_qa.json") as f:
    data = json.load(f)

required = {"id", "question", "reference_answer", "reference_contexts"}
for i, sample in enumerate(data):
    missing = required - set(sample.keys())
    if missing:
        print(f"Sample {i}: missing {missing}")
    if not isinstance(sample.get("reference_contexts"), list):
        print(f"Sample {i}: reference_contexts must be a list")
    if not sample.get("question", "").strip():
        print(f"Sample {i}: empty question")

print(f"Total samples: {len(data)}")
print(f"Unique IDs: {len(set(s['id'] for s in data))}")
```

## Source Documents for Ingestion
Located in `ingestion.documents_dir` (default: `data/privacy_qa/`).

Supported extensions (configurable): `.txt`, `.pdf`, `.md`, `.docx`

### Adding New Documents
1. Place files in `data/privacy_qa/` (or configured directory)
2. Ensure file extension matches `ingestion.file_extensions` in config
3. Run ingestion: `python run_benchmark.py --ingest-only`

### Generating Synthetic Data from Documents
```bash
python generate_test_data.py
```
This reads `.txt` files from `data/privacy_qa/`, generates Q/A pairs via DeepEval Synthesizer, and writes to `data/privacy_qa.json`.

## Tips
- Keep `reference_contexts` focused — include only the chunks that directly answer the question
- Use unique, descriptive `id` values for easier debugging
- `reference_answer` should be concise and factual — it's compared against the LLM-generated answer
- Run with `--limit 5` after dataset changes to verify everything loads correctly
- Synthetic data generation output is also saved to `synthetic_data/` for debugging
