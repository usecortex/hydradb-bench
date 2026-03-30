# Skill: Generate Synthetic Test Data

## Description
Generate synthetic Q/A evaluation datasets from source documents using DeepEval's Synthesizer.

## When to Use
When the user needs to create or regenerate test data, build a benchmark dataset from new documents, or expand the existing evaluation set.

## Prerequisites
- `OPENAI_API_KEY` in `.env` (DeepEval synthesizer uses OpenAI models)
- Source documents in `data/privacy_qa/` (or configured directory) with `.txt` extensions
- DeepEval package installed (`pip install -e .`)

## How It Works

1. Reads `.txt` files from `data/privacy_qa/`
2. Uses DeepEval's `Synthesizer` with 4 evolution types:
   - **Reasoning** — Questions requiring multi-step logic
   - **Multicontext** — Questions needing multiple context chunks
   - **Concretizing** — Questions about specific details
   - **Constrained** — Questions with constraints/conditions
3. Generates 2 golden samples per context chunk
4. Filters with 0.7 synthetic quality threshold
5. Saves raw output to `synthetic_data/<timestamp>.json`
6. Writes benchmark-ready dataset to `data/privacy_qa.json`

## Run Command
```bash
python generate_test_data.py
```

## Output Format
`data/privacy_qa.json` — JSON array of objects:
```json
[
  {
    "id": "uuid-string",
    "question": "What is the data retention policy?",
    "reference_answer": "The policy states that...",
    "reference_contexts": ["Context chunk 1", "Context chunk 2"]
  }
]
```

## Customizing Generation
Edit `generate_test_data.py` to change:
- **Number of evolutions**: Modify the `evolutions` dict (keys are Evolution enum members)
- **Goldens per context**: Change `max_goldens_per_context` parameter
- **Quality threshold**: Change `FILTER_THRESHOLD` (currently 0.7)
- **Source directory**: Change the glob path for document loading
- **Models**: Change the critic/filtration model names

## Notes
- The script patches DeepEval's Synthesizer to handle `None` synthesis cost (workaround for models not in DeepEval's pricing table)
- Context construction uses `gpt-4o` critic because `gpt-5.4` can return unparseable scores
- Raw synthesizer output is saved separately in `synthetic_data/` for debugging
- Generation can be slow and expensive depending on document count and model costs
