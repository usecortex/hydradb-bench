# HydraDB DeepEval Benchmark

A practical benchmark framework for evaluating retrieval + answer quality across:

- **HydraDB** (`full_recall`, `recall_preferences`, `boolean_recall`)
- **Supermemory** (`/v3/search`)
- **DeepEval** metrics for answer/retrieval quality

The benchmark can run either provider independently or run both in one pass and produce a side-by-side comparison report.

## What This Framework Does

For each run, the pipeline is:

1. Load config + validate environment variables
2. Ingest documents (optional)
3. Run retrieval queries for each test sample
4. Generate grounded answers from retrieved context (OpenAI model)
5. Score with DeepEval metrics
6. Save reports (`json`, `html`, optionally `csv`)

## Repository Structure

```text
.
|-- config/benchmark.yaml              # Main benchmark config
|-- data/privacy_qa/                   # Source documents for ingestion
|-- data/privacy_qa.json               # Evaluation dataset (Q/A + reference contexts)
|-- reports/                           # Generated reports
|-- synthetic_data/                    # DeepEval synthesizer outputs
|-- run_benchmark.py                   # Main CLI runner
|-- generate_test_data.py              # Build synthetic Q/A dataset from docs
|-- json_to_csv.py                     # Convert report JSON to CSV
`-- src/hydradb_deepeval/              # Core clients, evaluator, reporting, models
```

## Requirements

- Python **3.10+**
- API keys:
  - `HYDRADB_API_KEY` (required by current config loader)
  - `OPENAI_API_KEY` (required for answer generation + DeepEval model usage)
  - `SUPERMEMORY_API_KEY` (required when `supermemory:` section exists in config)

## Installation

```bash
python -m venv .venv
```

Activate venv:

- Windows PowerShell:
  - `.venv\Scripts\Activate.ps1`
- macOS/Linux:
  - `source .venv/bin/activate`

Install dependencies:

```bash
pip install -U pip
pip install -e .
```

Optional (for accurate context token counting):

```bash
pip install tiktoken
```

## Environment Setup

Create `.env` from example:

```bash
cp .env.example .env
```

Populate keys:

```env
HYDRADB_API_KEY=...
OPENAI_API_KEY=...
SUPERMEMORY_API_KEY=...  # needed when supermemory: exists in benchmark.yaml
```

Important behavior from current code:

- If `supermemory:` exists in `config/benchmark.yaml`, the loader expects `SUPERMEMORY_API_KEY` even if you run `--provider hydradb`.
- If you want HydraDB-only without Supermemory key, remove/comment out `supermemory:` in config.

## Configuration (`config/benchmark.yaml`)

Main sections:

- `benchmark`: display name used in reports
- `hydradb`: endpoint + tenant/sub-tenant and ingestion polling settings
- `ingestion`: documents directory, file extensions, upload concurrency/delay
- `evaluation`: dataset path + search endpoint and retrieval parameters
- `deepeval`: metric model, metrics list, thresholds, timeouts, answer generation model
- `reporting`: output directory + formats (`json`, `html`, `csv`)
- `supermemory` (optional): container/search/ingestion behavior for Supermemory

Environment interpolation is supported in YAML via `${ENV_VAR}` syntax.

## How To Write `config/benchmark.yaml`

Create the file at `config/benchmark.yaml` and start from this template:

```yaml
benchmark:
  name: "My Retrieval Benchmark"

hydradb:
  base_url: "https://api.hydradb.com"
  tenant_id: "my-tenant"
  sub_tenant_id: "my-sub-tenant"
  timeout_seconds: 30
  polling_interval_seconds: 10
  max_polling_attempts: 60
  create_tenant_on_start: false

ingestion:
  documents_dir: "./data/privacy_qa"
  file_extensions: [".txt", ".pdf", ".md", ".docx"]
  verify_before_querying: true
  upload_delay_seconds: 1
  upload_concurrency: 1

evaluation:
  test_dataset_path: "./data/privacy_qa.json"
  search_endpoint: "full_recall"      # full_recall | recall_preferences | boolean_recall
  max_results: 10
  concurrent_requests: 3
  mode: "thinking"                    # fast | thinking
  alpha: 0.8
  graph_context: true
  recency_bias: 0.0
  boolean_operator: "or"              # or | and | phrase
  boolean_search_mode: "sources"      # sources | memories

deepeval:
  model: "gpt-5.4"
  threshold: 0.5
  include_reason: true
  eval_concurrency: 5
  metric_timeout_seconds: 60
  generator_model: "gpt-5.4"
  generator_temperature: 0.0
  generator_max_tokens: 1024
  metrics:
    - answer_accuracy
    - contextual_precision
    - contextual_recall
    - contextual_relevancy

reporting:
  output_dir: "./reports"
  formats: ["json", "html", "csv"]
  include_per_sample: true
```

### If you want HydraDB-only config

Do not include `supermemory:` section at all.

```yaml
benchmark:
  name: "HydraDB Only Benchmark"

hydradb:
  base_url: "https://api.hydradb.com"
  tenant_id: "my-tenant"
  sub_tenant_id: "privacyqa"
  timeout_seconds: 30
  polling_interval_seconds: 10
  max_polling_attempts: 60
  create_tenant_on_start: false

ingestion:
  documents_dir: "./data/privacy_qa"
  file_extensions: [".txt", ".pdf", ".md", ".docx"]
  verify_before_querying: true
  upload_delay_seconds: 1
  upload_concurrency: 1

evaluation:
  test_dataset_path: "./data/privacy_qa.json"
  search_endpoint: "full_recall"
  max_results: 10
  concurrent_requests: 3
  mode: "thinking"
  alpha: 0.8
  graph_context: true
  boolean_search_mode: "sources"

deepeval:
  model: "gpt-5.4"
  threshold: 0.5
  include_reason: true
  eval_concurrency: 5
  metric_timeout_seconds: 60
  generator_model: "gpt-5.4"
  generator_temperature: 0.0
  generator_max_tokens: 1024
  metrics:
    - answer_accuracy
    - contextual_precision
    - contextual_recall
    - contextual_relevancy

reporting:
  output_dir: "./reports"
  formats: ["json", "html", "csv"]
  include_per_sample: true
```

### If you want HydraDB + Supermemory config

Add `supermemory:` section:

```yaml
supermemory:
  base_url: "https://api.supermemory.ai"
  container_tag: "privacyqa"
  timeout_seconds: 30
  verify_before_querying: true
  polling_interval_seconds: 5
  max_polling_attempts: 60
  reset_on_start: false
  search_mode: "hybrid"      # hybrid | memories
  rerank: true
  threshold: 0.0
  limit: 20
```

### Field guide (what to edit first)

- `hydradb.tenant_id`: your HydraDB tenant namespace
- `hydradb.sub_tenant_id`: optional sub-namespace (kept as `""` if unused)
- `ingestion.documents_dir`: folder containing docs to upload
- `evaluation.test_dataset_path`: JSON file with benchmark questions
- `evaluation.search_endpoint`: HydraDB endpoint under test
- `deepeval.model`: model used by DeepEval metrics
- `deepeval.generator_model`: model used to generate answer from retrieved context
- `reporting.formats`: output files to generate
- `supermemory.container_tag`: namespace for Supermemory docs (when enabled)

### Environment variable references inside YAML

You can inject env vars with `${...}`:

```yaml
hydradb:
  tenant_id: "${HYDRADB_TENANT_ID}"
  sub_tenant_id: "${HYDRADB_SUB_TENANT_ID}"
```

If an env var is missing, config loading fails fast with an explicit error.

## Input Data Format

The benchmark dataset (`evaluation.test_dataset_path`) must be a JSON array with objects like:

```json
{
  "id": "sample-1",
  "question": "What ...?",
  "reference_answer": "Expected answer",
  "reference_contexts": ["Relevant context chunk 1", "Relevant context chunk 2"]
}
```

Documents for ingestion are read from `ingestion.documents_dir` and filtered by `ingestion.file_extensions`.

## Run Commands

From project root:

```bash
python run_benchmark.py
```

Common variants:

```bash
# HydraDB only (default)
python run_benchmark.py --provider hydradb

# Supermemory only
python run_benchmark.py --provider supermemory

# Both providers + comparison report
python run_benchmark.py --provider both

# Skip ingestion (query/eval only)
python run_benchmark.py --skip-ingestion

# Ingest then exit
python run_benchmark.py --ingest-only

# Delete + recreate HydraDB tenant before ingestion
python run_benchmark.py --reset-tenant

# Run first N samples only
python run_benchmark.py --limit 20

# Verbose per-sample query logs
python run_benchmark.py --verbose

# Custom config path
python run_benchmark.py --config config/benchmark.yaml
```

### HydraDB command combos

```bash
# Full HydraDB run (ingest + query + eval)
python run_benchmark.py --provider hydradb

# HydraDB quick smoke test (first 10 samples)
python run_benchmark.py --provider hydradb --limit 10

# HydraDB query/eval only (reuse existing indexed docs)
python run_benchmark.py --provider hydradb --skip-ingestion

# HydraDB query/eval only + verbose logging + small subset
python run_benchmark.py --provider hydradb --skip-ingestion --limit 20 --verbose

# HydraDB ingest only (no query/eval)
python run_benchmark.py --provider hydradb --ingest-only

# HydraDB hard refresh: delete tenant, ingest, then run benchmark
python run_benchmark.py --provider hydradb --reset-tenant

# HydraDB with custom config file
python run_benchmark.py --provider hydradb --config config/benchmark.yaml
```

### Supermemory command combos

```bash
# Full Supermemory run (ingest + query + eval)
python run_benchmark.py --provider supermemory

# Supermemory quick smoke test (first 10 samples)
python run_benchmark.py --provider supermemory --limit 10

# Supermemory query/eval only (reuse existing container docs)
python run_benchmark.py --provider supermemory --skip-ingestion

# Supermemory query/eval only + verbose logging + subset
python run_benchmark.py --provider supermemory --skip-ingestion --limit 20 --verbose

# Supermemory ingest only (no query/eval)
python run_benchmark.py --provider supermemory --ingest-only

# Supermemory with custom config file
python run_benchmark.py --provider supermemory --config config/benchmark.yaml
```

### Both providers command combos

```bash
# Full compare run: HydraDB + Supermemory + comparison report
python run_benchmark.py --provider both

# Quick compare run (first 10 samples)
python run_benchmark.py --provider both --limit 10

# Compare query/eval only (skip ingestion for both providers)
python run_benchmark.py --provider both --skip-ingestion

# Compare query/eval only + verbose logs + subset
python run_benchmark.py --provider both --skip-ingestion --limit 20 --verbose

# Ingest into both providers then exit
python run_benchmark.py --provider both --ingest-only

# Reset HydraDB tenant, keep Supermemory as configured, then compare
python run_benchmark.py --provider both --reset-tenant

# Compare using custom config
python run_benchmark.py --provider both --config config/benchmark.yaml
```

## Output Reports

Saved into `reporting.output_dir` (default: `./reports`):

- `*.json`: machine-readable full run output (aggregate + optional per-sample)
- `*.html`: interactive human-readable run report
- `*.csv`: flattened per-sample rows (when `csv` is enabled and `include_per_sample: true`)

When running `--provider both`, an additional comparison file is generated:

- `<run_id>_comparison.html`

### JSON Report Shape (high level)

Includes:

- run metadata (`run_id`, `timestamp`, `name`)
- `aggregate_scores` by metric
- `per_sample` details (scores, reasons, answer, context, latency)
- latency percentiles/statistics
- context token statistics
- error count

## Convert JSON Report to CSV

If you already have a JSON report and need CSV:

```bash
python json_to_csv.py reports/<report>.json
python json_to_csv.py reports/<report>.json --output reports/<report>.csv
```

## Generate Synthetic Test Data

Script:

```bash
python generate_test_data.py
```

What it does:

- Uses DeepEval synthesizer over selected docs in `data/privacy_qa/`
- Saves raw synthesized goldens to `synthetic_data/<timestamp>.json`
- Writes benchmark-ready dataset to `data/privacy_qa.json`

Note: the script includes a patch/workaround for DeepEval synthesis cost handling with `gpt-5.4`.

## Supported Metrics

Configured via `deepeval.metrics`.

Built-in names supported by this code:

- `answer_accuracy` (custom GEval, strict 0/1 scoring)
- `answer_relevancy`
- `faithfulness`
- `contextual_precision`
- `contextual_recall`
- `contextual_relevancy`
- `hallucination`
- `bias`
- `toxicity`
- `summarization`

## Troubleshooting

- `HYDRADB_API_KEY environment variable is not set`
  - Add `HYDRADB_API_KEY` in `.env` (or shell env).

- `OPENAI_API_KEY environment variable is not set`
  - Add `OPENAI_API_KEY`; answer generation and DeepEval need it.

- `supermemory section is present ... SUPERMEMORY_API_KEY not set`
  - Add the key, or remove/comment `supermemory:` section for HydraDB-only runs.

- `No matching files found in ...`
  - Verify `ingestion.documents_dir` and `file_extensions`.

- Frequent metric timeouts/errors
  - Reduce `deepeval.eval_concurrency`
  - Increase `deepeval.metric_timeout_seconds`
  - Consider a faster eval model

- Retrieval seems too narrow
  - Increase `evaluation.max_results` (HydraDB)
  - Increase `supermemory.limit` (Supermemory)

## Security Notes

- Keep `.env` out of version control.
- Rotate keys immediately if they are ever exposed.
