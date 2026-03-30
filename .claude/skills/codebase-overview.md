# Skill: Codebase Overview

## Description
Understand the architecture, data flow, and key components of the HydraDB DeepEval Benchmark framework.

## When to Use
When an AI agent or developer needs to understand the codebase before making changes, or when the user asks "how does X work?"

## Architecture

```
run_benchmark.py (CLI entry point)
    |
    ├── src/hydradb_deepeval/config.py        -- Load YAML + env vars → BenchmarkConfig
    ├── src/hydradb_deepeval/models.py         -- All Pydantic data models
    ├── src/hydradb_deepeval/client.py         -- HydraDB async HTTP client (all API endpoints)
    ├── src/hydradb_deepeval/supermemory_client.py -- Supermemory async HTTP client + ingester
    ├── src/hydradb_deepeval/ingestion.py      -- HydraDB document upload + polling
    ├── src/hydradb_deepeval/context_builder.py -- Format HydraDB full_recall response → text
    ├── src/hydradb_deepeval/answer_generator.py -- OpenAI LLM answer generation
    ├── src/hydradb_deepeval/evaluator.py      -- DeepEval metric evaluation engine
    └── src/hydradb_deepeval/reporter.py       -- JSON/HTML/CSV report generation

generate_test_data.py  -- Synthetic Q/A dataset generation via DeepEval Synthesizer
json_to_csv.py         -- Convert JSON reports to CSV
```

## Data Flow

1. **Config Loading** (`config.py`)
   - Reads `config/benchmark.yaml`
   - Interpolates `${ENV_VAR}` references
   - Validates API keys from `.env`
   - Returns `BenchmarkConfig` (Pydantic model)

2. **Ingestion** (`ingestion.py` / `supermemory_client.py`)
   - Reads files from `ingestion.documents_dir`
   - Uploads to HydraDB via `/ingestion/upload_knowledge` or Supermemory via `/v3/documents/file`
   - Polls until processing completes
   - Returns `(indexed_count, failed_count, elapsed_seconds)`

3. **Querying** (`run_benchmark.py` + `client.py` / `supermemory_client.py`)
   - Loads test samples from JSON dataset
   - For each sample, calls the configured search endpoint
   - HydraDB: `full_recall`, `recall_preferences`, or `boolean_recall`
   - Supermemory: `/v3/search` with hybrid/memories mode
   - Builds context string from retrieved chunks
   - Generates grounded answer via OpenAI (`answer_generator.py`)
   - Returns `QueryResult` per sample

4. **Evaluation** (`evaluator.py`)
   - Creates `LLMTestCase` per sample (input, actual_output, retrieval_context, expected_output)
   - Runs configured DeepEval metrics in parallel with per-metric timeouts
   - Returns aggregate scores (mean) and per-sample `SampleScore` objects

5. **Reporting** (`reporter.py`)
   - Generates JSON with full results + latency/token stats
   - Generates interactive HTML with score bars, sortable tables, expandable details
   - Optionally generates CSV for spreadsheet analysis
   - For dual-provider runs, generates comparison HTML

## Key Models (`models.py`)

| Model | Purpose |
|-------|---------|
| `BenchmarkConfig` | Master config containing all sub-configs |
| `HydraConfig` | HydraDB connection settings |
| `SupermemoryConfig` | Supermemory connection settings |
| `IngestionConfig` | Document upload settings |
| `EvaluationConfig` | Query execution settings |
| `DeepEvalConfig` | Metric evaluation settings |
| `ReportingConfig` | Report output settings |
| `TestSample` | Single Q/A test case |
| `QueryResult` | Result from querying a provider |
| `SampleScore` | Evaluated scores for one sample |
| `BenchmarkResult` | Final aggregated results for a run |

## HydraDB Client Endpoints (`client.py`)

The `HydraDBClient` is a comprehensive async HTTP client covering:
- **Tenant management**: create, delete, list, status, stats
- **Ingestion**: upload_knowledge, verify_processing, delete_knowledge
- **Memories**: add_memory, delete_memory
- **Recall/Search**: full_recall, recall_preferences, boolean_recall
- **List/Fetch**: list_knowledge, list_memories, get_graph_relations, fetch_content
- **Raw embeddings**: insert, search, filter, delete

## CLI Arguments (`run_benchmark.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config/benchmark.yaml` | Config file path |
| `--provider` | `hydradb` | `hydradb`, `supermemory`, or `both` |
| `--skip-ingestion` | false | Skip document upload |
| `--ingest-only` | false | Upload and exit |
| `--reset-tenant` | false | Delete HydraDB tenant first |
| `--limit` | all | Run first N samples only |
| `--verbose` | false | Extra debug output |

## Dependencies
- `httpx` — Async HTTP client
- `pydantic` — Data validation / config models
- `pyyaml` — YAML config parsing
- `python-dotenv` — `.env` file loading
- `rich` — Terminal UI (progress bars, tables)
- `tiktoken` — Token counting
- `deepeval` — Evaluation metrics
- `openai` — LLM answer generation
