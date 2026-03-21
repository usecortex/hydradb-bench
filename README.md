# HydraDB Benchmark Framework

A complete benchmark framework that evaluates [HydraDB](https://api.hydradb.com)'s RAG quality using [RAGAS](https://docs.ragas.io) metrics.

## Features

- **End-to-end pipeline**: upload documents → query HydraDB → evaluate with RAGAS → generate reports
- **30+ RAGAS metrics**: Faithfulness, Context Recall, Factual Correctness, Semantic Similarity, and more
- **AspectCritic**: define custom evaluation criteria (e.g. `numerical_accuracy`, `source_grounding`)
- **Multi-turn evaluation**: conversation-level metrics via RAGAS MultiTurnSample
- **HuggingFace integration**: load Q&A datasets or document corpora directly from HuggingFace Hub
- **Multiple LLM providers**: OpenAI or OpenRouter (Claude, Llama, Gemini, Mistral, etc.)
- **Multiple embedding providers**: OpenAI, OpenRouter, local HuggingFace (sentence-transformers), Ollama
- **Throttled ingestion**: configurable delay between uploads to avoid overwhelming the API
- **Tenant management**: create, reset (`--reset-tenant`), and delete tenants via CLI
- **Corpus cleaning**: `scripts/clean_corpus.py` fixes PDF extraction artefacts in text files
- **3 output formats**: JSON, CSV, and color-coded HTML dashboard
- **Async**: concurrent queries with configurable parallelism
- **Configurable**: YAML config + `.env` for credentials

---

## Quick Start

### 1. Install

```bash
pip install -e .
# or
pip install -e ".[dev]"   # includes pytest
```

### 2. Configure credentials

```bash
cp .env.example .env
```

Edit `.env`:

```env
HYDRADB_API_KEY=your_hydradb_bearer_token
HYDRADB_BASE_URL=https://api.hydradb.com
HYDRADB_TENANT_ID=my-company
HYDRADB_SUB_TENANT_ID=bench-run-001

# LLM judge (required for most RAGAS metrics)
OPENAI_API_KEY=sk-...

# Optional: use OpenRouter as LLM judge instead of OpenAI
OPENROUTER_API_KEY=sk-or-...
```

### 3. Review config

Edit [`config/benchmark.yaml`](config/benchmark.yaml) to adjust metrics, search endpoint, LLM provider, embedding provider, and more.

### 4. Run

```bash
python run_benchmark.py
```

On subsequent runs (documents already indexed):

```bash
python run_benchmark.py --skip-ingestion
```

Reports appear in `./reports/` as `bench_<run_id>_<timestamp>.[json|csv|html]`.

---

## Command-Line Options

| Flag | Description |
|---|---|
| `--config PATH` | Path to config YAML (default: `config/benchmark.yaml`) |
| `--skip-ingestion` | Skip document upload — use when docs are already indexed |
| `--reset-tenant` | **Delete** all existing tenant data then re-ingest from scratch |
| `--generate-testset` | Auto-generate Q&A pairs from documents using RAGAS TestsetGenerator |
| `--multi-turn` | Also run multi-turn conversation evaluation |
| `--save-prompts` | Dump metric prompt instructions to `reports/prompts/` |
| `--run-id ID` | Custom run identifier (auto-generated if not set) |
| `--hf-dataset REPO_ID` | Load a HuggingFace dataset instead of local files |
| `--hf-mode qa\|corpus` | HuggingFace dataset mode |
| `--print-hf-info REPO_ID` | Preview a HuggingFace dataset schema and exit |
| `--extract-qa-corpus REPO_ID DIR` | Extract context passages from a HF Q&A dataset as `.txt` files |
| `--context-column COL` | Column name for context passages (used with `--extract-qa-corpus`) |
| `--group-by COL` | Group passages by column when extracting corpus (e.g. `doc_name`) |
| `--hf-split SPLIT` | HuggingFace dataset split to use |
| `-v` / `--verbose` | Enable debug logging |

---

## LLM & Embedding Providers

Configure in `config/benchmark.yaml` under the `ragas:` section.

### LLM Judge

```yaml
ragas:
  llm_provider: "openai"        # "openai" | "openrouter"
  llm_model: "gpt-4o-mini"      # any model name valid for the provider
```

For OpenRouter, `llm_model` uses `provider/model` format:
```yaml
  llm_provider: "openrouter"
  llm_model: "anthropic/claude-3.5-sonnet"
  # also: "meta-llama/llama-3.3-70b-instruct", "google/gemini-flash-1.5"
```

### Embeddings

```yaml
ragas:
  embeddings_provider: "openai"                    # OpenAI API
  embeddings_model: "text-embedding-3-small"
```

```yaml
  embeddings_provider: "openrouter"                # OpenRouter API
  embeddings_model: "openai/text-embedding-3-small"
```

```yaml
  embeddings_provider: "huggingface"               # local, no API key needed
  embeddings_model: "BAAI/bge-small-en-v1.5"
  # pip install sentence-transformers
```

```yaml
  embeddings_provider: "ollama"                    # local Ollama server
  embeddings_model: "nomic-embed-text"
  embeddings_base_url: "http://localhost:11434"    # optional, this is the default
  # pip install langchain-ollama
```

---

## HuggingFace Datasets

### Run a benchmark from a HF Q&A dataset

```bash
python run_benchmark.py --hf-dataset explodinggradients/amnesty_qa
```

Or configure in YAML:

```yaml
hf_dataset:
  enabled: true
  mode: "qa"
  repo_id: "PatronusAI/financebench"
  split: "train"
  max_samples: 30
  column_map:
    reference_contexts: "evidence"
    id: "financebench_id"
  save_qa_path: "./data/financebench_samples.json"
```

### Extract corpus from a HF dataset for ingestion

```bash
# One file per row
python run_benchmark.py --extract-qa-corpus explodinggradients/amnesty_qa ./data/corpus

# Group passages by source document (e.g. FinanceBench)
python run_benchmark.py --extract-qa-corpus PatronusAI/financebench ./data/corpus \
  --group-by doc_name --context-column evidence
```

### Inspect a dataset before benchmarking

```bash
python run_benchmark.py --print-hf-info PatronusAI/financebench
```

---

## Corpus Cleaning

PDF-extracted text files often contain broken table structure (numbers on separate lines, orphan `$` signs, split words). Use the included cleaning script before ingestion:

```bash
python scripts/clean_corpus.py ./data/financebench_corpus
```

**What it fixes:**
- Orphan `$` signs: `$\n34,229` → `$34,229`
- Broken words: `Cash Flow s` → `Cash Flows`
- Label + values joined: `Net sales\n34,229\n35,355` → `Net sales: 34,229 | 35,355`
- Whitespace-only lines removed
- Multiple blank lines collapsed

Typical result: 50–70% line reduction with all financial data preserved.

Options:
```bash
python scripts/clean_corpus.py ./data/corpus --dry-run          # preview only
python scripts/clean_corpus.py ./data/corpus --output ./clean   # write to new dir
```

---

## Tenant Management

### Reset and re-ingest from scratch

```bash
python run_benchmark.py --config config/financebench.yaml --reset-tenant
```

This **permanently deletes** all data for `tenant_id` in HydraDB, recreates the tenant, then re-ingests all documents. Use this when you want a clean slate (e.g. after cleaning the corpus).

> **Warning**: `--reset-tenant` is irreversible. All sub-tenants under the tenant are wiped.

### Ingest settings

```yaml
ingestion:
  documents_dir: "./data/financebench_corpus"
  file_extensions: [".txt"]
  verify_before_querying: true
  upload_delay_seconds: 0.5     # pause between uploads (0.0 to disable)
```

---

## RAGAS Metrics

### Standard metrics (configured in `evaluation.metrics`)

| Metric | What It Measures |
|---|---|
| `faithfulness` | Are all answer claims grounded in retrieved context? |
| `response_relevancy` | Is the answer relevant to the question? |
| `context_precision` | Are retrieved chunks ranked well? |
| `context_recall` | Did retrieval cover what the reference answer needs? |
| `factual_correctness` | Does the answer match the reference factually? |
| `semantic_similarity` | Embedding-based similarity to reference answer |
| `answer_correctness` | Combined factual + semantic correctness |
| `non_llm_context_recall` | String-overlap context recall (no LLM call) |
| `exact_match` | Exact string match (fully offline) |
| `bleu_score` | BLEU overlap score (fully offline) |
| `rouge_score` | ROUGE overlap score (fully offline) |
| *(+ 20 more)* | See `config/benchmark.yaml` for full list |

### AspectCritic (custom criteria)

```yaml
evaluation:
  aspect_critics:
    - name: "numerical_accuracy"
      definition: "Does the answer contain accurate numerical values?"
      strictness: 2
    - name: "source_grounding"
      definition: "Is the answer grounded in cited document evidence?"
      strictness: 1
```

---

## Auto-Generate Q&A from Your Own Documents

If you only have a corpus (no pre-made Q&A), RAGAS can generate questions + reference answers + reference contexts automatically:

```bash
# 1. Put your documents in data/documents/ (txt, pdf, md)

# 2. Generate Q&A pairs and run the full benchmark in one command
python run_benchmark.py --generate-testset

# 3. On subsequent runs, reuse the generated testset
python run_benchmark.py --skip-ingestion
```

Or enable in YAML:
```yaml
testset_generation:
  enabled: true
  output_path: "./data/test_dataset.json"
  testset_size: 15
  query_distribution:
    simple: 0.5        # single-passage questions
    multi_context: 0.3 # questions needing multiple passages
    reasoning: 0.2     # inference questions
```

Generated testset is saved to `test_dataset.json` in the same format as HF Q&A datasets. All metrics including `context_recall` will work because RAGAS knows which passages each question was derived from.

---

## Common Command Combinations

```bash
# Basic run (ingest + evaluate)
python run_benchmark.py

# Skip ingestion — docs already indexed
python run_benchmark.py --skip-ingestion

# Use a specific config file
python run_benchmark.py --config config/financebench.yaml

# Custom run ID (appears in report filename)
python run_benchmark.py --config config/financebench.yaml --run-id my-run-01

# Reset tenant data and re-ingest from scratch
python run_benchmark.py --config config/financebench.yaml --reset-tenant

# Ingest fresh + auto-generate Q&A + evaluate
python run_benchmark.py --generate-testset

# Skip ingestion + auto-generate Q&A + evaluate
python run_benchmark.py --skip-ingestion --generate-testset

# Full verbose run with custom ID
python run_benchmark.py --config config/legalbench.yaml --skip-ingestion --run-id legal-v2 -v

# Multi-turn evaluation alongside single-turn
python run_benchmark.py --multi-turn

# Inspect a HF dataset schema before using it
python run_benchmark.py --print-hf-info PatronusAI/financebench

# Save metric prompt instructions for debugging
python run_benchmark.py --save-prompts --skip-ingestion
```

---

## FinanceBench Config

A ready-made config for [PatronusAI/financebench](https://huggingface.co/datasets/PatronusAI/financebench) — 150 Q&A pairs from real SEC filings:

```bash
# First time: extract corpus (one-time)
python run_benchmark.py --extract-qa-corpus PatronusAI/financebench \
  ./data/financebench_corpus --group-by doc_name

# Clean the extracted text (fixes PDF table structure)
python scripts/clean_corpus.py ./data/financebench_corpus

# Run benchmark (ingest + evaluate)
python run_benchmark.py --config config/financebench.yaml

# Re-run with fresh data (after cleaning corpus)
python run_benchmark.py --config config/financebench.yaml --reset-tenant

# Skip ingestion on subsequent runs
python run_benchmark.py --config config/financebench.yaml --skip-ingestion
```

---

## LegalBench Config

A ready-made config for [isaacus/legal-rag-bench](https://huggingface.co/datasets/isaacus/legal-rag-bench) — 100 Q&A pairs grounded in Australian jury instructions:

```bash
# Corpus is already prepared in data/legalbench_corpus/ (112 chapter-level .txt files)

# Run benchmark (ingest + evaluate 30 samples)
python run_benchmark.py --config config/legalbench.yaml

# Skip ingestion on subsequent runs
python run_benchmark.py --config config/legalbench.yaml --skip-ingestion

# Reset tenant and re-ingest
python run_benchmark.py --config config/legalbench.yaml --reset-tenant

# Run full 100-sample evaluation (edit max_samples: 100 in legalbench.yaml first)
python run_benchmark.py --config config/legalbench.yaml --skip-ingestion --run-id legal-full
```

> **Note**: `context_recall` is excluded from legalbench config — the dataset links answers to passages via `relevant_passage_id` (not inline context text), so reference contexts are unavailable per sample.

---

## Project Structure

```
hydradb-bench/
├── config/
│   ├── benchmark.yaml           # Main configuration (all options documented)
│   ├── financebench.yaml        # FinanceBench-specific config
│   └── legalbench.yaml          # LegalBench-specific config
├── data/
│   ├── documents/               # Default knowledge documents directory
│   ├── financebench_corpus/     # FinanceBench extracted passages (.txt)
│   ├── legalbench_corpus/       # LegalBench corpus — 112 chapter-level .txt files
│   ├── financebench_samples.json
│   └── legalbench_samples.json
├── scripts/
│   └── clean_corpus.py          # PDF extraction artefact cleaner
├── src/
│   └── hydradb_bench/
│       ├── client.py            # HydraDB API client (async, httpx)
│       ├── config.py            # Config loader (YAML + .env)
│       ├── evaluator.py         # RAGAS evaluation wrapper (multi-provider)
│       ├── hf_loader.py         # HuggingFace dataset loader
│       ├── ingestion.py         # Document upload & indexing orchestrator
│       ├── models.py            # Pydantic data models
│       ├── multi_turn.py        # Multi-turn conversation runner
│       ├── reporter.py          # JSON/CSV/HTML report generator
│       ├── runner.py            # Concurrent query runner
│       └── testset_generator.py # RAGAS TestsetGenerator wrapper
├── tests/
│   ├── test_hf_loader.py        # Unit tests for HF loader
│   └── test_evaluator.py        # Unit tests for RAGAS evaluator
├── reports/                     # Output directory
├── run_benchmark.py             # Single-command entrypoint
├── .env.example
└── pyproject.toml
```

---

## Running Tests

```bash
pytest tests/ -v

# Also run HuggingFace integration tests (downloads real datasets)
HF_INTEGRATION_TESTS=1 pytest tests/ -v
```

Tests cover column detection, string normalisation, dataset building, metric registry, and report generation — all runnable without a live API or OpenAI key.

---

## Cost Estimates

With `gpt-4o-mini` as the judge:

| Samples | Metrics | Approx. cost |
|---|---|---|
| 15 | 5 standard | ~$0.02–0.05 |
| 30 | 5 standard + 3 aspect critics | ~$0.10–0.20 |
| 150 (full FinanceBench) | 5 standard + 3 aspect critics | ~$0.50–1.00 |

Cost tracking is reported at the end of each run. Disable with `cost_tracking.enabled: false`.
