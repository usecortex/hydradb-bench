# HydraDB Benchmark Framework

A full-featured RAG evaluation framework for HydraDB using [RAGAS](https://docs.ragas.io). Measures retrieval quality, answer accuracy, and context relevance across any document corpus — with support for multi-dataset runs, HuggingFace datasets, multi-turn conversations, cost tracking, and Telegram notifications.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Configuration](#configuration)
- [All Commands](#all-commands)
- [Command Combinations](#command-combinations)
- [Metrics Reference](#metrics-reference)
- [Test Dataset Format](#test-dataset-format)
- [Reports](#reports)
- [Project Structure](#project-structure)

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                     BENCHMARK PIPELINE                          │
│                                                                 │
│  1. INGEST      Upload .txt/.pdf/.md files → HydraDB tenant    │
│       ↓                                                         │
│  2. QUERY       Send each test question → HydraDB              │
│                 (qna endpoint OR full_recall + LLM answer)      │
│       ↓                                                         │
│  3. EVALUATE    RAGAS scores each (question, answer, context)   │
│                 using a judge LLM (OpenAI / OpenRouter)         │
│       ↓                                                         │
│  4. REPORT      JSON + CSV + HTML reports saved to ./reports/   │
│                 (optional: Telegram notification + zip)          │
└─────────────────────────────────────────────────────────────────┘
```

### Two Search Endpoints

| Endpoint | What HydraDB returns | When to use |
|----------|---------------------|-------------|
| `qna` | Answer + retrieved chunks | HydraDB generates the answer |
| `full_recall` | Retrieved chunks only, then optional external LLM generates answer | You want to control answer generation |

### Three Metric Types

| Type | How it scores | Config section |
|------|--------------|----------------|
| **Standard RAGAS** | Built-in metrics (faithfulness, context_recall, etc.) | `evaluation.metrics` |
| **Aspect Critics** | LLM votes pass/fail on a custom criterion | `evaluation.aspect_critics` |
| **Scored Criteria** | LLM gives 0–3 score on a custom rubric (normalized to 0–1) | `evaluation.scored_criteria` |

---

## Installation

```bash
# Clone and install (editable mode — changes to source take effect immediately)
git clone <repo>
cd hydradb-bench
pip install -e .

# Install dev dependencies (for running tests)
pip install -e ".[dev]"
```

---

## Environment Setup

Copy `.env.example` to `.env` and fill in your credentials:

```bash
# HydraDB
HYDRADB_BASE_URL=https://api.hydradb.ai
HYDRADB_API_KEY=your_hydradb_api_key
HYDRADB_TENANT_ID=your_tenant_id

# OpenAI (for RAGAS judge and/or answer generation)
OPENAI_API_KEY=sk-...

# OpenRouter (alternative LLM provider for judge / answer generation)
OPENROUTER_API_KEY=sk-or-...

# Telegram (optional — for report notifications)
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...   # from @BotFather
TELEGRAM_CHAT_ID=123456789             # your chat ID
```

---

## Configuration

All settings live in a YAML file. Pass it with `--config`.

```yaml
benchmark:
  name: "My Benchmark"
  output_dir: "./reports"

# ── HydraDB connection ────────────────────────────────────────────
hydradb:
  base_url: "${HYDRADB_BASE_URL}"
  api_key: "${HYDRADB_API_KEY}"
  tenant_id: "${HYDRADB_TENANT_ID}"
  sub_tenant_id: "my-corpus"
  create_tenant_on_start: true
  timeout_seconds: 60

# ── Document ingestion ────────────────────────────────────────────
ingestion:
  documents_dir: "./data/documents"
  file_extensions: [".txt", ".pdf", ".md"]
  verify_before_querying: true
  upload_delay_seconds: 0.5

# ── Evaluation ────────────────────────────────────────────────────
evaluation:
  test_dataset_path: "./data/test_dataset.json"
  search_endpoint: "full_recall"     # "qna" | "full_recall"
  retrieve_mode: "thinking"          # "fast" | "thinking" (full_recall only)
  max_results: 10                    # chunks to retrieve per query
  concurrent_requests: 5

  metrics:                           # standard RAGAS metrics
    - context_recall
    - context_relevance
    - faithfulness
    - factual_correctness

  answer_generation:                 # only used when search_endpoint = full_recall
    enabled: true
    provider: "openrouter"           # "openai" | "openrouter"
    model: "openai/gpt-4o-mini"
    system_prompt: "Answer using only the provided context."
    max_tokens: 1024
    temperature: 0.0

  aspect_critics:                    # LLM binary pass/fail judges
    - name: "answer_accuracy"
      definition: "Is the answer factually correct given the reference?"
      strictness: 3                  # majority vote over 3 LLM calls

  scored_criteria:                   # LLM 0-3 rubric judges (normalized to 0-1)
    - name: "answer_relevancy"
      definition: |
        3 — directly and specifically answers the question
        2 — mostly on-topic but includes minor tangents
        1 — loosely related, doesn't really answer
        0 — off-topic or ignores the question

# ── RAGAS judge LLM ───────────────────────────────────────────────
ragas:
  llm_provider: "openrouter"        # "openai" | "openrouter"
  llm_model: "openai/gpt-4o-mini"
  embeddings_provider: "openai"     # "openai" | "openrouter" | "huggingface" | "ollama"
  embeddings_model: "text-embedding-3-small"
  temperature: 0.01
  judge_max_tokens: 8192
  max_retries: 3
  timeout: 120
  max_workers: 4
  cost_tracking:
    enabled: true

# ── Multi-dataset mode ────────────────────────────────────────────
# When set, loops through each dataset — overrides sub_tenant_id,
# documents_dir, and test_dataset_path per entry.
datasets:
  - name: "PrivacyQA"
    sub_tenant_id: "legal-privacyqa"
    documents_dir: "./data/privacy_qa"
    test_dataset_path: "./data/legal_privacyqa_samples.json"
  - name: "ContractNLI"
    sub_tenant_id: "legal-contractnli"
    documents_dir: "./data/contractnli"
    test_dataset_path: "./data/legal_contractnli_samples.json"

# ── Reporting ─────────────────────────────────────────────────────
reporting:
  formats: ["json", "csv", "html"]
  include_per_sample_scores: true

# ── Telegram notifications ────────────────────────────────────────
telegram:
  enabled: false
  bot_token: "${TELEGRAM_BOT_TOKEN}"
  chat_id: "${TELEGRAM_CHAT_ID}"
```

### Telegram notifications

When `telegram.enabled: true`, after every run the bot sends:
1. A summary message — scores, latency, cost, sample count
2. A single zip file — JSON + CSV + HTML reports, directly downloadable

**Setup:**
1. Message [@BotFather](https://t.me/botfather) → `/newbot` → copy the token
2. Send `/start` to your new bot
3. Visit `https://api.telegram.org/bot<your_token>/getUpdates` → find `"chat": {"id": 123456789}` → that's your chat ID
4. Add both to `.env`:
```bash
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHAT_ID=123456789
```

---

### Embeddings providers

| Provider | Config value | Requires |
|----------|-------------|---------|
| OpenAI API | `openai` | `OPENAI_API_KEY` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` |
| Local sentence-transformers | `huggingface` | `pip install langchain-huggingface sentence-transformers` |
| Local Ollama server | `ollama` | Ollama running at `localhost:11434` (override with `embeddings_base_url`) |

---

## All Commands

### Core flags

| Flag | Description |
|------|-------------|
| `--config PATH` | Path to YAML config file. Default: `config/benchmark.yaml` |
| `--skip-ingestion` | Skip document upload — use when docs are already indexed in HydraDB |
| `--reset-tenant` | **Destructive.** Delete all existing tenant data then re-ingest from scratch |
| `--generate-testset` | Auto-generate Q&A pairs from your documents using RAGAS before running |
| `--multi-turn` | Also run multi-turn conversation evaluation after single-turn |
| `--save-prompts` | Dump all metric prompt instructions to `reports/prompts/` for inspection |
| `--run-id ID` | Set a custom run ID (auto-generated UUID if not provided) |
| `--limit N` | Only run the first N questions — useful for quick smoke tests |
| `--dataset NAME` | In multi-dataset configs, run only one dataset by name or sub_tenant_id |
| `--resume RUN_ID` | Resume an interrupted run — skips already-completed samples via checkpoints |
| `-v` / `--verbose` | Enable debug logging |

### HuggingFace flags

| Flag | Description |
|------|-------------|
| `--hf-dataset REPO_ID` | Load a HuggingFace dataset as the Q&A test set |
| `--hf-mode qa\|corpus` | `qa` = evaluation pairs, `corpus` = documents to ingest |
| `--hf-split SPLIT` | Dataset split to use (default: auto-detect or `train`) |
| `--print-hf-info REPO_ID` | Preview a HF dataset's schema and first row, then exit |
| `--extract-qa-corpus REPO_ID OUTPUT_DIR` | Extract context passages from a HF Q&A dataset as .txt files |
| `--context-column COL` | Column containing context passages (auto-detected if not set) |
| `--group-by COL` | Group passages by this column when extracting corpus (one file per unique value) |

---

## Command Combinations

### Standard full run

```bash
python run_benchmark.py --config config/legal_benchmark.yaml
```
Ingests documents → queries HydraDB → evaluates with RAGAS → saves reports.

---

### Skip ingestion (docs already indexed)

```bash
python run_benchmark.py --config config/legal_benchmark.yaml --skip-ingestion
```

---

### Quick smoke test (5 questions only)

```bash
python run_benchmark.py --config config/legal_benchmark.yaml --skip-ingestion --limit 5
```

---

### Run one dataset from a multi-dataset config

```bash
python run_benchmark.py --config config/legal_benchmark.yaml --skip-ingestion --dataset PrivacyQA
```

---

### One dataset, 5 questions, skip ingestion

```bash
python run_benchmark.py --config config/legal_benchmark.yaml --skip-ingestion --dataset PrivacyQA --limit 5
```

---

### Generate test questions automatically from documents

```bash
# Generate Q&A pairs from docs, then run the full benchmark
python run_benchmark.py --config config/legal_benchmark.yaml --generate-testset

# Generate only (no benchmark run) — set testset_generation.enabled: false in config after generating
python run_benchmark.py --config config/legal_benchmark.yaml --generate-testset --limit 0
```

Requires `testset_generation` block in config:
```yaml
testset_generation:
  enabled: true
  output_path: "./data/test_dataset.json"
  testset_size: 20
  query_distribution:
    simple: 0.5
    multi_context: 0.3
    reasoning: 0.2
```

---

### Resume an interrupted run

```bash
# Start a run with a known ID
python run_benchmark.py --config config/legal_benchmark.yaml --run-id my-run-001

# If it crashes mid-way, resume — already-done samples are skipped
python run_benchmark.py --config config/legal_benchmark.yaml --resume my-run-001
```

---

### Also run multi-turn conversation evaluation

```bash
python run_benchmark.py --config config/legal_benchmark.yaml --skip-ingestion --multi-turn
```

Requires `data/multi_turn_dataset.json`:
```json
[
  {
    "id": "conv_001",
    "turns": [
      {"role": "human", "content": "What is the data retention policy?"},
      {"role": "human", "content": "Does that apply to EU users too?"}
    ],
    "reference": "Data is retained for 90 days; EU users follow GDPR rules.",
    "reference_topics": ["data retention", "GDPR"]
  }
]
```

---

### Inspect a HuggingFace dataset before using it

```bash
python run_benchmark.py --print-hf-info explodinggradients/amnesty_qa
python run_benchmark.py --print-hf-info PatronusAI/financebench
```
Shows columns, row count, and first row preview — no API calls made.

---

### Run benchmark using a HuggingFace Q&A dataset

```bash
# Use HF dataset as test questions (auto-detects question/answer columns)
python run_benchmark.py --config config/legal_benchmark.yaml \
  --hf-dataset explodinggradients/amnesty_qa \
  --hf-mode qa \
  --skip-ingestion
```

---

### Ingest documents from a HuggingFace corpus dataset

```bash
# Download corpus from HF and ingest into HydraDB
python run_benchmark.py --config config/legal_benchmark.yaml \
  --hf-dataset isaacus/legal-rag-bench \
  --hf-mode corpus
```

---

### Extract corpus from a HuggingFace Q&A dataset and ingest

```bash
# One file per row (default)
python run_benchmark.py --extract-qa-corpus explodinggradients/amnesty_qa ./data/corpus

# Group all passages for the same source document into one file
python run_benchmark.py --extract-qa-corpus PatronusAI/financebench ./data/corpus --group-by doc_name

# Then run benchmark with the extracted corpus
python run_benchmark.py --config config/legal_benchmark.yaml
```

---

### Inspect and customize metric prompts

```bash
# Dump all prompt instructions to reports/prompts/
python run_benchmark.py --config config/legal_benchmark.yaml --save-prompts --limit 0

# Edit the JSON files, then add overrides to your config:
```
```yaml
ragas:
  prompt_overrides:
    overrides:
      faithfulness: "Judge whether every claim in the answer is supported by the context. Be strict."
      context_recall: "Check if all key facts from the reference answer appear in the contexts."
```

---

### Full run with all features

```bash
python run_benchmark.py \
  --config config/legal_benchmark.yaml \
  --generate-testset \
  --multi-turn \
  --save-prompts \
  --run-id full-run-v1 \
  -v
```

---

### Reset tenant and re-ingest from scratch

```bash
# WARNING: deletes all existing data in the tenant before re-ingesting
python run_benchmark.py --config config/legal_benchmark.yaml --reset-tenant
```

---

## Metrics Reference

### Standard RAGAS Metrics

| Metric | What it measures | Needs LLM | Needs Embeddings | Required inputs |
|--------|-----------------|-----------|-----------------|-----------------|
| `faithfulness` | Every claim in the answer is supported by retrieved contexts | Yes | No | response, retrieved_contexts |
| `response_relevancy` | Answer is relevant and complete for the question | Yes | Yes | user_input, response |
| `context_precision` | Retrieved contexts ranked — relevant ones appear first | Yes | No | user_input, retrieved_contexts, reference |
| `context_recall` | All facts needed to answer were retrieved | Yes | No | retrieved_contexts, reference_contexts |
| `context_relevance` | Retrieved chunks are about the question (no noise) | Yes | No | user_input, retrieved_contexts |
| `context_utilization` | How much of the retrieved context was used in the answer | Yes | No | user_input, response, retrieved_contexts |
| `context_entity_recall` | Named entities from reference appear in retrieved contexts | Yes | No | retrieved_contexts, reference |
| `noise_sensitivity` | Answer is not affected by irrelevant context | Yes | No | user_input, response, retrieved_contexts, reference |
| `factual_correctness` | Answer contains correct facts vs. reference answer | Yes | No | response, reference |
| `answer_correctness` | Combined factual + semantic correctness vs. reference | Yes | Yes | response, reference |
| `answer_accuracy` | Answer accuracy from NV metrics | Yes | No | response, reference |
| `response_groundedness` | Answer grounded in retrieved contexts | Yes | No | response, retrieved_contexts |
| `semantic_similarity` | Embedding cosine similarity between answer and reference | No | Yes | response, reference |
| `bleu_score` | BLEU overlap between answer and reference | No | No | response, reference |
| `rouge_score` | ROUGE overlap between answer and reference | No | No | response, reference |
| `chrf_score` | Character n-gram F-score between answer and reference | No | No | response, reference |
| `exact_match` | Exact string match between answer and reference | No | No | response, reference |
| `non_llm_string_similarity` | String edit distance between answer and reference | No | No | response, reference |
| `non_llm_context_recall` | String-match context recall (no LLM) | No | No | retrieved_contexts, reference_contexts |
| `summary_score` | Quality of a summarization output | Yes | No | response, reference |
| `topic_adherence` | Multi-turn: agent stays on allowed topics | Yes | No | user_input (multi-turn) |
| `agent_goal_accuracy` | Multi-turn: agent achieved the conversation goal | Yes | No | user_input, reference (multi-turn) |

### Aspect Critics (custom, binary)

Defined in config. LLM votes yes/no on your custom criterion. `strictness` controls how many votes are taken (majority wins).

```yaml
aspect_critics:
  - name: "cites_source"
    definition: "Does the answer reference or cite the source document?"
    strictness: 1
```

Output: `0` (fail) or `1` (pass).

### Scored Criteria (custom, continuous)

Defined in config. LLM scores 0–3 on your rubric; automatically normalized to 0–1 in results.

```yaml
scored_criteria:
  - name: "completeness"
    definition: |
      3 — answer covers all aspects of the question
      2 — covers most aspects, minor gaps
      1 — covers some aspects but misses important parts
      0 — answer is incomplete or missing key information
```

Output: `0.0` to `1.0`.

---

## Test Dataset Format

`test_dataset.json` is a JSON array of objects:

```json
[
  {
    "id": "q001",
    "question": "Can Fiverr share data with third parties?",
    "reference_answer": "Yes, Fiverr may share data with service providers and partners.",
    "reference_contexts": [
      "Fiverr may share your personal data with third-party service providers..."
    ]
  }
]
```

| Field | Required | Used by |
|-------|----------|---------|
| `id` | Yes | Checkpointing, reports |
| `question` | Yes | All metrics |
| `reference_answer` | Yes (for most metrics) | `context_recall`, `factual_correctness`, `answer_correctness`, Aspect Critics |
| `reference_contexts` | Only for `context_recall` | `context_recall`, `non_llm_context_recall` |

---

## Reports

After each run, three files are saved to `./reports/`:

| File | Contents |
|------|----------|
| `bench_<run_id>.json` | Full results including all per-sample scores, latency, token usage |
| `bench_<run_id>.csv` | Per-sample scores as a spreadsheet |
| `bench_<run_id>.html` | Self-contained visual report with charts and tables |

### Report contents

- Aggregate score per metric (mean across all samples)
- Per-sample breakdown: question, answer, retrieved context, every metric score, reasons (for Aspect Critics)
- Latency stats: min, max, p50, p95
- Token usage and estimated cost
- Config snapshot for reproducibility

---

## Project Structure

```
hydradb-bench/
├── run_benchmark.py              # Main entry point (CLI)
├── config/
│   └── legal_benchmark.yaml      # Example multi-dataset config
├── data/
│   ├── test_dataset.json         # Q&A test samples (single-turn)
│   ├── multi_turn_dataset.json   # Multi-turn conversation samples
│   └── documents/                # Documents to ingest into HydraDB
├── src/hydradb_bench/
│   ├── client.py                 # HydraDB HTTP client
│   ├── config.py                 # YAML + .env config loader
│   ├── models.py                 # Pydantic data models
│   ├── runner.py                 # Query runner (sends questions to HydraDB)
│   ├── evaluator.py              # RAGAS evaluation wrapper (30+ metrics)
│   ├── ingestion.py              # Document upload orchestrator
│   ├── testset_generator.py      # Auto-generate Q&A from documents
│   ├── hf_loader.py              # HuggingFace dataset loader
│   ├── multi_turn.py             # Multi-turn conversation evaluation
│   ├── context_builder.py        # Format HydraDB full_recall response
│   ├── checkpoint.py             # Resume interrupted runs
│   ├── reporter.py               # JSON / CSV / HTML report generator
│   ├── telegram_notifier.py      # Telegram notifications (summary + zip)
│   └── cli.py                    # Package entry point (hydradb-bench command)
├── tests/                        # Unit tests
├── reports/                      # Generated benchmark reports
├── .env                          # Credentials (never commit)
└── pyproject.toml
```
