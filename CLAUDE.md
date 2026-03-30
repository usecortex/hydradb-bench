# HydraDB DeepEval Benchmark Framework

## What This Is
A benchmark framework for evaluating retrieval + answer quality across HydraDB and Supermemory using DeepEval metrics. It handles document ingestion, multi-endpoint querying, LLM answer generation, metric evaluation, and rich report generation.

## Quick Reference
- **Entry point**: `run_benchmark.py` (CLI with `--provider`, `--skip-ingestion`, `--limit`, etc.)
- **Config**: `config/benchmark.yaml` (YAML with `${ENV_VAR}` interpolation)
- **Source code**: `src/hydradb_deepeval/` (client, evaluator, reporter, models, config, ingestion)
- **Test data**: `data/privacy_qa.json` (Q/A dataset), `data/privacy_qa/` (source documents)
- **Reports**: `./reports/` (JSON, HTML, CSV)
- **Dependencies**: Python 3.10+, see `pyproject.toml`

## Skills for AI Agents
Read the skill files in `.claude/skills/` before working on tasks:

| Skill | File | Use When |
|-------|------|----------|
| **Run Benchmark** | `.claude/skills/run-benchmark.md` | Running benchmarks, choosing flags, interpreting output |
| **Configure** | `.claude/skills/configure-benchmark.md` | Editing `config/benchmark.yaml`, tuning parameters |
| **Generate Test Data** | `.claude/skills/generate-test-data.md` | Creating synthetic Q/A datasets from documents |
| **Analyze Reports** | `.claude/skills/analyze-reports.md` | Reading/comparing benchmark results |
| **Add Provider** | `.claude/skills/add-provider.md` | Integrating a new retrieval system |
| **Add Metric** | `.claude/skills/add-metric.md` | Adding DeepEval or custom evaluation metrics |
| **Manage Test Data** | `.claude/skills/manage-test-data.md` | Editing datasets, adding samples, validation |
| **Troubleshoot** | `.claude/skills/troubleshoot.md` | Fixing errors, debugging pipeline issues |
| **Codebase Overview** | `.claude/skills/codebase-overview.md` | Understanding architecture and data flow |

## Common Commands
```bash
# Install
pip install -e .

# Run HydraDB benchmark
python run_benchmark.py --provider hydradb

# Quick smoke test
python run_benchmark.py --provider hydradb --skip-ingestion --limit 10 --verbose

# Compare providers
python run_benchmark.py --provider both --skip-ingestion

# Generate synthetic test data
python generate_test_data.py

# Convert report to CSV
python json_to_csv.py reports/<report>.json
```

## Environment Variables Required
- `HYDRADB_API_KEY` — Always required
- `OPENAI_API_KEY` — Required for answer generation and DeepEval metrics
- `SUPERMEMORY_API_KEY` — Required only if `supermemory:` section exists in config

## Key Conventions
- All HTTP clients use `httpx.AsyncClient` as async context managers
- Config models are Pydantic v2 classes in `models.py`
- Query functions return `QueryResult`, evaluation returns `SampleScore`
- Reports support JSON, HTML (interactive), and CSV formats
- The evaluator dynamically imports metrics from a registry dict
