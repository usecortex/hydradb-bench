# Skill: Run Benchmark

## Description
Run the HydraDB/Supermemory DeepEval benchmark pipeline.

## When to Use
When the user asks to run a benchmark, evaluate retrieval quality, test a provider, or produce benchmark reports.

## Prerequisites
- `.env` file with required API keys (`HYDRADB_API_KEY`, `OPENAI_API_KEY`, optionally `SUPERMEMORY_API_KEY`)
- `config/benchmark.yaml` configured for the target provider(s)
- Python virtual environment activated with dependencies installed (`pip install -e .`)
- Test dataset at the path specified in `evaluation.test_dataset_path` (default: `data/privacy_qa.json`)

## Steps

### 1. Verify Environment
```bash
# Check .env exists and has required keys
cat .env | grep -c "API_KEY"

# Check config exists
ls config/benchmark.yaml

# Check test dataset exists
python -c "import json; d=json.load(open('data/privacy_qa.json')); print(f'{len(d)} samples loaded')"
```

### 2. Choose the Run Command

**HydraDB only (default):**
```bash
python run_benchmark.py --provider hydradb
```

**Supermemory only:**
```bash
python run_benchmark.py --provider supermemory
```

**Both providers + comparison:**
```bash
python run_benchmark.py --provider both
```

### 3. Common Flags
| Flag | Purpose |
|------|---------|
| `--skip-ingestion` | Skip document upload, query existing indexed docs |
| `--ingest-only` | Upload documents and exit (no querying/eval) |
| `--reset-tenant` | Delete and recreate HydraDB tenant before ingestion |
| `--limit N` | Run only first N test samples (useful for smoke tests) |
| `--verbose` | Print per-sample query details |
| `--config PATH` | Use a custom config file |

### 4. Typical Workflows

**Quick smoke test:**
```bash
python run_benchmark.py --provider hydradb --skip-ingestion --limit 10 --verbose
```

**Full fresh run with tenant reset:**
```bash
python run_benchmark.py --provider hydradb --reset-tenant
```

**Re-evaluate without re-ingesting:**
```bash
python run_benchmark.py --provider hydradb --skip-ingestion
```

**Side-by-side comparison:**
```bash
python run_benchmark.py --provider both --skip-ingestion --limit 20
```

### 5. Output
Reports are saved to `reporting.output_dir` (default: `./reports/`):
- `<run_id>_hydradb.json` — Machine-readable full results
- `<run_id>_hydradb.html` — Interactive HTML report
- `<run_id>_hydradb.csv` — Flattened per-sample CSV (if enabled)
- `<run_id>_comparison.html` — Side-by-side comparison (when `--provider both`)

## Troubleshooting
- **Missing API key errors**: Check `.env` has the required keys and `python-dotenv` is loading them
- **Supermemory key required even for HydraDB-only**: Remove/comment the `supermemory:` section from `config/benchmark.yaml`
- **Metric timeouts**: Reduce `deepeval.eval_concurrency` or increase `deepeval.metric_timeout_seconds` in config
- **No documents found**: Check `ingestion.documents_dir` path and `ingestion.file_extensions` match your files
