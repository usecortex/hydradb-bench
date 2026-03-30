# Skill: Troubleshoot Benchmark Issues

## Description
Diagnose and fix common errors in the benchmark pipeline.

## When to Use
When the user encounters errors running benchmarks, ingestion failures, metric timeouts, or unexpected results.

## Common Errors and Fixes

### Environment / Configuration

**`HYDRADB_API_KEY environment variable is not set`**
- Ensure `.env` has `HYDRADB_API_KEY=<key>`
- Check `.env` is in the project root (not a subdirectory)
- Verify `python-dotenv` is installed: `pip install python-dotenv`

**`OPENAI_API_KEY environment variable is not set`**
- Required for answer generation and DeepEval metric judges
- Add `OPENAI_API_KEY=<key>` to `.env`

**`supermemory section is present ... SUPERMEMORY_API_KEY not set`**
- Either add `SUPERMEMORY_API_KEY` to `.env`
- Or remove/comment the entire `supermemory:` section from `config/benchmark.yaml`
- This error occurs even for `--provider hydradb` if `supermemory:` section exists in config

**`FileNotFoundError: config/benchmark.yaml`**
- Create from template in README or copy from another config

**`Environment variable ${VAR} is not set`**
- A YAML value references `${VAR}` but the variable is missing from the environment
- Add it to `.env` or set it in shell

### Ingestion

**`No matching files found in ./data/privacy_qa`**
- Check `ingestion.documents_dir` path exists
- Check `ingestion.file_extensions` matches your file types
- Verify files are in the directory: `ls data/privacy_qa/`

**Documents stuck in processing**
- HydraDB: Increase `hydradb.max_polling_attempts` or `hydradb.polling_interval_seconds`
- Supermemory: Increase `supermemory.max_polling_attempts` or `supermemory.polling_interval_seconds`
- Check if the service is actually processing: look at verbose logs

**Upload failures (HTTP 4xx/5xx)**
- Verify API key is valid and has write permissions
- Check `hydradb.base_url` or `supermemory.base_url` is correct
- Check file size limits for the provider
- Try reducing `ingestion.upload_concurrency` to 1

### Querying

**Empty retrieval results**
- Documents may not be fully indexed yet — run with `--verbose` to see response details
- Increase `evaluation.max_results` or `supermemory.limit`
- Try a different `evaluation.search_endpoint` (e.g., `full_recall` vs `boolean_recall`)
- Verify ingestion completed: run `--ingest-only` first, then `--skip-ingestion`

**HTTP timeout on queries**
- Increase `hydradb.timeout_seconds` or `supermemory.timeout_seconds`
- Reduce `evaluation.concurrent_requests`
- Check if the service is overloaded

### Evaluation / Metrics

**Frequent metric timeouts**
```yaml
deepeval:
  eval_concurrency: 2          # Reduce from 5
  metric_timeout_seconds: 120  # Increase from 60
```

**All scores are None**
- Check OpenAI API key is valid (DeepEval uses it for metric judges)
- Check `deepeval.model` refers to a valid model
- Look at error reasons in the JSON report's `per_sample[].reasons`

**Very low scores across the board**
- Check if retrieval is returning relevant content (`--verbose` mode)
- Verify test dataset `reference_contexts` are actually in the indexed documents
- Try increasing `evaluation.max_results` for broader recall
- Check if `evaluation.mode` should be "thinking" instead of "fast"

**`answer_accuracy` is always 0 or 1**
- This is expected — it uses `strict_mode=True` (binary GEval)

### Reports

**No CSV output**
- Ensure `"csv"` is in `reporting.formats` list
- Ensure `reporting.include_per_sample: true`

**HTML report missing per-sample details**
- Set `reporting.include_per_sample: true` in config

**Convert existing JSON to CSV manually**
```bash
python json_to_csv.py reports/<report>.json
```

### General Debugging

**Enable verbose logging:**
```bash
python run_benchmark.py --verbose
```

**Run a minimal test first:**
```bash
python run_benchmark.py --provider hydradb --skip-ingestion --limit 3 --verbose
```

**Check installed dependencies:**
```bash
pip list | grep -E "deepeval|httpx|pydantic|openai|tiktoken"
```

**Verify config loads correctly:**
```python
from src.hydradb_deepeval.config import load_config
cfg = load_config("config/benchmark.yaml")
print(cfg)
```

## Key Files for Debugging
| File | What to check |
|------|---------------|
| `src/hydradb_deepeval/config.py` | Config loading, env var interpolation |
| `src/hydradb_deepeval/client.py` | HydraDB API calls, error handling |
| `src/hydradb_deepeval/supermemory_client.py` | Supermemory API calls |
| `src/hydradb_deepeval/evaluator.py` | Metric instantiation, timeout handling |
| `src/hydradb_deepeval/answer_generator.py` | LLM answer generation |
| `run_benchmark.py` | Query orchestration, CLI args, pipeline flow |
