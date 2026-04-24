# Skill: Add a New Retrieval Provider

## Description
Extend the benchmark framework to support a new retrieval provider alongside HydraDB and Supermemory.

## When to Use
When the user wants to benchmark a new vector DB, search API, or retrieval system using this framework.

## Architecture Overview
The framework follows a consistent pattern for each provider:
1. **Client** — Async HTTP wrapper for the provider's API (`src/hydradb_deepeval/client.py`, `supermemory_client.py`)
2. **Ingestion** — Document upload + polling logic (`src/hydradb_deepeval/ingestion.py`, inside `supermemory_client.py`)
3. **Config model** — Pydantic settings class (`src/hydradb_deepeval/models.py`)
4. **Query function** — In `run_benchmark.py`, a `run_single_query_<provider>()` function
5. **Config section** — YAML block in `config/benchmark.yaml`
6. **CLI integration** — `--provider` flag in `run_benchmark.py`

## Step-by-Step Guide

### 1. Add Config Model in `src/hydradb_deepeval/models.py`
```python
class NewProviderConfig(BaseModel):
    base_url: str
    api_key: str = ""
    # Add provider-specific fields
    timeout_seconds: int = 30
    # Search-specific params
    limit: int = 10
```

Add to `BenchmarkConfig`:
```python
class BenchmarkConfig(BaseModel):
    # ... existing fields ...
    new_provider: NewProviderConfig | None = None
```

### 2. Create Client in `src/hydradb_deepeval/`
Create `new_provider_client.py` following the pattern in `supermemory_client.py`:
- Async context manager class using `httpx.AsyncClient`
- `search(query, ...) -> dict` method returning results
- `upload_file(file_path, ...) -> dict` for ingestion
- An ingester class with `run() -> (indexed_count, failed_count, elapsed_seconds)`

Key pattern:
```python
class NewProviderClient:
    def __init__(self, cfg: NewProviderConfig):
        self._cfg = cfg
        self._http: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._http = httpx.AsyncClient(
            base_url=self._cfg.base_url,
            headers={"Authorization": f"Bearer {self._cfg.api_key}"},
            timeout=self._cfg.timeout_seconds,
        )
        return self

    async def __aexit__(self, *exc):
        if self._http:
            await self._http.aclose()

    async def search(self, query: str, limit: int = 10) -> dict:
        resp = await self._http.post("/search", json={"query": query, "limit": limit})
        resp.raise_for_status()
        return resp.json()
```

### 3. Update Config Loader in `src/hydradb_deepeval/config.py`
Add validation for the new provider's API key when its section exists:
```python
if "new_provider" in raw:
    api_key = os.getenv("NEW_PROVIDER_API_KEY", "")
    if not api_key:
        raise EnvironmentError("NEW_PROVIDER_API_KEY not set")
    raw["new_provider"]["api_key"] = api_key
```

### 4. Add Query Function in `run_benchmark.py`
```python
async def run_single_query_new_provider(client, sample, eval_cfg, provider_cfg, gen_cfg):
    t0 = time.perf_counter()
    try:
        result = await client.search(sample.question, limit=provider_cfg.limit)
        chunks = [item["content"] for item in result.get("results", [])]
        context_str = "\n\n".join(chunks)
        answer = await generate_answer(
            sample.question, context_str,
            model=gen_cfg["model"], temperature=gen_cfg["temperature"],
            max_tokens=gen_cfg["max_tokens"],
        )
        latency = (time.perf_counter() - t0) * 1000
        return QueryResult(
            sample=sample, answer=answer, retrieved_contexts=chunks,
            context_string=context_str, context_tokens=count_tokens(context_str),
            latency_ms=latency,
        )
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        return QueryResult(sample=sample, answer="", latency_ms=latency, error=str(e))
```

### 5. Wire Into CLI in `run_benchmark.py`
- Add `"new_provider"` to `--provider` choices
- Add ingestion block in `main()` for the new provider
- Add query execution block
- Update `--provider both` logic or add multi-provider support

### 6. Add Config Section
In `config/benchmark.yaml`:
```yaml
new_provider:
  base_url: "https://api.newprovider.com"
  timeout_seconds: 30
  limit: 10
  # provider-specific settings
```

### 7. Update Exports in `src/hydradb_deepeval/__init__.py`
Add imports for the new client and config classes.

## Testing the New Provider
```bash
# Ingest only
python run_benchmark.py --provider new_provider --ingest-only

# Smoke test
python run_benchmark.py --provider new_provider --skip-ingestion --limit 5 --verbose

# Full run
python run_benchmark.py --provider new_provider
```

## Key Patterns to Follow
- All HTTP clients use `httpx.AsyncClient` as async context managers
- Ingestion returns `(indexed_count, failed_count, elapsed_seconds)` tuple
- Query functions return `QueryResult` objects
- Answer generation always uses `generate_answer()` from `answer_generator.py`
- Context is built as a single string for the LLM and a list of chunks for DeepEval metrics
