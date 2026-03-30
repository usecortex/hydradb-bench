# Skill: Configure Benchmark

## Description
Create or modify `config/benchmark.yaml` to set up benchmark parameters for HydraDB and/or Supermemory evaluation runs.

## When to Use
When the user wants to change benchmark settings, switch search endpoints, adjust metrics, tune retrieval parameters, add/remove Supermemory, or create a new config from scratch.

## Config File Location
`config/benchmark.yaml`

## Config Structure

```yaml
benchmark:
  name: "Display name for reports"

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
  search_endpoint: "full_recall"        # full_recall | recall_preferences | boolean_recall
  max_results: 10
  concurrent_requests: 3
  mode: "thinking"                      # fast | thinking
  alpha: 0.8                            # semantic vs keyword balance (0-1)
  graph_context: true
  recency_bias: 0.0
  boolean_operator: "or"                # or | and | phrase (for boolean_recall)
  boolean_search_mode: "sources"        # sources | memories (for boolean_recall)

deepeval:
  model: "gpt-5.4"                      # Model for DeepEval metric judges
  threshold: 0.5                        # Pass/fail threshold
  include_reason: true
  eval_concurrency: 5
  metric_timeout_seconds: 60
  generator_model: "gpt-5.4"            # Model for answer generation from context
  generator_temperature: 0.0
  generator_max_tokens: 1024
  metrics:                              # Which metrics to run
    - answer_accuracy
    - contextual_precision
    - contextual_recall
    - contextual_relevancy

reporting:
  output_dir: "./reports"
  formats: ["json", "html", "csv"]
  include_per_sample: true
```

### Optional: Add Supermemory Section
```yaml
supermemory:
  base_url: "https://api.supermemory.ai"
  container_tag: "privacyqa"
  timeout_seconds: 30
  verify_before_querying: true
  polling_interval_seconds: 5
  max_polling_attempts: 60
  reset_on_start: false
  search_mode: "hybrid"                # hybrid | memories
  rerank: true
  threshold: 0.0
  limit: 20
```

## Available Metrics
Configured in `deepeval.metrics` list. Supported values:
- `answer_accuracy` — Custom GEval, strict binary 0/1 scoring
- `answer_relevancy` — How relevant the answer is to the question
- `faithfulness` — Whether the answer is grounded in the retrieved context
- `contextual_precision` — Are the relevant contexts ranked higher?
- `contextual_recall` — Are all relevant contexts retrieved?
- `contextual_relevancy` — Are the retrieved contexts relevant?
- `hallucination` — Does the answer hallucinate beyond context?
- `bias` — Does the answer contain bias?
- `toxicity` — Does the answer contain toxic content?
- `summarization` — Quality of summarization

## Environment Variable Interpolation
Use `${ENV_VAR}` in YAML values to reference environment variables:
```yaml
hydradb:
  tenant_id: "${HYDRADB_TENANT_ID}"
```

## Key Tuning Parameters
| Parameter | What it controls | Typical adjustment |
|-----------|------------------|--------------------|
| `evaluation.search_endpoint` | HydraDB search type | Switch between full_recall/recall_preferences/boolean_recall |
| `evaluation.max_results` | Number of chunks returned | Increase for broader recall, decrease for precision |
| `evaluation.mode` | fast vs thinking mode | "thinking" is slower but more accurate |
| `evaluation.alpha` | Semantic vs keyword weight | Higher = more semantic, lower = more keyword |
| `deepeval.eval_concurrency` | Parallel metric evaluations | Lower if hitting rate limits or timeouts |
| `deepeval.metric_timeout_seconds` | Per-metric timeout | Increase if metrics keep timing out |
| `supermemory.limit` | Supermemory result count | Increase for broader recall |
| `supermemory.search_mode` | hybrid vs memories | hybrid uses both vector + keyword search |

## Important Notes
- If `supermemory:` section exists in config, `SUPERMEMORY_API_KEY` is required even for HydraDB-only runs
- For HydraDB-only runs, remove/comment the entire `supermemory:` section
- Missing env vars in `${...}` references cause config loading to fail immediately
