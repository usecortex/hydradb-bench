# Skill: Add a New Evaluation Metric

## Description
Add a new DeepEval metric or custom metric to the benchmark evaluation pipeline.

## When to Use
When the user wants to evaluate retrieval/answer quality with additional metrics beyond the built-in ones.

## Built-in Metrics
Already supported in `src/hydradb_deepeval/evaluator.py`:
- `answer_accuracy` (custom GEval, binary 0/1)
- `answer_relevancy`
- `faithfulness`
- `contextual_precision`
- `contextual_recall`
- `contextual_relevancy`
- `hallucination`
- `bias`
- `toxicity`
- `summarization`

## Adding a Standard DeepEval Metric

### 1. Check DeepEval supports it
```python
from deepeval.metrics import <MetricName>
```

### 2. Add to the registry in `src/hydradb_deepeval/evaluator.py`
Find the `_METRIC_REGISTRY` dict and add an entry:
```python
_METRIC_REGISTRY = {
    # ... existing entries ...
    "new_metric_name": {
        "module": "deepeval.metrics",
        "class": "NewMetricClass",
    },
}
```

### 3. Enable in config
Add to `deepeval.metrics` list in `config/benchmark.yaml`:
```yaml
deepeval:
  metrics:
    - answer_accuracy
    - contextual_precision
    - new_metric_name    # <-- add here
```

That's it. The evaluator dynamically imports and instantiates metrics from the registry.

## Adding a Custom GEval Metric

For metrics not built into DeepEval, use `GEval` (like `answer_accuracy` does):

### 1. Add to the evaluator's metric creation logic
In `src/hydradb_deepeval/evaluator.py`, find where `answer_accuracy` is handled (special case in the metric instantiation code) and add a similar block:

```python
if name == "my_custom_metric":
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams
    metric = GEval(
        name="My Custom Metric",
        criteria="Evaluate whether the actual output...",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=self._model,
        threshold=self._threshold,
        strict_mode=False,  # True for binary 0/1
    )
```

### 2. Enable in config
```yaml
deepeval:
  metrics:
    - my_custom_metric
```

## How the Evaluator Works
Understanding this helps when adding complex metrics:

1. `evaluate(results)` iterates over `QueryResult` objects
2. For each sample, it builds an `LLMTestCase`:
   ```python
   LLMTestCase(
       input=sample.question,
       actual_output=answer,
       retrieval_context=retrieved_chunks_list,
       expected_output=reference_answer,
   )
   ```
3. Each metric is instantiated fresh per sample (avoids shared state)
4. Metrics run in parallel per sample with `asyncio.gather`
5. Per-metric timeout is enforced via `asyncio.wait_for`
6. Scores are aggregated as mean across all samples

## Important Notes
- Metrics that need `retrieval_context` (contextual_*) require the query to return chunks as a list
- Metrics that need `expected_output` (answer_accuracy, contextual_recall) require `reference_answer` and `reference_contexts` in the test dataset
- The `threshold` in config is the pass/fail cutoff displayed in reports
- Set `include_reason: true` in config to get explanations for each metric score
- If a metric times out frequently, increase `deepeval.metric_timeout_seconds`
