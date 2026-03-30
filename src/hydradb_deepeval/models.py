from __future__ import annotations

from pydantic import BaseModel


class SupermemoryConfig(BaseModel):
    base_url: str = "https://api.supermemory.ai"
    api_key: str
    container_tag: str = "benchmark"
    timeout_seconds: int = 30
    # Ingestion
    verify_before_querying: bool = True
    polling_interval_seconds: int = 5
    max_polling_attempts: int = 60
    reset_on_start: bool = False
    # Search
    search_mode: str = "hybrid"  # "hybrid" | "memories"
    rerank: bool = False
    threshold: float = 0.0        # minimum similarity score (0 = return all)
    limit: int = 20               # total chunks returned (not documents) — keep high enough
                                  # so results span multiple documents (5 chunks ≈ 1 doc only)


class HydraConfig(BaseModel):
    base_url: str = "https://api.hydradb.com"
    api_key: str
    tenant_id: str
    sub_tenant_id: str = ""
    timeout_seconds: int = 30
    polling_interval_seconds: int = 5
    max_polling_attempts: int = 60
    create_tenant_on_start: bool = True


class IngestionConfig(BaseModel):
    documents_dir: str = "./data/documents"
    file_extensions: list[str] = [".txt", ".pdf", ".md", ".docx"]
    verify_before_querying: bool = True
    upload_delay_seconds: float = 0.5
    upload_concurrency: int = 10  # files uploaded in parallel per batch


class DeepEvalConfig(BaseModel):
    model: str = "gpt-4o-mini"
    metrics: list[str] = [
        "answer_relevancy",
        "contextual_precision",
        "contextual_recall",
        "contextual_relevancy",
    ]
    threshold: float = 0.5
    include_reason: bool = True
    eval_concurrency: int = 5       # samples evaluated in parallel
    metric_timeout_seconds: int = 60  # per-metric a_measure() timeout
    # Answer generation (used when endpoint is full_recall / recall_preferences / boolean_recall)
    generator_model: str = "gpt-4o-mini"
    generator_temperature: float = 0.0
    generator_max_tokens: int = 1024


class EvaluationConfig(BaseModel):
    test_dataset_path: str = "./data/test_dataset.json"
    # Endpoint: "full_recall" | "recall_preferences" | "boolean_recall"
    search_endpoint: str = "full_recall"
    max_results: int = 5
    concurrent_requests: int = 3
    # Shared recall params
    mode: str = "fast"       # "fast" | "thinking"
    alpha: float = 0.8       # hybrid vector/BM25 weight (0=BM25, 1=vector)
    # full_recall / recall_preferences
    graph_context: bool = True
    recency_bias: float = 0.0
    # boolean_recall
    boolean_operator: str = "or"   # "or" | "and" | "phrase"
    boolean_search_mode: str = "sources"  # "sources" | "memories"


class ReportingConfig(BaseModel):
    output_dir: str = "./reports"
    formats: list[str] = ["json", "html"]
    include_per_sample: bool = True


class BenchmarkConfig(BaseModel):
    name: str = "HydraDB DeepEval Benchmark"
    hydradb: HydraConfig
    ingestion: IngestionConfig
    evaluation: EvaluationConfig
    deepeval: DeepEvalConfig
    reporting: ReportingConfig
    supermemory: SupermemoryConfig | None = None  # optional; required for --provider supermemory|both


class TestSample(BaseModel):
    id: str
    question: str
    reference_answer: str
    reference_contexts: list[str] = []


class QueryResult(BaseModel):
    sample: TestSample
    answer: str = ""
    retrieved_contexts: list[str] = []
    context_string: str = ""  # output of build_context_string (for display in report)
    context_tokens: int = 0   # tiktoken count of context_string
    latency_ms: float = 0.0
    error: str | None = None


class SampleScore(BaseModel):
    sample_id: str
    question: str
    answer: str
    reference_answer: str
    context_string: str = ""  # full build_context_string output for debugging
    context_tokens: int = 0
    scores: dict[str, float | None] = {}
    reasons: dict[str, str | None] = {}
    latency_ms: float = 0.0


class BenchmarkResult(BaseModel):
    run_id: str
    timestamp: str
    name: str
    aggregate_scores: dict[str, float] = {}
    per_sample: list[SampleScore] = []
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_stats: dict[str, float] = {}       # min, mean, p50, p75, p95, p99, max
    context_token_stats: dict[str, float] = {} # min, mean, max of context tokens per query
    total_samples: int = 0
    error_count: int = 0
