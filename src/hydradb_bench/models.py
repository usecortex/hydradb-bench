"""Pydantic data models for the HydraDB benchmark framework."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------

class HydraConfig(BaseModel):
    base_url: str = "https://api.hydradb.com"
    api_key: str
    timeout_seconds: int = 30
    polling_interval_seconds: int = 5
    max_polling_attempts: int = 60
    tenant_id: str
    sub_tenant_id: str = ""
    create_tenant_on_start: bool = True


class IngestionConfig(BaseModel):
    documents_dir: str = "./data/documents"
    file_extensions: list[str] = [".txt", ".pdf", ".md", ".docx"]
    verify_before_querying: bool = True
    upload_delay_seconds: float = 0.5   # delay between consecutive uploads


class AspectCriticConfig(BaseModel):
    """Configuration for a single AspectCritic metric."""
    name: str                  # used as metric name in results
    definition: str            # criteria the LLM judges against
    strictness: int = 1        # >1 enables majority-vote self-consistency


class ScoredCriteriaConfig(BaseModel):
    """Configuration for a SimpleCriteriaScore metric — outputs continuous 0-1 score."""
    name: str        # used as metric name in results
    definition: str  # rubric the LLM grades against (0=worst … 3=best, normalised to 0-1)


class TestsetGenerationConfig(BaseModel):
    """Auto-generate Q&A pairs from documents using RAGAS TestsetGenerator."""
    enabled: bool = False
    output_path: str = "./data/test_dataset.json"
    testset_size: int = 15
    # Distribution of query complexity types (must sum to 1.0)
    query_distribution: dict[str, float] = Field(default_factory=lambda: {
        "simple": 0.5,
        "multi_context": 0.3,
        "reasoning": 0.2,
    })


class MultiTurnConfig(BaseModel):
    """Multi-turn conversation evaluation settings."""
    enabled: bool = False
    dataset_path: str = "./data/multi_turn_dataset.json"
    metrics: list[str] = Field(default_factory=lambda: [
        "topic_adherence",
        "agent_goal_accuracy",
    ])


class CostTrackingConfig(BaseModel):
    """Track OpenAI token usage and estimate costs."""
    enabled: bool = True
    # gpt-4o-mini defaults (per token)
    cost_per_input_token: float = 0.00000015   # $0.15 / 1M tokens
    cost_per_output_token: float = 0.0000006   # $0.60 / 1M tokens


class AnswerGenerationConfig(BaseModel):
    """When using full_recall, generate answers via an external LLM instead of HydraDB qna."""
    enabled: bool = False
    provider: str = "openai"        # "openai" | "openrouter"
    model: str = "gpt-4o-mini"
    system_prompt: str = (
        "You are a helpful assistant. Answer the question using only the provided context. "
        "Be concise and accurate."
    )
    max_tokens: int = 1024
    temperature: float = 0.0        # 0.0 = deterministic, higher = more creative


class EvaluationConfig(BaseModel):
    test_dataset_path: str = "./data/test_dataset.json"
    search_endpoint: str = "qna"  # "qna" | "full_recall"
    retrieve_mode: str = "fast"   # "fast" | "thinking" (full_recall only)
    max_results: int = 5
    concurrent_requests: int = 3
    metrics: list[str] = Field(default_factory=list)
    aspect_critics: list[AspectCriticConfig] = Field(default_factory=list)
    scored_criteria: list[ScoredCriteriaConfig] = Field(default_factory=list)
    multi_turn: MultiTurnConfig = Field(default_factory=MultiTurnConfig)
    answer_generation: AnswerGenerationConfig = Field(default_factory=AnswerGenerationConfig)


class PromptOverridesConfig(BaseModel):
    """Map of metric_name → custom instruction string."""
    overrides: dict[str, str] = Field(default_factory=dict)


class RAGASConfig(BaseModel):
    llm_model: str = "gpt-4o-mini"
    embeddings_model: str = "text-embedding-3-small"
    # LLM provider: "openai" (default) | "openrouter"
    # For openrouter, set llm_model to e.g. "anthropic/claude-3.5-sonnet"
    llm_provider: str = "openai"
    # Embeddings provider:
    #   "openai"       — OpenAI text-embedding-* (needs OPENAI_API_KEY)
    #   "openrouter"   — OpenRouter-hosted embedding models (needs OPENROUTER_API_KEY)
    #   "huggingface"  — sentence-transformers run locally (no API key required)
    #                    e.g. embeddings_model: "BAAI/bge-small-en-v1.5"
    #   "ollama"       — Ollama local server (no API key required)
    #                    e.g. embeddings_model: "nomic-embed-text"
    embeddings_provider: str = "openai"
    # Base URL override for embeddings (used by "ollama" provider;
    # defaults to http://localhost:11434 when provider is "ollama")
    embeddings_base_url: str | None = None
    max_retries: int = 3
    timeout: int = 60
    max_workers: int = 4
    raise_exceptions: bool = False
    temperature: float = 0.0        # judge LLM temperature; 0.0 = deterministic
    judge_max_tokens: int = 4096    # max tokens for judge LLM response
    cost_tracking: CostTrackingConfig = Field(default_factory=CostTrackingConfig)
    prompt_overrides: PromptOverridesConfig = Field(default_factory=PromptOverridesConfig)


class ReportingConfig(BaseModel):
    formats: list[str] = ["json", "csv", "html"]
    include_per_sample_scores: bool = True
    include_reasons: bool = True


class SlackConfig(BaseModel):
    enabled: bool = False
    # Bot token from env: SLACK_BOT_TOKEN
    bot_token: str = ""
    # Your Slack member ID (starts with U) — click your profile in Slack → ... → Copy member ID
    # The bot will use conversations.open to get/create your DM channel automatically.
    user_id: str = ""


class HFDatasetConfig(BaseModel):
    """Load a HuggingFace dataset as evaluation data or as documents for ingestion."""
    enabled: bool = False
    mode: str = "qa"                          # "qa" | "corpus"
    repo_id: str = ""                         # HuggingFace repo ID
    split: str = "test"
    config_name: str | None = None            # HF dataset config/subset name (e.g. "qa", "corpus")
    # QA mode
    max_samples: int | None = None
    column_map: dict[str, str] = Field(default_factory=dict)
    save_qa_path: str | None = None
    id_prefix: str = "hf"
    # Corpus mode
    max_docs: int | None = None
    text_column: str | None = None
    title_column: str | None = None
    corpus_output_dir: str = "./data/hf_documents"


class DatasetEntry(BaseModel):
    """One dataset in a multi-dataset benchmark run."""
    name: str                          # display name used in reports
    sub_tenant_id: str                 # isolated HydraDB sub-tenant for this corpus
    documents_dir: str                 # path to corpus documents for ingestion
    test_dataset_path: str             # path to converted Q&A samples JSON


class BenchmarkConfig(BaseModel):
    name: str = "HydraDB RAG Benchmark"
    run_id: str | None = None
    output_dir: str = "./reports"
    hydradb: HydraConfig
    ingestion: IngestionConfig
    testset_generation: TestsetGenerationConfig = Field(default_factory=TestsetGenerationConfig)
    hf_dataset: HFDatasetConfig = Field(default_factory=HFDatasetConfig)
    evaluation: EvaluationConfig
    ragas: RAGASConfig
    reporting: ReportingConfig
    slack: SlackConfig = Field(default_factory=SlackConfig)
    # Multi-dataset mode: when set, loops through each entry overriding
    # sub_tenant_id / documents_dir / test_dataset_path per dataset.
    datasets: list[DatasetEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Test data models
# ---------------------------------------------------------------------------

class TestSample(BaseModel):
    """One row in test_dataset.json (single-turn Q&A)."""
    id: str
    question: str
    reference_answer: str
    reference_contexts: list[str] = Field(default_factory=list)


class MultiTurnMessage(BaseModel):
    """A single message in a multi-turn conversation."""
    role: str   # "human" | "ai"
    content: str


class MultiTurnSampleConfig(BaseModel):
    """One entry in multi_turn_dataset.json."""
    id: str
    turns: list[MultiTurnMessage]   # human turns only; AI responses come from HydraDB
    reference: str = ""             # expected final outcome / goal
    reference_topics: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# HydraDB response models
# ---------------------------------------------------------------------------

class HydraSearchResult(BaseModel):
    answer: str
    retrieved_contexts: list[str]
    raw_response: dict[str, Any] = Field(default_factory=dict)
    network_latency_ms: float = 0.0  # time from request send to response received (HydraDB only)
    context_string: str = ""   # formatted output of build_context_string()
    context_tokens: int = 0    # tiktoken cl100k_base token count of context_string


# ---------------------------------------------------------------------------
# Benchmark pipeline models
# ---------------------------------------------------------------------------

class BenchmarkSample(BaseModel):
    test_sample: TestSample
    hydra_result: HydraSearchResult | None = None
    latency_ms: float = 0.0
    error: str | None = None


class MultiTurnBenchmarkSample(BaseModel):
    conversation_id: str
    conversation_turns: list[dict[str, str]]  # {"role": ..., "content": ..., "latency_ms": ...}
    reference: str = ""
    reference_topics: list[str] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    error: str | None = None


class IngestionFileStatus(BaseModel):
    file_path: str
    file_id: str = ""
    status: str = "pending"
    error: str | None = None


class IngestionReport(BaseModel):
    files: list[IngestionFileStatus] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    all_indexed: bool = False

    def summary(self) -> str:
        indexed = sum(1 for f in self.files if f.status == "indexed")
        failed = sum(1 for f in self.files if f.status == "failed")
        return (
            f"{indexed}/{len(self.files)} files indexed, "
            f"{failed} failed, {self.elapsed_seconds:.1f}s elapsed"
        )


class TokenUsageResult(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    actual_cost_usd: float = 0.0
    model: str = ""


class BenchmarkResult(BaseModel):
    run_id: str
    timestamp: str
    benchmark_name: str
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    samples: list[BenchmarkSample] = Field(default_factory=list)
    multi_turn_samples: list[MultiTurnBenchmarkSample] = Field(default_factory=list)
    ragas_scores: dict[str, float] = Field(default_factory=dict)
    multi_turn_scores: dict[str, float] = Field(default_factory=dict)
    per_sample_scores: list[dict[str, Any]] = Field(default_factory=list)
    latency_stats: dict[str, float] = Field(default_factory=dict)
    token_usage: TokenUsageResult = Field(default_factory=TokenUsageResult)
    error_count: int = 0
    evaluated_count: int = 0
