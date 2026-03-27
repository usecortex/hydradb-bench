"""Configuration loading from YAML + .env."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .models import (
    AnswerGenerationConfig,
    AspectCriticConfig,
    BenchmarkConfig,
    CostTrackingConfig,
    DatasetEntry,
    EvaluationConfig,
    HFDatasetConfig,
    HydraConfig,
    IngestionConfig,
    MultiTurnConfig,
    PromptOverridesConfig,
    RAGASConfig,
    ReportingConfig,
    ScoredCriteriaConfig,
    SlackConfig,
    TestsetGenerationConfig,
)


def _interpolate_env_vars(value: Any) -> Any:
    """Recursively replace ${ENV_VAR} patterns with environment values."""
    if isinstance(value, str):
        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))
        return re.sub(r"\$\{([^}]+)\}", replacer, value)
    elif isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_interpolate_env_vars(item) for item in value]
    return value


def load_config(config_path: str = "config/benchmark.yaml") -> BenchmarkConfig:
    """Load and validate benchmark configuration from YAML + environment."""
    load_dotenv(override=False)

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    raw = _interpolate_env_vars(raw)

    bench_section = raw.get("benchmark", {})
    hydra_section = raw.get("hydradb", {})
    ingestion_section = raw.get("ingestion", {})
    testset_section = raw.get("testset_generation", {})
    hf_section = raw.get("hf_dataset", {})
    eval_section = raw.get("evaluation", {})
    ragas_section = raw.get("ragas", {})
    reporting_section = raw.get("reporting", {})
    slack_section = raw.get("slack", {})

    # Credentials must come from environment
    api_key = os.environ.get("HYDRADB_API_KEY", "")
    llm_provider = raw.get("ragas", {}).get("llm_provider", "openai")

    if not api_key:
        raise ValueError(
            "HYDRADB_API_KEY is not set. "
            "Copy .env.example to .env and fill in your credentials."
        )

    # Validate the LLM API key for whichever provider is selected
    if llm_provider == "openrouter":
        if not os.environ.get("OPENROUTER_API_KEY", ""):
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to your .env file to use OpenRouter as the judge LLM."
            )
        # Still need OpenAI key for embeddings (OpenRouter doesn't do embeddings)
        if not os.environ.get("OPENAI_API_KEY", ""):
            raise ValueError(
                "OPENAI_API_KEY is required for embeddings even when using OpenRouter. "
                "OpenRouter does not provide an embeddings endpoint."
            )
    else:
        if not os.environ.get("OPENAI_API_KEY", ""):
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "RAGAS requires OpenAI for LLM-based evaluation metrics."
            )

    hydra_config = HydraConfig(
        base_url=hydra_section.get("base_url", "https://api.hydradb.com"),
        api_key=api_key,
        timeout_seconds=hydra_section.get("timeout_seconds", 30),
        polling_interval_seconds=hydra_section.get("polling_interval_seconds", 5),
        max_polling_attempts=hydra_section.get("max_polling_attempts", 60),
        tenant_id=hydra_section.get("tenant_id", ""),
        sub_tenant_id=hydra_section.get("sub_tenant_id", ""),
        create_tenant_on_start=hydra_section.get("create_tenant_on_start", True),
    )

    ingestion_config = IngestionConfig(
        documents_dir=ingestion_section.get("documents_dir", "./data/documents"),
        file_extensions=ingestion_section.get("file_extensions", [".txt", ".pdf", ".md"]),
        verify_before_querying=ingestion_section.get("verify_before_querying", True),
        upload_delay_seconds=ingestion_section.get("upload_delay_seconds", 0.5),
    )

    testset_config = TestsetGenerationConfig(
        enabled=testset_section.get("enabled", False),
        output_path=testset_section.get("output_path", "./data/test_dataset.json"),
        testset_size=testset_section.get("testset_size", 15),
        query_distribution=testset_section.get("query_distribution", {
            "simple": 0.5, "multi_context": 0.3, "reasoning": 0.2,
        }),
    )

    hf_config = HFDatasetConfig(
        enabled=hf_section.get("enabled", False),
        mode=hf_section.get("mode", "qa"),
        repo_id=hf_section.get("repo_id", ""),
        split=hf_section.get("split", "test"),
        config_name=hf_section.get("config_name"),
        max_samples=hf_section.get("max_samples"),
        column_map=hf_section.get("column_map", {}),
        save_qa_path=hf_section.get("save_qa_path"),
        id_prefix=hf_section.get("id_prefix", "hf"),
        max_docs=hf_section.get("max_docs"),
        text_column=hf_section.get("text_column"),
        title_column=hf_section.get("title_column"),
        corpus_output_dir=hf_section.get("corpus_output_dir", "./data/hf_documents"),
    )

    # Parse aspect critics
    aspect_critics = [
        AspectCriticConfig(**ac)
        for ac in eval_section.get("aspect_critics", [])
    ]

    # Parse scored criteria (continuous 0-1 via SimpleCriteriaScore)
    scored_criteria = [
        ScoredCriteriaConfig(**sc)
        for sc in eval_section.get("scored_criteria", [])
    ]

    # Parse multi-turn config
    mt_section = eval_section.get("multi_turn", {})
    multi_turn_config = MultiTurnConfig(
        enabled=mt_section.get("enabled", False),
        dataset_path=mt_section.get("dataset_path", "./data/multi_turn_dataset.json"),
        metrics=mt_section.get("metrics", ["topic_adherence", "agent_goal_accuracy"]),
    )

    ag_section = eval_section.get("answer_generation", {})
    answer_gen_config = AnswerGenerationConfig(
        enabled=ag_section.get("enabled", False),
        provider=ag_section.get("provider", "openai"),
        model=ag_section.get("model", "gpt-4o-mini"),
        system_prompt=ag_section.get("system_prompt", AnswerGenerationConfig().system_prompt),
        max_tokens=ag_section.get("max_tokens", 1024),
        temperature=ag_section.get("temperature", 0.0),
    )

    eval_config = EvaluationConfig(
        test_dataset_path=eval_section.get("test_dataset_path", "./data/test_dataset.json"),
        search_endpoint=eval_section.get("search_endpoint", "qna"),
        retrieve_mode=eval_section.get("retrieve_mode", "fast"),
        max_results=eval_section.get("max_results", 5),
        concurrent_requests=eval_section.get("concurrent_requests", 3),
        metrics=eval_section.get("metrics", [
            "faithfulness", "response_relevancy", "context_precision",
            "context_recall", "factual_correctness",
        ]),
        aspect_critics=aspect_critics,
        scored_criteria=scored_criteria,
        multi_turn=multi_turn_config,
        answer_generation=answer_gen_config,
    )

    # Parse cost tracking
    cost_section = ragas_section.get("cost_tracking", {})
    cost_config = CostTrackingConfig(
        enabled=cost_section.get("enabled", True),
        cost_per_input_token=cost_section.get("cost_per_input_token", 0.00000015),
        cost_per_output_token=cost_section.get("cost_per_output_token", 0.0000006),
    )

    # Parse prompt overrides
    prompt_overrides = PromptOverridesConfig(
        overrides=ragas_section.get("prompt_overrides", {})
    )

    ragas_config = RAGASConfig(
        llm_model=ragas_section.get("llm_model", "gpt-4o-mini"),
        embeddings_model=ragas_section.get("embeddings_model", "text-embedding-3-small"),
        llm_provider=ragas_section.get("llm_provider", "openai"),
        embeddings_provider=ragas_section.get("embeddings_provider", "openai"),
        embeddings_base_url=ragas_section.get("embeddings_base_url"),
        max_retries=ragas_section.get("max_retries", 3),
        timeout=ragas_section.get("timeout", 60),
        max_workers=ragas_section.get("max_workers", 4),
        raise_exceptions=ragas_section.get("raise_exceptions", False),
        temperature=ragas_section.get("temperature", 0.0),
        cost_tracking=cost_config,
        prompt_overrides=prompt_overrides,
    )

    reporting_config = ReportingConfig(
        formats=reporting_section.get("formats", ["json", "csv", "html"]),
        include_per_sample_scores=reporting_section.get("include_per_sample_scores", True),
        include_reasons=reporting_section.get("include_reasons", True),
    )

    datasets = [
        DatasetEntry(**entry)
        for entry in raw.get("datasets", [])
    ]

    slack_config = SlackConfig(
        enabled=slack_section.get("enabled", False),
        bot_token=os.environ.get("SLACK_BOT_TOKEN", slack_section.get("bot_token", "")),
        user_id=slack_section.get("user_id", os.environ.get("SLACK_USER_ID", "")),
    )

    return BenchmarkConfig(
        name=bench_section.get("name", "HydraDB RAG Benchmark"),
        run_id=bench_section.get("run_id"),
        output_dir=bench_section.get("output_dir", "./reports"),
        hydradb=hydra_config,
        ingestion=ingestion_config,
        testset_generation=testset_config,
        hf_dataset=hf_config,
        evaluation=eval_config,
        ragas=ragas_config,
        reporting=reporting_config,
        slack=slack_config,
        datasets=datasets,
    )
