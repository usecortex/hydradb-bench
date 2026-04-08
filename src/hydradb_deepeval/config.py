from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .models import (
    BenchmarkConfig,
    DeepEvalConfig,
    EvaluationConfig,
    HydraConfig,
    IngestionConfig,
    ReportingConfig,
    SupermemoryConfig,
)

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _interpolate(value: Any) -> Any:
    """Recursively replace ${ENV_VAR} references with environment values."""
    if isinstance(value, str):

        def _replace(match: re.Match) -> str:
            var = match.group(1)
            env_val = os.environ.get(var)
            if env_val is None:
                raise OSError(f"Environment variable '{var}' is referenced in config but not set.")
            return env_val

        return _ENV_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(item) for item in value]
    return value


def load_config(config_path: str = "config/benchmark.yaml") -> BenchmarkConfig:
    """Load and validate BenchmarkConfig from a YAML file.

    Environment variable lookup order for secrets:
    1. Real env vars
    2. .env file (auto-loaded from cwd)
    """
    load_dotenv()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    raw: dict = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = _interpolate(raw)

    # Gather required secrets from environment
    api_key = os.environ.get("HYDRADB_API_KEY", "")
    if not api_key:
        raise OSError("HYDRADB_API_KEY environment variable is not set. Set it in .env or your shell before running.")

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        raise OSError(
            "OPENAI_API_KEY environment variable is not set. "
            "DeepEval requires OpenAI access. "
            "Set it in .env or your shell before running."
        )

    # Inject the OpenAI key so DeepEval picks it up automatically
    os.environ.setdefault("OPENAI_API_KEY", openai_key)

    hydra_raw: dict = raw.get("hydradb", {})
    hydra_raw["api_key"] = api_key  # Always inject from env

    hydradb = HydraConfig(**hydra_raw)
    ingestion = IngestionConfig(**raw.get("ingestion", {}))
    evaluation = EvaluationConfig(**raw.get("evaluation", {}))
    deepeval = DeepEvalConfig(**raw.get("deepeval", {}))
    reporting = ReportingConfig(**raw.get("reporting", {}))

    # Supermemory is optional — only loaded when the section exists in YAML
    supermemory: SupermemoryConfig | None = None
    sm_raw: dict = raw.get("supermemory", {})
    if sm_raw:
        sm_api_key = os.environ.get("SUPERMEMORY_API_KEY", "")
        if not sm_api_key:
            raise OSError(
                "supermemory section is present in config but SUPERMEMORY_API_KEY "
                "environment variable is not set. Set it in .env or your shell."
            )
        sm_raw["api_key"] = sm_api_key
        supermemory = SupermemoryConfig(**sm_raw)

    benchmark_raw = raw.get("benchmark", {})
    name = benchmark_raw.get("name", "HydraDB DeepEval Benchmark")

    return BenchmarkConfig(
        name=name,
        hydradb=hydradb,
        ingestion=ingestion,
        evaluation=evaluation,
        deepeval=deepeval,
        reporting=reporting,
        supermemory=supermemory,
    )
