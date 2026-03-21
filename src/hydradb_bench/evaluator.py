"""RAGAS evaluation wrapper — full feature coverage for RAGAS 0.4.x."""

from __future__ import annotations

import logging
import warnings
from typing import Any

from .models import AspectCriticConfig, BenchmarkSample, RAGASConfig, TokenUsageResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Full metric registry
# Tuple: (import_path, needs_llm, needs_embeddings)
# All standard metrics use ragas.metrics.collections (RAGAS 0.4+)
# ---------------------------------------------------------------------------
_METRIC_REGISTRY: dict[str, tuple[str, bool, bool]] = {
    # --- Core RAG metrics ---
    "faithfulness":                     ("ragas.metrics._faithfulness.Faithfulness",                          True,  False),
    "response_relevancy":               ("ragas.metrics._answer_relevance.AnswerRelevancy",                   True,  True),
    "answer_relevancy":                 ("ragas.metrics._answer_relevance.AnswerRelevancy",                   True,  True),  # alias
    # --- Context metrics ---
    "context_precision":                ("ragas.metrics._context_precision.LLMContextPrecisionWithReference", True,  False),
    "context_precision_no_ref":         ("ragas.metrics._context_precision.LLMContextPrecisionWithoutReference", True, False),
    "context_recall":                   ("ragas.metrics._context_recall.LLMContextRecall",                    True,  False),
    "non_llm_context_recall":           ("ragas.metrics._context_recall.NonLLMContextRecall",                 False, False),
    "context_relevance":                ("ragas.metrics._nv_metrics.ContextRelevance",                        True,  False),
    "context_utilization":              ("ragas.metrics._context_precision.ContextUtilization",               True,  False),
    "context_entity_recall":            ("ragas.metrics._context_entities_recall.ContextEntityRecall",        True,  False),
    "noise_sensitivity":                ("ragas.metrics._noise_sensitivity.NoiseSensitivity",                 True,  False),
    # --- Answer quality ---
    "factual_correctness":              ("ragas.metrics._factual_correctness.FactualCorrectness",             True,  False),
    "answer_correctness":               ("ragas.metrics._answer_correctness.AnswerCorrectness",               True,  True),
    "answer_accuracy":                  ("ragas.metrics._nv_metrics.AnswerAccuracy",                          True,  False),
    "response_groundedness":            ("ragas.metrics._nv_metrics.ResponseGroundedness",                    True,  False),
    # --- Summarisation ---
    "summary_score":                    ("ragas.metrics._summarization.SummarizationScore",                   True,  False),
    # --- Semantic / embedding-based (no LLM call) ---
    "semantic_similarity":              ("ragas.metrics._answer_similarity.SemanticSimilarity",               False, True),
    "non_llm_string_similarity":        ("ragas.metrics._string.NonLLMStringSimilarity",                      False, False),
    # --- String-overlap (fully offline) ---
    "bleu_score":                       ("ragas.metrics._bleu_score.BleuScore",                               False, False),
    "rouge_score":                      ("ragas.metrics._rouge_score.RougeScore",                             False, False),
    "chrf_score":                       ("ragas.metrics._chrf_score.ChrfScore",                               False, False),
    "exact_match":                      ("ragas.metrics._string.ExactMatch",                                  False, False),
    "string_presence":                  ("ragas.metrics._string.StringPresence",                              False, False),
    # --- Agent / tool-use metrics (multi-turn capable) ---
    "agent_goal_accuracy":              ("ragas.metrics._goal_accuracy.AgentGoalAccuracyWithReference",       True,  False),
    "agent_goal_accuracy_no_ref":       ("ragas.metrics._goal_accuracy.AgentGoalAccuracyWithoutReference",    True,  False),
    "tool_call_accuracy":               ("ragas.metrics._tool_call_accuracy.ToolCallAccuracy",                True,  False),
    "tool_call_f1":                     ("ragas.metrics._tool_call_f1.ToolCallF1",                            True,  False),
    "topic_adherence":                  ("ragas.metrics._topic_adherence.TopicAdherenceScore",                True,  False),
    # --- Rubric-based scoring ---
    "rubrics_score":                    ("ragas.metrics._domain_specific_rubrics.RubricsScore",               True,  False),
    "instance_rubrics":                 ("ragas.metrics._instance_specific_rubrics.InstanceRubrics",          True,  False),
    "simple_criteria":                  ("ragas.metrics._simple_criteria.SimpleCriteriaScore",                True,  False),
}


def _import_metric(class_path: str) -> type:
    """Dynamically import a metric class by dotted path."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


class RAGASEvaluator:
    """
    Full-featured RAGAS evaluator supporting:
    - 30+ metrics
    - AspectCritic (custom-defined criteria)
    - Cost / token usage tracking
    - Prompt customization per metric
    - @experiment decorator for run tracking
    - Multi-turn evaluation via MultiTurnSample
    """

    def __init__(self, config: RAGASConfig) -> None:
        self.config = config
        self._llm = None
        self._embeddings = None

    # ------------------------------------------------------------------
    # LLM / embeddings accessors (lazy-init)
    # ------------------------------------------------------------------

    _PROVIDER_BASE_URLS = {
        "openai":      None,                           # default OpenAI base URL
        "openrouter":  "https://openrouter.ai/api/v1",
    }
    _PROVIDER_ENV_KEYS = {
        "openai":      "OPENAI_API_KEY",
        "openrouter":  "OPENROUTER_API_KEY",
    }

    def _make_openai_client(self, provider: str):
        import os
        from openai import OpenAI
        env_key = self._PROVIDER_ENV_KEYS.get(provider, "OPENAI_API_KEY")
        base_url = self._PROVIDER_BASE_URLS.get(provider)
        api_key = os.environ.get(env_key, "")
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

    def _get_llm(self):
        if self._llm is None:
            from ragas.llms import llm_factory
            provider = self.config.llm_provider
            client = self._make_openai_client(provider)
            self._llm = llm_factory(self.config.llm_model, client=client)
        return self._llm

    def _get_embeddings(self):
        if self._embeddings is None:
            self._embeddings = self._build_embeddings()
        return self._embeddings

    def _build_embeddings(self):
        """
        Build a RAGAS-compatible embeddings object based on embeddings_provider:

          openai       — OpenAI text-embedding-* via API (needs OPENAI_API_KEY)
          openrouter   — OpenRouter-hosted embeddings via OpenAI-compat API
                         (needs OPENROUTER_API_KEY; set embeddings_model to the
                          full OpenRouter model ID, e.g. "openai/text-embedding-3-small")
          huggingface  — sentence-transformers running locally (no API key)
                         e.g. embeddings_model: "BAAI/bge-small-en-v1.5"
          ollama       — Ollama local server (no API key)
                         e.g. embeddings_model: "nomic-embed-text"
                         Override server URL with embeddings_base_url in config.
        """
        provider = self.config.embeddings_provider
        model = self.config.embeddings_model

        if provider in ("openai", "openrouter"):
            import os
            from langchain_openai import OpenAIEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper
            env_key = self._PROVIDER_ENV_KEYS.get(provider, "OPENAI_API_KEY")
            base_url = self._PROVIDER_BASE_URLS.get(provider)
            kwargs = {"model": model, "api_key": os.environ.get(env_key, "")}
            if base_url:
                kwargs["base_url"] = base_url
            return LangchainEmbeddingsWrapper(OpenAIEmbeddings(**kwargs))

        if provider == "huggingface":
            lc_emb = self._make_hf_embeddings(model)
            try:
                from ragas.embeddings import LangchainEmbeddingsWrapper
                return LangchainEmbeddingsWrapper(lc_emb)
            except ImportError:
                return lc_emb

        if provider == "ollama":
            base_url = self.config.embeddings_base_url or "http://localhost:11434"
            lc_emb = self._make_ollama_embeddings(model, base_url)
            try:
                from ragas.embeddings import LangchainEmbeddingsWrapper
                return LangchainEmbeddingsWrapper(lc_emb)
            except ImportError:
                # Older RAGAS versions — embeddings inherit from LC Embeddings directly
                return lc_emb

        raise ValueError(
            f"Unknown embeddings_provider: {provider!r}. "
            f"Supported: openai, openrouter, huggingface, ollama"
        )

    @staticmethod
    def _make_hf_embeddings(model: str):
        """Return a LangChain-compatible HuggingFace sentence-transformers embeddings object."""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model)
        except ImportError:
            pass
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model)
        except ImportError:
            raise ImportError(
                "langchain-huggingface (or langchain-community) is required for HuggingFace embeddings. "
                "Install with: pip install langchain-huggingface sentence-transformers"
            )

    @staticmethod
    def _make_ollama_embeddings(model: str, base_url: str):
        """Return a LangChain-compatible Ollama embeddings object."""
        try:
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(model=model, base_url=base_url)
        except ImportError:
            pass
        try:
            from langchain_community.embeddings import OllamaEmbeddings
            return OllamaEmbeddings(model=model, base_url=base_url)
        except ImportError:
            raise ImportError(
                "langchain-ollama (or langchain-community) is required for Ollama embeddings. "
                "Install with: pip install langchain-ollama"
            )

    # ------------------------------------------------------------------
    # Metric construction
    # ------------------------------------------------------------------

    def build_metrics(self, metric_names: list[str]) -> list:
        """Instantiate standard RAGAS metric objects from config name list."""
        metrics = []
        for name in metric_names:
            entry = _METRIC_REGISTRY.get(name)
            if entry is None:
                logger.warning("Unknown metric '%s', skipping.", name)
                continue
            class_path, needs_llm, needs_embeddings = entry
            try:
                cls = _import_metric(class_path)
                kwargs: dict[str, Any] = {}
                if needs_llm:
                    kwargs["llm"] = self._get_llm()
                if needs_embeddings:
                    kwargs["embeddings"] = self._get_embeddings()
                metric_instance = cls(**kwargs)

                # Force metric name to match the configured name so RAGAS
                # uses it as the DataFrame column (e.g. "response_relevancy"
                # instead of RAGAS's internal "answer_relevancy").
                if hasattr(metric_instance, "name"):
                    metric_instance.name = name

                # Apply prompt override if configured
                override = self.config.prompt_overrides.overrides.get(name)
                if override:
                    self._apply_prompt_override(metric_instance, name, override)

                metrics.append(metric_instance)
                logger.info("Loaded metric: %s", name)
            except Exception as e:
                logger.warning("Could not load metric '%s' (%s): %s", name, class_path, e)
        return metrics

    def build_aspect_critics(self, aspect_configs: list[AspectCriticConfig]) -> list:
        """Build AspectCritic metric instances from config."""
        critics = []
        for ac in aspect_configs:
            try:
                from ragas.metrics._aspect_critic import AspectCritic
                critic = AspectCritic(
                    name=ac.name,
                    definition=ac.definition,
                    llm=self._get_llm(),
                    strictness=ac.strictness,
                )
                critics.append(critic)
                logger.info("Loaded AspectCritic: %s", ac.name)
            except Exception as e:
                logger.warning("Could not load AspectCritic '%s': %s", ac.name, e)
        return critics

    def _apply_prompt_override(self, metric_instance: Any, name: str, instruction: str) -> None:
        """Override the instruction text on a metric's prompts via PromptMixin API."""
        try:
            if hasattr(metric_instance, "get_prompts"):
                prompts = metric_instance.get_prompts()
                for prompt_name, prompt_obj in prompts.items():
                    if hasattr(prompt_obj, "instruction"):
                        prompt_obj.instruction = instruction
                        logger.info("Applied prompt override to %s.%s", name, prompt_name)
        except Exception as e:
            logger.warning("Could not apply prompt override to '%s': %s", name, e)

    def adapt_prompts_to_language(self, metric_instances: list, language: str) -> None:
        """Adapt all metric prompts to a target language using the LLM."""
        import asyncio
        for metric in metric_instances:
            if hasattr(metric, "adapt_prompts"):
                try:
                    asyncio.get_event_loop().run_until_complete(
                        metric.adapt_prompts(language=language, llm=self._get_llm())
                    )
                    logger.info("Adapted %s prompts to %s", metric.__class__.__name__, language)
                except Exception as e:
                    logger.warning("Could not adapt prompts for %s: %s", metric.__class__.__name__, e)

    # ------------------------------------------------------------------
    # Dataset building
    # ------------------------------------------------------------------

    def build_dataset(self, samples: list[BenchmarkSample]):
        """Convert BenchmarkSamples → RAGAS EvaluationDataset (single-turn)."""
        from ragas import EvaluationDataset, SingleTurnSample

        valid_samples = []
        skipped = 0
        for bs in samples:
            if bs.error or bs.hydra_result is None:
                skipped += 1
                continue
            valid_samples.append(SingleTurnSample(
                user_input=bs.test_sample.question,
                response=bs.hydra_result.answer,
                retrieved_contexts=bs.hydra_result.retrieved_contexts or [""],
                reference=bs.test_sample.reference_answer,
                reference_contexts=bs.test_sample.reference_contexts or None,
            ))

        if skipped:
            logger.warning("Skipped %d errored samples.", skipped)
        logger.info("Built EvaluationDataset with %d samples.", len(valid_samples))
        return EvaluationDataset(samples=valid_samples)

    def build_multi_turn_dataset(self, conversations: list[dict[str, Any]]):
        """Convert multi-turn conversation dicts → RAGAS EvaluationDataset."""
        from ragas import EvaluationDataset
        from ragas.messages import AIMessage, HumanMessage

        ragas_samples = []
        for conv in conversations:
            try:
                from ragas import MultiTurnSample
            except ImportError:
                from ragas.dataset_schema import MultiTurnSample

            messages = []
            for turn in conv.get("conversation_turns", []):
                role = turn.get("role", "human")
                content = turn.get("content", "")
                if role == "human":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(AIMessage(content=content))

            if not messages:
                continue

            sample = MultiTurnSample(
                user_input=messages,
                reference=conv.get("reference", ""),
                reference_topics=conv.get("reference_topics") or None,
            )
            ragas_samples.append(sample)

        return EvaluationDataset(samples=ragas_samples)

    # ------------------------------------------------------------------
    # Core evaluation (with @experiment decorator + cost tracking)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        samples: list[BenchmarkSample],
        metric_names: list[str],
        aspect_configs: list[AspectCriticConfig] | None = None,
    ) -> tuple[dict[str, float], list[dict[str, Any]], TokenUsageResult]:
        """
        Run single-turn RAGAS evaluation.

        Returns:
            aggregate_scores  : {metric_name: mean_score}
            per_sample_scores : one dict per evaluated sample
            token_usage       : token counts + estimated cost
        """
        import nest_asyncio
        nest_asyncio.apply()

        dataset = self.build_dataset(samples)
        if not dataset.samples:
            logger.error("No valid samples to evaluate.")
            return {}, [], TokenUsageResult()

        standard_metrics = self.build_metrics(metric_names)
        critic_metrics = self.build_aspect_critics(aspect_configs or [])
        all_metrics = standard_metrics + critic_metrics

        if not all_metrics:
            logger.error("No metrics could be loaded.")
            return {}, [], TokenUsageResult()

        result = self._run_evaluate(dataset, all_metrics)

        aggregate_scores = self._extract_aggregate_scores(result)
        per_sample_scores = self._extract_per_sample_scores(result, samples)
        token_usage = self._extract_token_usage(result)

        return aggregate_scores, per_sample_scores, token_usage

    def evaluate_multi_turn(
        self,
        conversations: list[dict[str, Any]],
        metric_names: list[str],
    ) -> tuple[dict[str, float], TokenUsageResult]:
        """Run multi-turn RAGAS evaluation."""
        import nest_asyncio
        nest_asyncio.apply()

        dataset = self.build_multi_turn_dataset(conversations)
        if not dataset.samples:
            logger.error("No valid multi-turn samples to evaluate.")
            return {}, TokenUsageResult()

        metrics = self.build_metrics(metric_names)
        if not metrics:
            return {}, TokenUsageResult()

        result = self._run_evaluate(dataset, metrics)
        aggregate_scores = self._extract_aggregate_scores(result)
        token_usage = self._extract_token_usage(result)
        return aggregate_scores, token_usage

    def _run_evaluate(self, dataset, metrics):
        """Internal: call ragas.evaluate() with full options."""
        import asyncio
        import inspect
        from ragas import RunConfig, evaluate
        from ragas.cost import get_token_usage_for_openai

        run_config = RunConfig(
            max_retries=self.config.max_retries,
            timeout=self.config.timeout,
            max_workers=self.config.max_workers,
        )
        token_parser = get_token_usage_for_openai if self.config.cost_tracking.enabled else None

        def _call_evaluate():
            return evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self._get_llm(),
                embeddings=self._get_embeddings(),
                run_config=run_config,
                raise_exceptions=self.config.raise_exceptions,
                token_usage_parser=token_parser,
                show_progress=True,
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            result = _call_evaluate()

            # RAGAS 0.4.x evaluate() is async — run it if we got a coroutine back
            if inspect.isawaitable(result):
                try:
                    loop = asyncio.get_event_loop()
                    result = loop.run_until_complete(result)
                except RuntimeError:
                    result = asyncio.run(result)

        return result

    def _make_experiment_fn(self, dataset, metrics, run_config):
        """Wrap evaluate() with RAGAS @experiment decorator for tracked runs."""
        from ragas import evaluate
        from ragas.cost import get_token_usage_for_openai

        token_parser = get_token_usage_for_openai if self.config.cost_tracking.enabled else None

        try:
            from ragas.experiment import experiment

            @experiment(name_prefix="hydradb-bench")
            def _eval_fn():
                return evaluate(
                    dataset=dataset,
                    metrics=metrics,
                    llm=self._get_llm(),
                    embeddings=self._get_embeddings(),
                    run_config=run_config,
                    raise_exceptions=self.config.raise_exceptions,
                    token_usage_parser=token_parser,
                    show_progress=True,
                )

            return _eval_fn
        except Exception:
            # @experiment not available or not compatible — return plain callable
            def _plain_fn():
                return evaluate(
                    dataset=dataset,
                    metrics=metrics,
                    llm=self._get_llm(),
                    embeddings=self._get_embeddings(),
                    run_config=run_config,
                    raise_exceptions=self.config.raise_exceptions,
                    token_usage_parser=token_parser,
                    show_progress=True,
                )
            return _plain_fn

    # ------------------------------------------------------------------
    # Result extraction helpers
    # ------------------------------------------------------------------

    def _extract_aggregate_scores(self, result) -> dict[str, float]:
        aggregate: dict[str, float] = {}
        skip = {"user_input", "response", "retrieved_contexts",
                "reference", "reference_contexts"}
        try:
            df = result.to_pandas()
            for col in df.columns:
                if col in skip:
                    continue
                series = df[col].dropna()
                if not series.empty:
                    aggregate[col] = round(float(series.mean()), 4)
        except Exception as e:
            logger.warning("Could not extract aggregate scores: %s", e)
            if hasattr(result, "scores"):
                for k, v in result.scores.items():
                    if v is not None:
                        aggregate[str(k)] = round(float(v), 4)
        return aggregate

    def _extract_per_sample_scores(
        self, result, samples: list[BenchmarkSample]
    ) -> list[dict[str, Any]]:
        skip = {"user_input", "response", "retrieved_contexts",
                "reference", "reference_contexts"}
        per_sample: list[dict[str, Any]] = []
        valid = [bs for bs in samples if not bs.error and bs.hydra_result]

        try:
            df = result.to_pandas()
            metric_cols = [c for c in df.columns if c not in skip]
            for i, row in df.iterrows():
                entry: dict[str, Any] = {}
                if i < len(valid):
                    bs = valid[i]
                    entry["sample_id"] = bs.test_sample.id
                    entry["question"] = bs.test_sample.question
                    entry["reference_answer"] = bs.test_sample.reference_answer
                    entry["hydra_answer"] = bs.hydra_result.answer
                    entry["latency_ms"] = round(bs.latency_ms, 2)
                    entry["contexts_retrieved"] = len(bs.hydra_result.retrieved_contexts)
                for col in metric_cols:
                    val = row.get(col)
                    entry[col] = (
                        round(float(val), 4)
                        if val is not None and val == val  # NaN check
                        else None
                    )
                per_sample.append(entry)
        except Exception as e:
            logger.warning("Could not extract per-sample scores: %s", e)
        return per_sample

    def _extract_token_usage(self, result) -> TokenUsageResult:
        if not self.config.cost_tracking.enabled:
            return TokenUsageResult()
        try:
            usage_list = result.total_tokens()
            # total_tokens() can return a single TokenUsage or a list
            if not isinstance(usage_list, list):
                usage_list = [usage_list]

            total_input = sum(getattr(u, "input_tokens", 0) for u in usage_list)
            total_output = sum(getattr(u, "output_tokens", 0) for u in usage_list)
            total = total_input + total_output

            cfg = self.config.cost_tracking
            cost = (
                total_input * cfg.cost_per_input_token
                + total_output * cfg.cost_per_output_token
            )
            model = getattr(usage_list[0], "model", self.config.llm_model) if usage_list else self.config.llm_model

            return TokenUsageResult(
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total,
                estimated_cost_usd=round(cost, 6),
                model=model,
            )
        except Exception as e:
            logger.warning("Could not extract token usage: %s", e)
            return TokenUsageResult()

    # ------------------------------------------------------------------
    # Prompt management utilities (exposed for CLI use)
    # ------------------------------------------------------------------

    def save_prompts(self, metric_names: list[str], output_dir: str) -> None:
        """Save all metric prompts to disk for inspection/customization."""
        import json
        from pathlib import Path
        out = Path(output_dir) / "prompts"
        out.mkdir(parents=True, exist_ok=True)
        for name in metric_names:
            metrics = self.build_metrics([name])
            for metric in metrics:
                if hasattr(metric, "get_prompts"):
                    prompts = metric.get_prompts()
                    for pname, pobj in prompts.items():
                        data = {"instruction": getattr(pobj, "instruction", "")}
                        (out / f"{name}_{pname}.json").write_text(
                            json.dumps(data, indent=2), encoding="utf-8"
                        )
                        logger.info("Saved prompt: %s/%s_%s.json", out, name, pname)

    def load_prompts(self, metric_instances: list, prompts_dir: str) -> None:
        """Load saved prompt overrides from disk and apply them."""
        import json
        from pathlib import Path
        prompts_path = Path(prompts_dir)
        for metric in metric_instances:
            name = getattr(metric, "name", metric.__class__.__name__.lower())
            if hasattr(metric, "get_prompts"):
                prompts = metric.get_prompts()
                for pname, pobj in prompts.items():
                    fpath = prompts_path / f"{name}_{pname}.json"
                    if fpath.exists():
                        data = json.loads(fpath.read_text())
                        if "instruction" in data and hasattr(pobj, "instruction"):
                            pobj.instruction = data["instruction"]
                            logger.info("Loaded prompt override: %s", fpath)
