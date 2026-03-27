"""RAGAS evaluation wrapper — full feature coverage for RAGAS 0.4.x."""

from __future__ import annotations

import logging
import warnings
from typing import Any

from .models import AspectCriticConfig, BenchmarkSample, RAGASConfig, ScoredCriteriaConfig, TokenUsageResult

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
        self._scored_criteria_names: set[str] = set()  # names that need 0-3 → 0-1 normalization

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
            import os
            from langchain_openai import ChatOpenAI
            from ragas.llms import LangchainLLMWrapper

            provider = self.config.llm_provider
            env_key = self._PROVIDER_ENV_KEYS.get(provider, "OPENAI_API_KEY")
            base_url = self._PROVIDER_BASE_URLS.get(provider)
            api_key = os.environ.get(env_key, "")

            base_kwargs = dict(
                model=self.config.llm_model,
                api_key=api_key,
                max_tokens=self.config.judge_max_tokens,
                temperature=self.config.temperature,
            )
            if base_url:
                base_kwargs["base_url"] = base_url

            llm = ChatOpenAI(**base_kwargs)
            # bypass_n=True: RAGAS sends n separate prompts instead of requesting
            # n completions in one call — works around OpenRouter not supporting n>1.
            self._llm = LangchainLLMWrapper(llm, bypass_n=True)
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

    def build_scored_criteria(self, scored_configs: list[ScoredCriteriaConfig]) -> list:
        """Build SimpleCriteriaScore metric instances — raw 0-3 output, normalized to 0-1 on extraction."""
        metrics = []
        for sc in scored_configs:
            try:
                from ragas.metrics._simple_criteria import SimpleCriteriaScore
                metric = SimpleCriteriaScore(
                    name=sc.name,
                    definition=sc.definition,
                    llm=self._get_llm(),
                )
                metrics.append(metric)
                # Track names so extraction can normalize 0-3 → 0-1
                self._scored_criteria_names.add(sc.name)
                logger.info("Loaded ScoredCriteria: %s", sc.name)
            except Exception as e:
                logger.warning("Could not load ScoredCriteria '%s': %s", sc.name, e)
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
        scored_criteria_configs: list[ScoredCriteriaConfig] | None = None,
        include_reasons: bool = True,
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
        scored_metrics = self.build_scored_criteria(scored_criteria_configs or [])
        all_metrics = standard_metrics + critic_metrics + scored_metrics

        if not all_metrics:
            logger.error("No metrics could be loaded.")
            return {}, [], TokenUsageResult()

        result = self._run_evaluate(dataset, all_metrics)

        aggregate_scores = self._extract_aggregate_scores(result)
        per_sample_scores = self._extract_per_sample_scores(result, samples, include_reasons)
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

    @staticmethod
    def _openrouter_token_parser(llm_result):
        """Token parser that uses OpenRouter's exact cost when available,
        falling back to standard OpenAI token counts."""
        from ragas.cost import TokenUsage

        llm_output = getattr(llm_result, "llm_output", None) or {}
        token_usage = llm_output.get("token_usage", {}) or {}
        input_tokens  = token_usage.get("prompt_tokens", 0) or 0
        output_tokens = token_usage.get("completion_tokens", 0) or 0
        model = llm_output.get("model_name", "")

        # OpenRouter returns the actual upstream cost in cost_details
        cost_details = token_usage.get("cost_details", {}) or {}
        exact_cost = cost_details.get("upstream_inference_cost") or token_usage.get("cost") or None

        usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens, model=model)
        # Stash exact cost as a custom attribute so _extract_token_usage can use it
        if exact_cost is not None:
            usage._exact_cost_usd = float(exact_cost)
        return usage

    def _run_evaluate(self, dataset, metrics):
        """Internal: call ragas.evaluate() with full options."""
        import asyncio
        import inspect
        from ragas import RunConfig, evaluate

        run_config = RunConfig(
            max_retries=self.config.max_retries,
            timeout=self.config.timeout,
            max_workers=self.config.max_workers,
        )
        token_parser = self._openrouter_token_parser if self.config.cost_tracking.enabled else None

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
                    val = float(series.mean())
                    if col in self._scored_criteria_names:
                        val = val / 3.0  # SimpleCriteriaScore returns 0-3; normalize to 0-1
                    aggregate[col] = round(val, 4)
        except Exception as e:
            logger.warning("Could not extract aggregate scores: %s", e)
            if hasattr(result, "scores"):
                for k, v in result.scores.items():
                    if v is not None:
                        aggregate[str(k)] = round(float(v), 4)
        return aggregate

    def _extract_reasons_from_traces(self, result) -> dict[int, dict[str, str]]:
        """Extract AspectCritic reasons from ragas_traces.
        RAGAS stores reasons in the trace tree but does not expose them in the DataFrame.
        Structure: evaluation -> row N -> metric_name -> prompt_leaf (has reason in outputs)
        """
        reasons: dict[int, dict[str, str]] = {}
        try:
            traces = getattr(result, "ragas_traces", {}) or {}
            for run in traces.values():
                name = getattr(run, "name", "") or ""
                if not name.startswith("row "):
                    continue
                try:
                    row_index = int(name.split("row ")[1])
                except (IndexError, ValueError):
                    continue
                reasons.setdefault(row_index, {})
                for metric_run_id in (getattr(run, "children", None) or []):
                    metric_run = traces.get(metric_run_id)
                    if not metric_run:
                        continue
                    metric_name = getattr(metric_run, "name", "") or ""
                    for prompt_run_id in (getattr(metric_run, "children", None) or []):
                        prompt_run = traces.get(prompt_run_id)
                        if not prompt_run:
                            continue
                        outputs = getattr(prompt_run, "outputs", {}) or {}
                        output_list = outputs.get("output", [])
                        if isinstance(output_list, list) and output_list:
                            reason = getattr(output_list[0], "reason", None)
                            if reason:
                                reasons[row_index][f"{metric_name}_reason"] = str(reason)
        except Exception as e:
            logger.warning("Could not extract reasons from traces: %s", e)
        return reasons

    def _extract_per_sample_scores(
        self, result, samples: list[BenchmarkSample], include_reasons: bool = True
    ) -> list[dict[str, Any]]:
        skip = {"user_input", "response", "retrieved_contexts",
                "reference", "reference_contexts"}
        per_sample: list[dict[str, Any]] = []
        valid = [bs for bs in samples if not bs.error and bs.hydra_result]

        # Extract reasons from traces (AspectCritic reasons are not in the DataFrame)
        trace_reasons = self._extract_reasons_from_traces(result) if include_reasons else {}

        try:
            df = result.to_pandas()
            all_cols = [c for c in df.columns if c not in skip]
            reason_cols = [c for c in all_cols if c.endswith("_reason")]
            score_cols  = [c for c in all_cols if not c.endswith("_reason")]

            for i, row in df.iterrows():
                entry: dict[str, Any] = {}
                if i < len(valid):
                    bs = valid[i]
                    entry["sample_id"] = bs.test_sample.id
                    entry["question"] = bs.test_sample.question
                    entry["reference_answer"] = bs.test_sample.reference_answer
                    entry["hydra_answer"] = bs.hydra_result.answer
                    entry["context_string"] = bs.hydra_result.context_string
                    entry["context_tokens"] = bs.hydra_result.context_tokens
                    entry["latency_ms"] = round(bs.latency_ms, 2)
                    entry["contexts_retrieved"] = len(bs.hydra_result.retrieved_contexts)
                for col in score_cols:
                    val = row.get(col)
                    if val is not None and val == val:  # NaN check
                        fval = float(val)
                        if col in self._scored_criteria_names:
                            fval = fval / 3.0  # normalize 0-3 → 0-1
                        entry[col] = round(fval, 4)
                    else:
                        entry[col] = None
                if include_reasons:
                    # Reasons from DataFrame columns (standard metrics)
                    for col in reason_cols:
                        val = row.get(col)
                        entry[col] = str(val).strip() if val is not None and val == val else None
                    # Reasons from traces (AspectCritic)
                    for col, reason in trace_reasons.get(i, {}).items():
                        entry[col] = reason
                per_sample.append(entry)
        except Exception as e:
            logger.warning("Could not extract per-sample scores: %s", e)
        return per_sample

    def _extract_token_usage(self, result) -> TokenUsageResult:
        if not self.config.cost_tracking.enabled:
            return TokenUsageResult()
        try:
            usage_list = result.total_tokens()
            if usage_list is None:
                return TokenUsageResult()
            # total_tokens() can return a single TokenUsage or a list
            if not isinstance(usage_list, list):
                usage_list = [usage_list]
            usage_list = [u for u in usage_list if u is not None]
            if not usage_list:
                return TokenUsageResult()

            total_input = sum(getattr(u, "input_tokens", 0) for u in usage_list)
            total_output = sum(getattr(u, "output_tokens", 0) for u in usage_list)
            total = total_input + total_output

            cfg = self.config.cost_tracking
            # Use exact cost from OpenRouter if available, otherwise estimate from token rates
            exact_costs = [getattr(u, "_exact_cost_usd", None) for u in usage_list]
            if any(c is not None for c in exact_costs):
                cost = sum(c for c in exact_costs if c is not None)
            else:
                cost = (
                    total_input * cfg.cost_per_input_token
                    + total_output * cfg.cost_per_output_token
                )
            model = getattr(usage_list[0], "model", self.config.llm_model) if usage_list else self.config.llm_model

            return TokenUsageResult(
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total,
                actual_cost_usd=round(cost, 6),
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
