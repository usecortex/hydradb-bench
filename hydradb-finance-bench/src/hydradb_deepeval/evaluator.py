from __future__ import annotations

import asyncio
import importlib
import re
from statistics import mean
from typing import Any

from rich.console import Console

from .models import DeepEvalConfig, QueryResult, SampleScore

console = Console()

_METRIC_REGISTRY: dict[str, str] = {
    "answer_relevancy": "deepeval.metrics.AnswerRelevancyMetric",
    "faithfulness": "deepeval.metrics.FaithfulnessMetric",
    "contextual_precision": "deepeval.metrics.ContextualPrecisionMetric",
    "contextual_recall": "deepeval.metrics.ContextualRecallMetric",
    "contextual_relevancy": "deepeval.metrics.ContextualRelevancyMetric",
    "hallucination": "deepeval.metrics.HallucinationMetric",
    "bias": "deepeval.metrics.BiasMetric",
    "toxicity": "deepeval.metrics.ToxicityMetric",
    "summarization": "deepeval.metrics.SummarizationMetric",
    # Ragas-backed wrappers — looser than DeepEval's native metrics.
    # "ragas_context_precision" is the closest analog to DeepEval's contextual_relevancy;
    # DeepEval does not ship a RAGASContextualRelevancyMetric wrapper.
    "ragas_context_recall": "deepeval.metrics.ragas.RAGASContextualRecallMetric",
    "ragas_context_precision": "deepeval.metrics.ragas.RAGASContextualPrecisionMetric",
    "ragas_answer_relevancy": "deepeval.metrics.ragas.RAGASAnswerRelevancyMetric",
    "ragas_faithfulness": "deepeval.metrics.ragas.RAGASFaithfulnessMetric",
}

# GEval criteria for answer_accuracy
_ANSWER_ACCURACY_CRITERIA = (
    "Determine whether the actual output is factually correct and complete "
    "when compared to the expected output (reference answer). "
    "Award full marks if all key facts match. Deduct for missing facts, "
    "incorrect facts, or contradictions. Ignore differences in phrasing or style."
)

# ─── Deterministic recall@k (token-overlap) ──────────────────────────────────
# A reference passage matches a retrieved chunk if they share enough unigram
# tokens. The check is bidirectional: either (a) the chunk contains ≥ 50% of
# the reference's tokens (chunk is larger and contains the ref), or (b) ≥ 50%
# of the chunk's tokens appear in the reference (chunk is a subset of a
# larger ref — common in FinanceBench where refs are full-table excerpts).
# Also requires a small absolute overlap to avoid trivial stopword matches.
_RECALL_AT_K_PATTERN = re.compile(r"^recall_at_(\d+)$")
_RECALL_MATCH_THRESHOLD = 0.5
_RECALL_MIN_SHARED_TOKENS = 5
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _reference_covered_by(ref: str, retrieved: str) -> bool:
    ref_tokens = _tokenize(ref)
    ret_tokens = _tokenize(retrieved)
    if not ref_tokens or not ret_tokens:
        return False
    shared = ref_tokens & ret_tokens
    if len(shared) < _RECALL_MIN_SHARED_TOKENS:
        return False
    coverage = max(len(shared) / len(ref_tokens), len(shared) / len(ret_tokens))
    return coverage >= _RECALL_MATCH_THRESHOLD


def _compute_recall_at_k(refs: list[str], retrieved: list[str], k: int) -> tuple[float | None, str]:
    if not refs:
        return None, "no reference_contexts on sample"
    top_k = retrieved[:k]
    matched = sum(1 for r in refs if any(_reference_covered_by(r, c) for c in top_k))
    score = matched / len(refs)
    return score, f"{matched}/{len(refs)} references covered in top-{k}"


def _load_metric_class(dotted_path: str) -> type:
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _patch_deepeval_ragas_telemetry() -> None:
    """deepeval 3.9+ made `in_component` a required arg on capture_metric_type,
    but its own RAGAS wrappers still call the old 3-arg signature — every
    ragas_* metric raises TypeError without this shim. Idempotent."""
    from deepeval import telemetry
    from deepeval.metrics import ragas as ragas_mod

    orig = telemetry.capture_metric_type
    if getattr(orig, "_hydradb_patched", False):
        return

    def patched(metric_name, async_mode=False, in_component=False, _track=True):
        return orig(metric_name, async_mode=async_mode, in_component=in_component, _track=_track)

    patched._hydradb_patched = True  # type: ignore[attr-defined]
    telemetry.capture_metric_type = patched
    ragas_mod.capture_metric_type = patched


def _build_metric(name: str, config: DeepEvalConfig) -> Any:
    # answer_accuracy is a custom GEval metric — binary 0/1 via strict_mode
    if name == "answer_accuracy":
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCaseParams

        return GEval(
            name="answer_accuracy",
            criteria=_ANSWER_ACCURACY_CRITERIA,
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=config.threshold,
            model=config.model,
            strict_mode=True,  # forces score to 0.0 or 1.0
        )

    dotted_path = _METRIC_REGISTRY.get(name)
    if dotted_path is None:
        raise ValueError(f"Unknown metric: {name!r}. Available: {list(_METRIC_REGISTRY) + ['answer_accuracy']}")
    cls = _load_metric_class(dotted_path)
    # Ragas-backed wrappers: patch stale telemetry signature, and pass a
    # LangChain ChatOpenAI directly — deepeval's GPTModel wrapper returns a raw
    # openai.OpenAI client, which Ragas can't drive (expects BaseChatModel).
    if name.startswith("ragas_"):
        _patch_deepeval_ragas_telemetry()
        from langchain_openai import ChatOpenAI

        chat_model = ChatOpenAI(model=config.model, temperature=0)
        return cls(threshold=config.threshold, model=chat_model)
    return cls(
        threshold=config.threshold,
        model=config.model,
        include_reason=config.include_reason,
    )


class DeepEvalEvaluator:
    def __init__(self, config: DeepEvalConfig) -> None:
        self._config = config
        self._metrics = self._config.metrics

    async def evaluate(self, results: list[QueryResult]) -> tuple[dict[str, float], list[SampleScore]]:
        """Run DeepEval metrics over all QueryResults concurrently.

        Returns:
            aggregate_scores: dict[metric_name, mean_score]
            per_sample: list of SampleScore objects (in original order)
        """
        from deepeval.test_case import LLMTestCase

        # Validate metric names once up front
        valid_metrics: list[str] = []
        for name in self._metrics:
            if _RECALL_AT_K_PATTERN.match(name):
                valid_metrics.append(name)
                continue
            try:
                _build_metric(name, self._config)  # dry-run to catch bad names early
                valid_metrics.append(name)
            except Exception as exc:
                console.print(f"[yellow]Warning: could not load metric {name!r}: {exc}[/yellow]")

        if not valid_metrics:
            console.print("[red]No metrics could be loaded — skipping evaluation.[/red]")
            return {}, []

        semaphore = asyncio.Semaphore(max(1, self._config.eval_concurrency))
        total = len(results)

        async def evaluate_sample(idx: int, qr: QueryResult) -> SampleScore:
            async with semaphore:
                sample = qr.sample
                console.print(f"  Evaluating [{idx}/{total}] [bold]{sample.id}[/bold]")

                if qr.error:
                    console.print(f"    [yellow]Skipped (query error): {qr.error}[/yellow]")
                    return SampleScore(
                        sample_id=sample.id,
                        question=sample.question,
                        answer=qr.answer,
                        reference_answer=sample.reference_answer,
                        context_string=qr.context_string,
                        context_tokens=qr.context_tokens,
                        scores={name: None for name in valid_metrics},
                        reasons={name: f"Query error: {qr.error}" for name in valid_metrics},
                        latency_ms=qr.latency_ms,
                    )

                test_case = LLMTestCase(
                    input=sample.question,
                    actual_output=qr.answer,
                    retrieval_context=qr.retrieved_contexts if qr.retrieved_contexts else None,
                    expected_output=sample.reference_answer,
                )

                # Compute deterministic recall@k metrics up front (no LLM calls)
                scores: dict[str, float | None] = {}
                reasons: dict[str, str | None] = {}
                llm_metric_names: list[str] = []
                for name in valid_metrics:
                    m = _RECALL_AT_K_PATTERN.match(name)
                    if m:
                        k = int(m.group(1))
                        score, reason = _compute_recall_at_k(
                            sample.reference_contexts, qr.retrieved_contexts, k
                        )
                        scores[name] = score
                        reasons[name] = reason
                        score_str = f"{score:.3f}" if score is not None else "N/A"
                        console.print(f"    {name}: [bold]{score_str}[/bold] — {reason}")
                    else:
                        llm_metric_names.append(name)

                # Build fresh DeepEval metric instances per sample — avoids shared state races
                metric_objects: dict[str, Any] = {}
                for name in llm_metric_names:
                    try:
                        metric_objects[name] = _build_metric(name, self._config)
                    except Exception as exc:
                        console.print(f"    [yellow]Could not build {name!r}: {exc}[/yellow]")

                # Run all metrics for this sample in parallel
                timeout = self._config.metric_timeout_seconds

                async def measure_one(metric_name: str, metric: Any) -> tuple[str, float | None, str | None]:
                    try:
                        await asyncio.wait_for(metric.a_measure(test_case), timeout=timeout)
                        score = metric.score
                        reason = getattr(metric, "reason", None)
                        console.print(
                            f"    {metric_name}: [bold]{score:.3f}[/bold]"
                            + (f" — {reason}" if reason and self._config.include_reason else "")
                        )
                        return metric_name, score, reason
                    except asyncio.TimeoutError:
                        msg = f"timed out after {timeout}s"
                        console.print(f"    [red]{metric_name}: TIMEOUT — {msg}[/red]")
                        return metric_name, None, msg
                    except Exception as exc:
                        console.print(f"    [red]{metric_name}: ERROR — {exc}[/red]")
                        return metric_name, None, str(exc)

                metric_results = await asyncio.gather(*[measure_one(name, m) for name, m in metric_objects.items()])

                for metric_name, score, reason in metric_results:
                    scores[metric_name] = score
                    reasons[metric_name] = reason

                return SampleScore(
                    sample_id=sample.id,
                    question=sample.question,
                    answer=qr.answer,
                    reference_answer=sample.reference_answer,
                    context_string=qr.context_string,
                    context_tokens=qr.context_tokens,
                    scores=scores,
                    reasons=reasons,
                    latency_ms=qr.latency_ms,
                )

        # Launch all samples concurrently (semaphore limits actual parallelism)
        per_sample: list[SampleScore] = list(
            await asyncio.gather(*[evaluate_sample(i, qr) for i, qr in enumerate(results, 1)])
        )

        # Aggregate scores
        all_scores: dict[str, list[float]] = {name: [] for name in valid_metrics}
        for ss in per_sample:
            for name, score in ss.scores.items():
                if score is not None and name in all_scores:
                    all_scores[name].append(score)

        aggregate: dict[str, float] = {}
        for metric_name, score_list in all_scores.items():
            if score_list:
                aggregate[metric_name] = round(mean(score_list), 4)

        return aggregate, per_sample
