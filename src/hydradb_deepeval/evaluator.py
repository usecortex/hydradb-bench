from __future__ import annotations

import asyncio
import importlib
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
}

# GEval criteria for answer_accuracy
_ANSWER_ACCURACY_CRITERIA = (
    "Determine whether the actual output is factually correct and complete "
    "when compared to the expected output (reference answer). "
    "Award full marks if all key facts match. Deduct for missing facts, "
    "incorrect facts, or contradictions. Ignore differences in phrasing or style."
)


def _load_metric_class(dotted_path: str) -> type:
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


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

                # Build fresh metric instances per sample — avoids shared state races
                metric_objects: dict[str, Any] = {}
                for name in valid_metrics:
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

                scores: dict[str, float | None] = {}
                reasons: dict[str, str | None] = {}
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
