"""Benchmark query runner - executes test samples against HydraDB."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .checkpoint import CheckpointManager
from .client import HydraDBClient
from .context_builder import build_context_string
from .models import AnswerGenerationConfig, BenchmarkSample, EvaluationConfig, HydraSearchResult, TestSample

console = Console()
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs test queries against HydraDB and collects results."""

    def __init__(self, client: HydraDBClient, config: EvaluationConfig) -> None:
        self.client = client
        self.config = config

    def load_test_dataset(self, path: str) -> list[TestSample]:
        """Load and validate test_dataset.json."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Test dataset not found: {path}")

        with open(p, encoding="utf-8") as f:
            raw = json.load(f)

        samples = [TestSample(**item) for item in raw]
        logger.info("Loaded %d test samples from %s", len(samples), path)
        return samples

    async def run(
        self,
        test_samples: list[TestSample],
        checkpoint: CheckpointManager | None = None,
    ) -> list[BenchmarkSample]:
        """Execute all test samples with controlled concurrency."""
        # Skip already-completed samples when resuming
        pending = [s for s in test_samples if not (checkpoint and checkpoint.is_done(s.id))]
        skipped = len(test_samples) - len(pending)
        if skipped:
            console.print(f"  [yellow]Checkpoint:[/yellow] skipping {skipped} already-completed sample(s)")

        semaphore = asyncio.Semaphore(self.config.concurrent_requests)
        results: list[BenchmarkSample | None] = [None] * len(pending)

        console.print(f"[bold]Running {len(pending)} queries against HydraDB...[/bold]")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Querying", total=len(pending))

            async def run_one(idx: int, sample: TestSample) -> None:
                async with semaphore:
                    result = await self._query_single(sample)
                    results[idx] = result
                    if checkpoint:
                        checkpoint.save(result)
                    progress.advance(task_id)

            await asyncio.gather(*[run_one(i, s) for i, s in enumerate(pending)])

        return [r for r in results if r is not None]

    async def _query_single(self, sample: TestSample) -> BenchmarkSample:
        """Execute one query and return a BenchmarkSample."""
        error: str | None = None
        hydra_result: HydraSearchResult | None = None

        try:
            if self.config.search_endpoint == "full_recall":
                hydra_result = await self.client.full_recall(
                    query=sample.question,
                    max_results=self.config.max_results,
                    mode=self.config.retrieve_mode,
                )

                # If external LLM answer generation is enabled, replace the
                # HydraDB answer with one generated from the formatted context
                ag = self.config.answer_generation
                if ag.enabled and hydra_result is not None:
                    context_str = build_context_string(hydra_result.raw_response)
                    import tiktoken
                    _enc = tiktoken.get_encoding("cl100k_base")
                    context_token_count = len(_enc.encode(context_str))
                    if context_str.strip():
                        logger.info(
                            "[%s] Calling %s for answer generation (context=%d chars)",
                            sample.id, ag.model, len(context_str),
                        )
                        answer = await self._generate_answer(
                            query=sample.question,
                            context=context_str,
                            ag_config=ag,
                        )
                        if answer.strip():
                            logger.info(
                                "[%s] LLM answer (%d chars): %s",
                                sample.id, len(answer), answer[:200],
                            )
                            hydra_result = HydraSearchResult(
                                answer=answer,
                                retrieved_contexts=hydra_result.retrieved_contexts,
                                raw_response=hydra_result.raw_response,
                                network_latency_ms=hydra_result.network_latency_ms,
                                context_string=context_str,
                                context_tokens=context_token_count,
                            )
                        else:
                            logger.warning("Answer generation returned empty string for sample %s", sample.id)
                    else:
                        logger.warning(
                            "[%s] build_context_string returned empty — skipping LLM answer generation",
                            sample.id,
                        )
            else:  # default: qna
                hydra_result = await self.client.search_qna(
                    query=sample.question,
                    max_results=self.config.max_results,
                )

            if hydra_result is not None:
                logger.debug(
                    "Query '%s' -> answer length=%d, contexts=%d",
                    sample.question[:60],
                    len(hydra_result.answer),
                    len(hydra_result.retrieved_contexts),
                )
        except Exception as e:
            error = str(e)
            logger.error("Query failed for sample %s: %s", sample.id, e)

        return BenchmarkSample(
            test_sample=sample,
            hydra_result=hydra_result,
            latency_ms=hydra_result.network_latency_ms if hydra_result else 0.0,
            error=error,
        )

    async def _generate_answer(
        self,
        query: str,
        context: str,
        ag_config: AnswerGenerationConfig,
    ) -> str:
        """Call an external LLM to generate an answer from the formatted context."""
        import os
        from openai import AsyncOpenAI

        _PROVIDER_KEYS = {"openrouter": "OPENROUTER_API_KEY"}
        _PROVIDER_URLS = {"openrouter": "https://openrouter.ai/api/v1"}

        api_key = os.environ.get(_PROVIDER_KEYS.get(ag_config.provider, "OPENAI_API_KEY"), "")
        base_url = _PROVIDER_URLS.get(ag_config.provider)

        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        client = AsyncOpenAI(**kwargs)
        response = await client.chat.completions.create(
            model=ag_config.model,
            messages=[
                {"role": "system", "content": ag_config.system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
            max_tokens=ag_config.max_tokens,
            temperature=ag_config.temperature,
        )
        return response.choices[0].message.content or ""
