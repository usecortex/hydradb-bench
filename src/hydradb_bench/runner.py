"""Benchmark query runner - executes test samples against HydraDB."""

from __future__ import annotations

import asyncio
import json
import logging
import time
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

from .client import HydraDBClient
from .models import BenchmarkSample, EvaluationConfig, HydraSearchResult, TestSample

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

    async def run(self, test_samples: list[TestSample]) -> list[BenchmarkSample]:
        """Execute all test samples with controlled concurrency."""
        semaphore = asyncio.Semaphore(self.config.concurrent_requests)
        results: list[BenchmarkSample | None] = [None] * len(test_samples)

        console.print(f"[bold]Running {len(test_samples)} queries against HydraDB...[/bold]")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Querying", total=len(test_samples))

            async def run_one(idx: int, sample: TestSample) -> None:
                async with semaphore:
                    result = await self._query_single(sample)
                    results[idx] = result
                    progress.advance(task_id)

            await asyncio.gather(*[run_one(i, s) for i, s in enumerate(test_samples)])

        return [r for r in results if r is not None]

    async def _query_single(self, sample: TestSample) -> BenchmarkSample:
        """Execute one query and return a BenchmarkSample."""
        start = time.monotonic()
        error: str | None = None
        hydra_result: HydraSearchResult | None = None

        try:
            if self.config.search_endpoint == "full_recall":
                hydra_result = await self.client.full_recall(
                    query=sample.question,
                    max_results=self.config.max_results,
                )
            else:  # default: qna
                hydra_result = await self.client.search_qna(
                    query=sample.question,
                    max_results=self.config.max_results,
                )
            logger.debug(
                "Query '%s' → answer length=%d, contexts=%d",
                sample.question[:60],
                len(hydra_result.answer),
                len(hydra_result.retrieved_contexts),
            )
        except Exception as e:
            error = str(e)
            logger.error("Query failed for sample %s: %s", sample.id, e)

        latency_ms = (time.monotonic() - start) * 1000

        return BenchmarkSample(
            test_sample=sample,
            hydra_result=hydra_result,
            latency_ms=latency_ms,
            error=error,
        )
