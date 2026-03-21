"""Multi-turn conversation benchmarking with RAGAS MultiTurnSample."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from .client import HydraDBClient
from .models import EvaluationConfig, MultiTurnBenchmarkSample, MultiTurnSampleConfig

console = Console()
logger = logging.getLogger(__name__)


def load_multi_turn_dataset(path: str) -> list[MultiTurnSampleConfig]:
    """Load multi_turn_dataset.json."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Multi-turn dataset not found: {path}")

    with open(p, encoding="utf-8") as f:
        raw = json.load(f)

    samples = [MultiTurnSampleConfig(**item) for item in raw]
    logger.info("Loaded %d multi-turn conversations from %s", len(samples), path)
    return samples


class MultiTurnRunner:
    """
    Runs multi-turn conversations against HydraDB.

    For each conversation in the dataset:
      - Sends each human turn to HydraDB sequentially (maintaining context via session_id)
      - Collects AI responses
      - Returns a MultiTurnBenchmarkSample ready for RAGAS evaluation
    """

    def __init__(self, client: HydraDBClient, config: EvaluationConfig) -> None:
        self.client = client
        self.config = config

    async def run(
        self, conversations: list[MultiTurnSampleConfig]
    ) -> list[MultiTurnBenchmarkSample]:
        semaphore = asyncio.Semaphore(self.config.concurrent_requests)
        results: list[MultiTurnBenchmarkSample | None] = [None] * len(conversations)

        console.print(
            f"[bold]Running {len(conversations)} multi-turn conversation(s)...[/bold]"
        )

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Conversations", total=len(conversations))

            async def run_one(idx: int, conv: MultiTurnSampleConfig) -> None:
                async with semaphore:
                    result = await self._run_conversation(conv)
                    results[idx] = result
                    progress.advance(task_id)

            await asyncio.gather(*[run_one(i, c) for i, c in enumerate(conversations)])

        return [r for r in results if r is not None]

    async def _run_conversation(
        self, conv: MultiTurnSampleConfig
    ) -> MultiTurnBenchmarkSample:
        """Execute a single multi-turn conversation with HydraDB."""
        conversation_turns: list[dict[str, Any]] = []
        total_latency_ms = 0.0
        error: str | None = None

        for turn in conv.turns:
            if turn.role != "human":
                continue  # skip non-human turns in dataset spec

            # Query HydraDB with the human message
            start = time.monotonic()
            try:
                if self.config.search_endpoint == "full_recall":
                    hydra_result = await self.client.full_recall(
                        query=turn.content,
                        max_results=self.config.max_results,
                    )
                else:
                    hydra_result = await self.client.search_qna(
                        query=turn.content,
                        max_results=self.config.max_results,
                    )
                latency_ms = (time.monotonic() - start) * 1000
                total_latency_ms += latency_ms

                # Add human turn
                conversation_turns.append({
                    "role": "human",
                    "content": turn.content,
                    "latency_ms": 0.0,
                })
                # Add AI response
                conversation_turns.append({
                    "role": "ai",
                    "content": hydra_result.answer,
                    "latency_ms": round(latency_ms, 2),
                    "contexts": hydra_result.retrieved_contexts,
                })
                logger.debug(
                    "Conv %s turn: '%s...' → %d chars",
                    conv.id, turn.content[:40], len(hydra_result.answer)
                )
            except Exception as e:
                error = str(e)
                logger.error("Multi-turn query failed for conv %s: %s", conv.id, e)
                conversation_turns.append({
                    "role": "human", "content": turn.content, "latency_ms": 0.0
                })
                conversation_turns.append({
                    "role": "ai", "content": f"[ERROR: {e}]", "latency_ms": 0.0
                })

        return MultiTurnBenchmarkSample(
            conversation_id=conv.id,
            conversation_turns=conversation_turns,
            reference=conv.reference,
            reference_topics=conv.reference_topics,
            total_latency_ms=round(total_latency_ms, 2),
            error=error,
        )


def multi_turn_samples_to_evaluator_input(
    samples: list[MultiTurnBenchmarkSample],
) -> list[dict[str, Any]]:
    """
    Convert MultiTurnBenchmarkSamples to the dict format expected by
    RAGASEvaluator.build_multi_turn_dataset().
    """
    result = []
    for s in samples:
        if s.error:
            continue
        result.append({
            "conversation_id": s.conversation_id,
            "conversation_turns": s.conversation_turns,
            "reference": s.reference,
            "reference_topics": s.reference_topics,
        })
    return result
