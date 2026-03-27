"""Send benchmark results to a Telegram chat when a run completes."""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import httpx

from .models import BenchmarkResult, TelegramConfig

logger = logging.getLogger(__name__)


def _api(token: str, method: str) -> str:
    return f"https://api.telegram.org/bot{token}/{method}"


def _build_summary(result: BenchmarkResult) -> str:
    lines = [
        f"*HydraDB Benchmark Complete* — `{result.run_id}`",
        f"*Run:* {result.benchmark_name}",
        f"*Timestamp:* {result.timestamp}",
        "",
    ]

    if result.ragas_scores:
        lines.append("*RAGAS Scores:*")
        for metric, score in result.ragas_scores.items():
            bar = "Good" if score >= 0.8 else ("Fair" if score >= 0.5 else "Poor")
            lines.append(f"  • {metric.replace('_', ' ').title()}: `{score:.1%}` ({bar})")
        lines.append("")

    lines.append(f"*Samples:* {len(result.samples)} queried, {result.error_count} errors")

    if result.latency_stats:
        p50 = result.latency_stats.get("p50", 0)
        p95 = result.latency_stats.get("p95", 0)
        lines.append(f"*Latency:* p50={p50:.0f}ms  p95={p95:.0f}ms")

    if result.token_usage.total_tokens > 0:
        lines.append(
            f"*Judge Cost:* ${result.token_usage.actual_cost_usd:.4f} "
            f"({result.token_usage.total_tokens:,} tokens)"
        )

    return "\n".join(lines)


async def send_results(
    config: TelegramConfig,
    result: BenchmarkResult,
    report_paths: list[str],
) -> None:
    """Send summary message + zipped reports to the configured Telegram chat."""
    if not config.enabled:
        return

    if not config.bot_token or not config.chat_id:
        logger.warning("Telegram enabled but bot_token or chat_id is empty — skipping.")
        return

    async with httpx.AsyncClient(timeout=120) as client:
        # 1 — send summary text
        r = await client.post(
            _api(config.bot_token, "sendMessage"),
            json={
                "chat_id": config.chat_id,
                "text": _build_summary(result),
                "parse_mode": "Markdown",
            },
        )
        r.raise_for_status()
        if not r.json().get("ok"):
            logger.error("Telegram sendMessage failed: %s", r.json().get("description"))
            return

        # 2 — zip all report files and send as a document
        existing = [Path(p) for p in report_paths if Path(p).exists()]
        if not existing:
            logger.warning("No report files found to send.")
            return

        zip_name = f"bench_{result.run_id}_reports.zip"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in existing:
                zf.write(path, path.name)
        buf.seek(0)

        doc_r = await client.post(
            _api(config.bot_token, "sendDocument"),
            data={
                "chat_id": config.chat_id,
                "caption": f"Reports for run `{result.run_id}`",
                "parse_mode": "Markdown",
            },
            files={"document": (zip_name, buf, "application/zip")},
        )
        doc_r.raise_for_status()
        body = doc_r.json()
        if not body.get("ok"):
            logger.error("Telegram sendDocument failed: %s", body.get("description"))
            return

    logger.info("Telegram notification sent for run %s", result.run_id)
