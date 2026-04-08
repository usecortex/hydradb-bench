#!/usr/bin/env python3
"""HydraDB + Supermemory DeepEval Benchmark — main runner.

Usage examples:
  python run_benchmark.py                                  # HydraDB only (default)
  python run_benchmark.py --provider supermemory           # Supermemory only
  python run_benchmark.py --provider both                  # Run both, compare
  python run_benchmark.py --skip-ingestion                 # Skip upload phase
  python run_benchmark.py --ingest-only                   # Upload then exit
  python run_benchmark.py --reset-tenant                  # Wipe HydraDB tenant first
  python run_benchmark.py --limit 20                      # Only 20 test samples
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from hydradb_deepeval.answer_generator import generate_answer
from hydradb_deepeval.client import HydraDBClient
from hydradb_deepeval.config import load_config
from hydradb_deepeval.context_builder import build_context_string
from hydradb_deepeval.evaluator import DeepEvalEvaluator
from hydradb_deepeval.ingestion import DocumentIngester
from hydradb_deepeval.models import (
    BenchmarkResult,
    EvaluationConfig,
    QueryResult,
    SupermemoryConfig,
    TestSample,
)
from hydradb_deepeval.reporter import BenchmarkReporter
from hydradb_deepeval.supermemory_client import SupermemoryClient, SupermemoryIngester

console = Console()


# ---------------------------------------------------------------------------
# Token counting (tiktoken, fallback to word count)
# ---------------------------------------------------------------------------


def _count_tokens(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_benchmark",
        description="HydraDB / Supermemory DeepEval Benchmark Runner",
    )
    p.add_argument(
        "--config",
        default="config/benchmark.yaml",
        metavar="PATH",
        help="Path to benchmark.yaml (default: config/benchmark.yaml)",
    )
    p.add_argument(
        "--provider",
        choices=["hydradb", "supermemory", "both"],
        default="hydradb",
        help=(
            "Which provider(s) to benchmark. "
            "'both' runs HydraDB and Supermemory in sequence and compares results. "
            "(default: hydradb)"
        ),
    )
    p.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip document upload and jump straight to querying",
    )
    p.add_argument(
        "--ingest-only",
        action="store_true",
        help="Upload and index documents then exit (no querying or evaluation)",
    )
    p.add_argument(
        "--reset-tenant",
        action="store_true",
        help="Delete the HydraDB tenant then recreate it before ingestion",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only run the first N test samples",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show extra debug output",
    )
    return p


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_test_dataset(path: str) -> list[TestSample]:
    p = Path(path)
    if not p.exists():
        console.print(f"[red]Test dataset not found: {p.resolve()}[/red]")
        sys.exit(1)
    raw = json.loads(p.read_text(encoding="utf-8"))
    return [TestSample(**item) for item in raw]


# ---------------------------------------------------------------------------
# HydraDB query runner
# ---------------------------------------------------------------------------


async def run_single_query_hydradb(
    client: HydraDBClient,
    sample: TestSample,
    cfg: EvaluationConfig,
    gen_cfg,  # DeepEvalConfig
) -> QueryResult:
    endpoint = cfg.search_endpoint
    t0 = time.monotonic()
    try:
        if endpoint == "full_recall":
            response = await client.full_recall(
                sample.question,
                max_results=cfg.max_results,
                mode=cfg.mode,
                alpha=cfg.alpha,
                graph_context=cfg.graph_context,
                recency_bias=cfg.recency_bias,
            )
            latency_ms = (time.monotonic() - t0) * 1000
            context_str = build_context_string(response)
            chunks = response.get("chunks", [])
            contexts = [c.get("chunk_content", "") for c in chunks if c.get("chunk_content")]

        elif endpoint == "recall_preferences":
            response = await client.recall_preferences(
                sample.question,
                max_results=cfg.max_results,
                mode=cfg.mode,
                alpha=cfg.alpha,
                graph_context=cfg.graph_context,
            )
            latency_ms = (time.monotonic() - t0) * 1000
            chunks = response.get("chunks", [])
            contexts = [c.get("chunk_content", "") for c in chunks if c.get("chunk_content")]
            context_str = "\n\n".join(contexts)

        elif endpoint == "boolean_recall":
            response = await client.boolean_recall(
                sample.question,
                operator=cfg.boolean_operator,
                max_results=cfg.max_results,
                search_mode=cfg.boolean_search_mode,
            )
            latency_ms = (time.monotonic() - t0) * 1000
            chunks = response.get("chunks", [])
            contexts = [c.get("chunk_content", "") for c in chunks if c.get("chunk_content")]
            context_str = "\n\n".join(contexts)

        else:
            raise ValueError(
                f"Unknown search_endpoint: {endpoint!r}. "
                "Valid values: 'full_recall', 'recall_preferences', 'boolean_recall'."
            )

        context_tokens = _count_tokens(context_str) if context_str else 0

        if context_str.strip():
            answer = await generate_answer(
                question=sample.question,
                context_str=context_str,
                model=gen_cfg.generator_model,
                temperature=gen_cfg.generator_temperature,
                max_tokens=gen_cfg.generator_max_tokens,
            )
        else:
            answer = ""

        return QueryResult(
            sample=sample,
            answer=answer,
            retrieved_contexts=contexts,  # raw chunk_content list for DeepEval metrics
            context_string=context_str,  # full formatted context for report display
            context_tokens=context_tokens,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        return QueryResult(sample=sample, latency_ms=latency_ms, error=str(exc))


# ---------------------------------------------------------------------------
# Supermemory query runner
# ---------------------------------------------------------------------------


async def run_single_query_supermemory(
    client: SupermemoryClient,
    sample: TestSample,
    eval_cfg: EvaluationConfig,
    sm_cfg: SupermemoryConfig,
    gen_cfg,  # DeepEvalConfig
) -> QueryResult:
    t0 = time.monotonic()
    try:
        response = await client.search(
            query=sample.question,
            container_tag=sm_cfg.container_tag,
            limit=sm_cfg.limit,  # total chunks — use sm_cfg.limit not eval_cfg.max_results
            search_mode=sm_cfg.search_mode,
            rerank=sm_cfg.rerank,
            threshold=sm_cfg.threshold,
        )
        latency_ms = (time.monotonic() - t0) * 1000

        # Results are grouped by document. Flatten all chunks across all
        # documents, then re-sort globally by score (descending) so the most
        # relevant chunks come first — critical for contextual_precision which
        # is positional, and for LLM answer quality.
        results = response.get("results", [])
        all_chunks = [
            (chunk.get("score", 0.0), chunk.get("content", ""))
            for r in results
            for chunk in r.get("chunks", [])
            if chunk.get("content")
        ]
        all_chunks.sort(key=lambda x: x[0], reverse=True)
        contexts = [content for _, content in all_chunks]
        context_str = "\n\n".join(contexts)
        context_tokens = _count_tokens(context_str) if context_str else 0

        if context_str.strip():
            answer = await generate_answer(
                question=sample.question,
                context_str=context_str,
                model=gen_cfg.generator_model,
                temperature=gen_cfg.generator_temperature,
                max_tokens=gen_cfg.generator_max_tokens,
            )
        else:
            answer = ""

        return QueryResult(
            sample=sample,
            answer=answer,
            retrieved_contexts=contexts,
            context_string=context_str,
            context_tokens=context_tokens,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        return QueryResult(sample=sample, latency_ms=latency_ms, error=str(exc))


# ---------------------------------------------------------------------------
# Generic concurrent query runner (works for either provider)
# ---------------------------------------------------------------------------


async def run_all_queries(
    query_fn,  # async callable: (sample) -> QueryResult
    samples: list[TestSample],
    concurrency: int,
    label: str,
    verbose: bool,
) -> list[QueryResult]:
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(sample: TestSample) -> QueryResult:
        async with semaphore:
            return await query_fn(sample)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Querying {label}…", total=len(samples))

        async def tracked(sample: TestSample) -> QueryResult:
            result = await bounded(sample)
            progress.advance(task)
            if verbose:
                status = "[red]ERROR[/red]" if result.error else "[green]OK[/green]"
                console.print(f"  {status} {sample.id} ({result.latency_ms:.0f} ms)")
            return result

        results = await asyncio.gather(*[tracked(s) for s in samples])

    return list(results)


# ---------------------------------------------------------------------------
# Latency / token stats
# ---------------------------------------------------------------------------


def compute_context_token_stats(results: list[QueryResult]) -> dict:
    tokens = [r.context_tokens for r in results if not r.error and r.context_tokens > 0]
    if not tokens:
        return {}
    tokens.sort()
    return {"min": tokens[0], "mean": round(mean(tokens)), "max": tokens[-1]}


def compute_latency_stats(results: list[QueryResult]) -> tuple[float, float, dict]:
    latencies = sorted(r.latency_ms for r in results if not r.error)
    if not latencies:
        return 0.0, 0.0, {}
    n = len(latencies)

    def pct(p: float) -> float:
        return latencies[min(int(n * p), n - 1)]

    stats = {
        "min": round(latencies[0], 1),
        "mean": round(sum(latencies) / n, 1),
        "p50": round(median(latencies), 1),
        "p75": round(pct(0.75), 1),
        "p95": round(pct(0.95), 1),
        "p99": round(pct(0.99), 1),
        "max": round(latencies[-1], 1),
    }
    return stats["p50"], stats["p95"], stats


# ---------------------------------------------------------------------------
# Rich display helpers
# ---------------------------------------------------------------------------


def print_score_table(result: BenchmarkResult) -> None:
    table = Table(title=f"Aggregate Scores — {result.name}", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Status")

    for metric, score in result.aggregate_scores.items():
        if score >= 0.8:
            status = "[green]PASS[/green]"
            score_str = f"[green]{score:.4f}[/green]"
        elif score >= 0.5:
            status = "[yellow]MARGINAL[/yellow]"
            score_str = f"[yellow]{score:.4f}[/yellow]"
        else:
            status = "[red]FAIL[/red]"
            score_str = f"[red]{score:.4f}[/red]"
        table.add_row(metric, score_str, status)

    console.print(table)
    console.print(
        f"  Latency p50: [bold]{result.latency_p50_ms:.0f} ms[/bold]  "
        f"p95: [bold]{result.latency_p95_ms:.0f} ms[/bold]  "
        f"Errors: [bold]{result.error_count}/{result.total_samples}[/bold]"
    )


def print_comparison_table(
    hydra_result: BenchmarkResult,
    sm_result: BenchmarkResult,
) -> None:
    """Side-by-side metric comparison table for --provider both."""
    all_metrics = sorted(set(hydra_result.aggregate_scores) | set(sm_result.aggregate_scores))
    table = Table(
        title="Benchmark Comparison: HydraDB vs Supermemory",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="bold")
    table.add_column("HydraDB", justify="right")
    table.add_column("Supermemory", justify="right")
    table.add_column("Winner")

    for metric in all_metrics:
        h_score = hydra_result.aggregate_scores.get(metric)
        s_score = sm_result.aggregate_scores.get(metric)

        h_str = f"{h_score:.4f}" if h_score is not None else "—"
        s_str = f"{s_score:.4f}" if s_score is not None else "—"

        if h_score is not None and s_score is not None:
            if h_score > s_score + 0.005:
                winner = "[cyan]HydraDB[/cyan]"
            elif s_score > h_score + 0.005:
                winner = "[magenta]Supermemory[/magenta]"
            else:
                winner = "[dim]Tie[/dim]"
        else:
            winner = "—"

        table.add_row(metric, h_str, s_str, winner)

    console.print(table)
    console.print(
        f"  HydraDB      — p50: [bold]{hydra_result.latency_p50_ms:.0f} ms[/bold]  "
        f"p95: [bold]{hydra_result.latency_p95_ms:.0f} ms[/bold]  "
        f"Errors: {hydra_result.error_count}/{hydra_result.total_samples}"
    )
    console.print(
        f"  Supermemory  — p50: [bold]{sm_result.latency_p50_ms:.0f} ms[/bold]  "
        f"p95: [bold]{sm_result.latency_p95_ms:.0f} ms[/bold]  "
        f"Errors: {sm_result.error_count}/{sm_result.total_samples}"
    )


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


async def _run_provider_queries_and_evaluate(
    query_results: list[QueryResult],
    config,
    run_id: str,
    timestamp: str,
    provider_name: str,
) -> tuple[BenchmarkResult, list[Path]]:
    """Evaluate a list of QueryResults and save reports. Returns (result, saved_paths)."""
    error_count = sum(1 for r in query_results if r.error)

    evaluator = DeepEvalEvaluator(config.deepeval)
    aggregate_scores, per_sample = await evaluator.evaluate(query_results)

    p50, p95, latency_stats = compute_latency_stats(query_results)
    context_token_stats = compute_context_token_stats(query_results)

    result = BenchmarkResult(
        run_id=f"{run_id}_{provider_name}",
        timestamp=timestamp,
        name=f"{config.name} [{provider_name}]",
        aggregate_scores=aggregate_scores,
        per_sample=per_sample,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        latency_stats=latency_stats,
        context_token_stats=context_token_stats,
        total_samples=len(query_results),
        error_count=error_count,
    )

    reporter = BenchmarkReporter(
        output_dir=config.reporting.output_dir,
        formats=config.reporting.formats,
        include_per_sample=config.reporting.include_per_sample,
    )
    saved_paths = reporter.save(result)
    return result, saved_paths


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


async def main(args: argparse.Namespace) -> None:
    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()
    provider = args.provider

    console.print(
        Panel(
            f"[bold cyan]DeepEval Benchmark[/bold cyan]\n"
            f"Provider: [bold]{provider}[/bold]   "
            f"Run ID: [dim]{run_id}[/dim]   Started: [dim]{timestamp}[/dim]",
            expand=False,
        )
    )

    # ------------------------------------------------------------------ 1/4
    console.rule("[bold]1/4  Load config + validate env[/bold]")
    try:
        config = load_config(args.config)
    except OSError as exc:
        console.print(f"[red]Configuration error: {exc}[/red]")
        sys.exit(1)

    # Validate Supermemory config is present when needed
    if provider in ("supermemory", "both") and config.supermemory is None:
        console.print(
            "[red]Error: --provider supermemory/both requires a [supermemory] section "
            "in benchmark.yaml and SUPERMEMORY_API_KEY set in .env[/red]"
        )
        sys.exit(1)

    console.print(f"  Benchmark : [bold]{config.name}[/bold]")
    if provider in ("hydradb", "both"):
        console.print(
            f"  HydraDB   : [bold]{config.hydradb.base_url}[/bold]  "
            f"tenant=[bold]{config.hydradb.tenant_id}[/bold]  "
            f"endpoint=[bold]{config.evaluation.search_endpoint}[/bold]"
        )
    if provider in ("supermemory", "both") and config.supermemory:
        console.print(
            f"  Supermemory: [bold]{config.supermemory.base_url}[/bold]  "
            f"container=[bold]{config.supermemory.container_tag}[/bold]  "
            f"mode=[bold]{config.supermemory.search_mode}[/bold]"
        )
    console.print(f"  Metrics   : {', '.join(config.deepeval.metrics)}")

    # ------------------------------------------------------------------ 2/4
    console.rule("[bold]2/4  Ingest documents[/bold]")

    # ── HydraDB ingestion ─────────────────────────────────────────────────
    if provider in ("hydradb", "both"):
        async with HydraDBClient(config.hydradb) as hydra_client:
            if args.reset_tenant:
                console.print("  [HydraDB] Deleting tenant…")
                try:
                    await hydra_client.delete_tenant()
                    console.print("  [green]Tenant deleted.[/green]")
                except Exception as exc:
                    console.print(f"  [yellow]Delete failed (may not exist): {exc}[/yellow]")

            if config.hydradb.create_tenant_on_start and not args.skip_ingestion:
                console.print("  [HydraDB] Creating tenant…")
                try:
                    result = await hydra_client.create_tenant()
                    if result.get("status") == "already_exists":
                        console.print("  [yellow]Tenant already exists — continuing.[/yellow]")
                    else:
                        console.print("  [green]Tenant created.[/green]")
                except Exception as exc:
                    console.print(f"  [yellow]Warning: create_tenant failed: {exc}[/yellow]")

            if args.skip_ingestion:
                console.print("  [dim][HydraDB] --skip-ingestion set; skipping upload.[/dim]")
            else:
                ingester = DocumentIngester(hydra_client, config.ingestion, config.hydradb)
                indexed, failed, elapsed = await ingester.run()
                console.print(
                    f"  [HydraDB] [green]Indexed {indexed} file(s)[/green], "
                    f"[red]{failed} failed[/red], elapsed {elapsed:.1f}s"
                )

    # ── Supermemory ingestion ─────────────────────────────────────────────
    if provider in ("supermemory", "both") and config.supermemory:
        sm_cfg = config.supermemory
        async with SupermemoryClient(sm_cfg) as sm_client:
            if sm_cfg.reset_on_start and not args.skip_ingestion:
                console.print(f"  [Supermemory] Deleting container '{sm_cfg.container_tag}'…")
                try:
                    await sm_client.delete_container_tag(sm_cfg.container_tag)
                    console.print("  [green]Container deleted.[/green]")
                except Exception as exc:
                    console.print(f"  [yellow]Delete failed (may not exist): {exc}[/yellow]")

            if args.skip_ingestion:
                console.print("  [dim][Supermemory] --skip-ingestion set; skipping upload.[/dim]")
            else:
                sm_ingester = SupermemoryIngester(sm_client, config.ingestion, sm_cfg)
                indexed, failed, elapsed = await sm_ingester.run()
                console.print(
                    f"  [Supermemory] [green]Indexed {indexed} file(s)[/green], "
                    f"[red]{failed} failed[/red], elapsed {elapsed:.1f}s"
                )

    if args.ingest_only:
        console.print(Panel("[bold green]Ingestion complete![/bold green]", expand=False))
        return

    # ------------------------------------------------------------------ 3/4
    console.rule("[bold]3/4  Run queries[/bold]")
    samples = load_test_dataset(config.evaluation.test_dataset_path)
    if args.limit:
        samples = samples[: args.limit]
    console.print(f"  Loaded [bold]{len(samples)}[/bold] test sample(s)")

    hydra_query_results: list[QueryResult] = []
    sm_query_results: list[QueryResult] = []

    # ── HydraDB queries ───────────────────────────────────────────────────
    if provider in ("hydradb", "both"):
        async with HydraDBClient(config.hydradb) as hydra_client:

            def hydra_fn(s):
                return run_single_query_hydradb(hydra_client, s, config.evaluation, config.deepeval)

            hydra_query_results = await run_all_queries(
                query_fn=hydra_fn,
                samples=samples,
                concurrency=config.evaluation.concurrent_requests,
                label=f"HydraDB ({config.evaluation.search_endpoint})",
                verbose=args.verbose,
            )
        h_errors = sum(1 for r in hydra_query_results if r.error)
        console.print(
            f"  [HydraDB] Queries done: [bold]{len(hydra_query_results)}[/bold]  errors: [red]{h_errors}[/red]"
        )

    # ── Supermemory queries ───────────────────────────────────────────────
    if provider in ("supermemory", "both") and config.supermemory:
        sm_cfg = config.supermemory
        async with SupermemoryClient(sm_cfg) as sm_client:

            def sm_fn(s):
                return run_single_query_supermemory(sm_client, s, config.evaluation, sm_cfg, config.deepeval)

            sm_query_results = await run_all_queries(
                query_fn=sm_fn,
                samples=samples,
                concurrency=config.evaluation.concurrent_requests,
                label=f"Supermemory ({sm_cfg.search_mode})",
                verbose=args.verbose,
            )
        s_errors = sum(1 for r in sm_query_results if r.error)
        console.print(
            f"  [Supermemory] Queries done: [bold]{len(sm_query_results)}[/bold]  errors: [red]{s_errors}[/red]"
        )

    # ------------------------------------------------------------------ 4/4
    console.rule("[bold]4/4  Evaluate with DeepEval + generate reports[/bold]")

    all_saved_paths: list[Path] = []

    hydra_result: BenchmarkResult | None = None
    sm_result: BenchmarkResult | None = None

    if provider in ("hydradb", "both") and hydra_query_results:
        console.print("\n  [bold cyan]Evaluating HydraDB results…[/bold cyan]")
        hydra_result, paths = await _run_provider_queries_and_evaluate(
            hydra_query_results, config, run_id, timestamp, provider_name="HydraDB"
        )
        all_saved_paths.extend(paths)
        print_score_table(hydra_result)

    if provider in ("supermemory", "both") and sm_query_results:
        console.print("\n  [bold magenta]Evaluating Supermemory results…[/bold magenta]")
        sm_result, paths = await _run_provider_queries_and_evaluate(
            sm_query_results, config, run_id, timestamp, provider_name="Supermemory"
        )
        all_saved_paths.extend(paths)
        print_score_table(sm_result)

    # Side-by-side comparison when both ran
    if hydra_result and sm_result:
        console.print()
        print_comparison_table(hydra_result, sm_result)
        cmp_reporter = BenchmarkReporter(
            output_dir=config.reporting.output_dir,
            formats=config.reporting.formats,
            include_per_sample=config.reporting.include_per_sample,
        )
        cmp_path = cmp_reporter.save_comparison(hydra_result, sm_result, run_id)
        all_saved_paths.append(cmp_path)
        console.print(f"\n  [bold]Comparison report:[/bold] {cmp_path.resolve()}")

    console.print("\n[bold green]Reports saved:[/bold green]")
    for path in all_saved_paths:
        console.print(f"  {path.resolve()}")

    console.print(Panel("[bold green]Benchmark complete![/bold green]", expand=False))


def cli() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
