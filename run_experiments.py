#!/usr/bin/env python3
"""Run all experiment configurations and collect results into a unified analysis.

Usage:
  python run_experiments.py                    # Run all experiments
  python run_experiments.py --ids 01_baseline 02_alpha_0_5  # Run specific experiments
  python run_experiments.py --limit 5          # Only 5 samples per experiment
  python run_experiments.py --dry-run          # Show what would run without executing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

MANIFEST_PATH = Path("config/experiments/manifest.yaml")
RESULTS_DIR = Path("reports/experiments")
EXPERIMENT_LOG = RESULTS_DIR / "experiment_log.jsonl"


def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        console.print(f"[red]Manifest not found: {MANIFEST_PATH}[/red]")
        console.print("Run: python generate_experiment_configs.py")
        sys.exit(1)
    with open(MANIFEST_PATH) as f:
        data = yaml.safe_load(f)
    return data.get("experiments", [])


def append_experiment_log(entry: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


async def run_single_experiment(
    experiment: dict,
    limit: int | None,
    verbose: bool,
) -> dict:
    """Run a single experiment and return summary metrics."""
    from hydradb_deepeval.config import load_config

    exp_id = experiment["id"]
    exp_name = experiment["name"]
    config_path = experiment["config"]

    console.print(
        Panel(
            f"[bold cyan]Experiment: {exp_name}[/bold cyan]\nConfig: [dim]{config_path}[/dim]  ID: [dim]{exp_id}[/dim]",
            expand=False,
        )
    )

    t0 = time.monotonic()

    try:
        config = load_config(config_path)
    except Exception as exc:
        console.print(f"[red]Config load failed: {exc}[/red]")
        return {"experiment_id": exp_id, "error": str(exc), "status": "config_error"}

    # Import here to avoid circular issues
    from hydradb_deepeval.client import HydraDBClient
    from run_benchmark import (
        _run_provider_queries_and_evaluate,
        load_test_dataset,
        run_all_queries,
        run_single_query_hydradb,
    )

    # Load samples
    samples = load_test_dataset(config.evaluation.test_dataset_path)
    if limit:
        samples = samples[:limit]

    # Run queries
    try:
        async with HydraDBClient(config.hydradb) as client:

            def query_fn(s):
                return run_single_query_hydradb(client, s, config.evaluation, config.deepeval)

            query_results = await run_all_queries(
                query_fn=query_fn,
                samples=samples,
                concurrency=config.evaluation.concurrent_requests,
                label=f"{exp_id} ({config.evaluation.search_endpoint})",
                verbose=verbose,
            )
    except Exception as exc:
        console.print(f"[red]Query phase failed: {exc}[/red]")
        return {"experiment_id": exp_id, "error": str(exc), "status": "query_error"}

    error_count = sum(1 for r in query_results if r.error)
    if error_count == len(query_results):
        first_error = next(r.error for r in query_results if r.error)
        console.print(f"[red]All queries failed. First error: {first_error}[/red]")
        return {
            "experiment_id": exp_id,
            "error": first_error,
            "status": "all_queries_failed",
            "total_samples": len(query_results),
        }

    # Evaluate
    try:
        import uuid

        run_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now(timezone.utc).isoformat()
        result, saved_paths = await _run_provider_queries_and_evaluate(
            query_results,
            config,
            run_id,
            timestamp,
            provider_name=exp_id,
        )
    except Exception as exc:
        console.print(f"[red]Evaluation failed: {exc}[/red]")
        return {"experiment_id": exp_id, "error": str(exc), "status": "eval_error"}

    elapsed = time.monotonic() - t0

    # Build summary
    summary = {
        "experiment_id": exp_id,
        "name": exp_name,
        "status": "completed",
        "config": {
            "search_endpoint": config.evaluation.search_endpoint,
            "alpha": config.evaluation.alpha,
            "mode": config.evaluation.mode,
            "graph_context": config.evaluation.graph_context,
            "max_results": config.evaluation.max_results,
        },
        "metrics": result.aggregate_scores,
        "latency": result.latency_stats,
        "latency_p50_ms": result.latency_p50_ms,
        "latency_p95_ms": result.latency_p95_ms,
        "context_token_stats": result.context_token_stats,
        "total_samples": result.total_samples,
        "error_count": result.error_count,
        "elapsed_seconds": round(elapsed, 1),
        "report_paths": [str(p) for p in saved_paths],
    }

    # Print summary table
    table = Table(title=f"Results: {exp_name}", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")
    for metric, score in result.aggregate_scores.items():
        color = "green" if score >= 0.8 else ("yellow" if score >= 0.5 else "red")
        table.add_row(metric, f"[{color}]{score:.4f}[/{color}]")
    table.add_row("---", "---")
    table.add_row("Latency p50", f"{result.latency_p50_ms:.0f} ms")
    table.add_row("Latency p95", f"{result.latency_p95_ms:.0f} ms")
    table.add_row("Errors", f"{result.error_count}/{result.total_samples}")
    console.print(table)

    return summary


async def main():
    parser = argparse.ArgumentParser(description="Run HydraDB retrieval quality experiments")
    parser.add_argument("--ids", nargs="+", help="Run only these experiment IDs")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per experiment")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Show experiments without running")
    args = parser.parse_args()

    manifest = load_manifest()

    if args.ids:
        manifest = [e for e in manifest if e["id"] in args.ids]
        if not manifest:
            console.print(f"[red]No matching experiments for IDs: {args.ids}[/red]")
            sys.exit(1)

    console.print(
        Panel(
            f"[bold]HydraDB Retrieval Quality Experiment Suite[/bold]\n"
            f"Experiments: [bold]{len(manifest)}[/bold]  "
            f"Samples/exp: [bold]{args.limit or 'all (30)'}[/bold]",
            expand=False,
        )
    )

    if args.dry_run:
        table = Table(title="Experiment Plan", show_header=True)
        table.add_column("#", style="dim")
        table.add_column("ID")
        table.add_column("Name")
        table.add_column("Config")
        for i, exp in enumerate(manifest, 1):
            table.add_row(str(i), exp["id"], exp["name"], exp["config"])
        console.print(table)
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for i, experiment in enumerate(manifest, 1):
        console.rule(f"[bold]Experiment {i}/{len(manifest)}[/bold]")
        result = await run_single_experiment(experiment, args.limit, args.verbose)
        all_results.append(result)
        append_experiment_log(result)
        console.print()

    # ── Final comparison table ──────────────────────────────────────────────
    console.rule("[bold]Final Comparison[/bold]")
    completed = [r for r in all_results if r.get("status") == "completed"]

    if not completed:
        console.print("[red]No experiments completed successfully.[/red]")
        return

    # Get all metric names
    all_metrics = set()
    for r in completed:
        all_metrics.update(r.get("metrics", {}).keys())
    all_metrics = sorted(all_metrics)

    table = Table(title="Experiment Comparison", show_header=True, header_style="bold cyan")
    table.add_column("Experiment", style="bold", max_width=40)
    for m in all_metrics:
        table.add_column(m.replace("_", "\n"), justify="right", max_width=12)
    table.add_column("p50\n(ms)", justify="right")
    table.add_column("p95\n(ms)", justify="right")
    table.add_column("Errors", justify="right")

    for r in completed:
        row = [r["name"][:40]]
        for m in all_metrics:
            score = r.get("metrics", {}).get(m)
            if score is not None:
                color = "green" if score >= 0.8 else ("yellow" if score >= 0.5 else "red")
                row.append(f"[{color}]{score:.3f}[/{color}]")
            else:
                row.append("[dim]N/A[/dim]")
        row.append(f"{r.get('latency_p50_ms', 0):.0f}")
        row.append(f"{r.get('latency_p95_ms', 0):.0f}")
        row.append(f"{r.get('error_count', '?')}/{r.get('total_samples', '?')}")
        table.add_row(*row)

    console.print(table)

    # Save unified results
    unified_path = RESULTS_DIR / "all_experiments.json"
    with open(unified_path, "w") as f:
        json.dump(all_results, f, indent=2)
    console.print(f"\n[bold green]All results saved to: {unified_path}[/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
