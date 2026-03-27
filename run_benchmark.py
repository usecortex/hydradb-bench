#!/usr/bin/env python3
"""
HydraDB Benchmark — full RAGAS feature coverage.

Usage:
    python run_benchmark.py                                       # standard run
    python run_benchmark.py --skip-ingestion                      # skip doc upload
    python run_benchmark.py --generate-testset                    # auto-gen Q&A from docs
    python run_benchmark.py --multi-turn                          # also run multi-turn eval
    python run_benchmark.py --save-prompts                        # dump metric prompts
    python run_benchmark.py --config my_config.yaml               # custom config path
    python run_benchmark.py --hf-dataset explodinggradients/amnesty_qa          # HF Q&A dataset
    python run_benchmark.py --hf-dataset isaacus/legal-rag-bench --hf-mode corpus  # HF corpus
    python run_benchmark.py --print-hf-info explodinggradients/amnesty_qa       # inspect dataset
    python run_benchmark.py --extract-qa-corpus PatronusAI/financebench ./data/corpus --group-by doc_name
    python run_benchmark.py --extract-qa-corpus explodinggradients/amnesty_qa ./data/corpus
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hydradb_bench.checkpoint import CheckpointManager
from hydradb_bench.client import HydraDBClient
from hydradb_bench.slack_notifier import send_results as slack_send
from hydradb_bench.config import load_config
from hydradb_bench.evaluator import RAGASEvaluator
from hydradb_bench.hf_loader import (
    extract_qa_corpus,
    load_corpus_as_documents,
    load_qa_dataset,
    print_dataset_info,
)
from hydradb_bench.ingestion import IngestionOrchestrator
from hydradb_bench.models import BenchmarkConfig, BenchmarkResult, DatasetEntry, TokenUsageResult
from hydradb_bench.multi_turn import (
    MultiTurnRunner,
    load_multi_turn_dataset,
    multi_turn_samples_to_evaluator_input,
)
from hydradb_bench.reporter import BenchmarkReporter, _compute_latency_stats
from hydradb_bench.runner import BenchmarkRunner
from hydradb_bench.testset_generator import TestsetGeneratorWrapper

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not verbose:
        for name in ("httpx", "httpcore", "openai", "langchain", "urllib3"):
            logging.getLogger(name).setLevel(logging.WARNING)


def _print_scores(title: str, scores: dict[str, float]) -> None:
    if not scores:
        return
    table = Table(title=title, show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Rating")
    for metric, score in scores.items():
        pct = f"{score:.1%}"
        if score >= 0.8:
            rating = "[green]Good[/green]"
        elif score >= 0.5:
            rating = "[yellow]Fair[/yellow]"
        else:
            rating = "[red]Poor[/red]"
        table.add_row(metric.replace("_", " ").title(), pct, rating)
    console.print(table)


_LOCAL_EMBEDDING_PROVIDERS = {"huggingface", "ollama"}


def _validate_llm_api_key(ragas_config) -> str | None:
    """
    Validate API key(s) for the configured LLM and embeddings providers.
    Makes a cheap live call (models.list) — no tokens consumed.
    Returns an error string on failure, None on success.
    Local embedding providers (huggingface, ollama) don't need an API key.
    """
    import os
    try:
        from openai import OpenAI, AuthenticationError, APIConnectionError
    except ImportError:
        return None  # openai not installed — skip validation

    def _check(provider: str, label: str) -> str | None:
        if provider == "openrouter":
            key_name, base_url = "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"
        else:
            key_name, base_url = "OPENAI_API_KEY", None

        api_key = os.environ.get(key_name, "")
        if not api_key or api_key.startswith("your-") or api_key == key_name:
            return f"{label}: {key_name} is not set or is a placeholder value."
        try:
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            OpenAI(**kwargs).models.list()
            return None
        except AuthenticationError as e:
            return f"{label}: Invalid {key_name} — {e}"
        except APIConnectionError as e:
            return f"{label}: Cannot reach {provider} API — {e}"
        except Exception as e:
            return f"{label}: API key check failed ({type(e).__name__}) — {e}"

    # Check LLM key
    err = _check(ragas_config.llm_provider, "LLM")
    if err:
        return err

    # Check embeddings key only when it requires a remote API
    emb_provider = getattr(ragas_config, "embeddings_provider", "openai")
    if emb_provider not in _LOCAL_EMBEDDING_PROVIDERS:
        err = _check(emb_provider, "Embeddings")
        if err:
            return err

    return None


async def _run_dataset_pipeline(
    config: BenchmarkConfig,
    args: argparse.Namespace,
    run_id: str,
    dataset_name: str | None = None,
) -> int:
    """Run the full ingest → query → evaluate → report pipeline for one dataset."""

    _dataset_label = dataset_name or config.name
    checkpoint = CheckpointManager(run_id, _dataset_label)
    if checkpoint.completed_count:
        console.print(
            f"  [yellow]Resume mode:[/yellow] {checkpoint.completed_count} sample(s) already done "
            f"for dataset '{_dataset_label}' (run {run_id})"
        )

    async with HydraDBClient(config.hydradb) as client:

        # ── 3. Ingest documents ─────────────────────────────────────────
        console.print("\n[bold]3/6  Document ingestion...[/bold]")
        if args.reset_tenant:
            console.print("  [yellow]--reset-tenant:[/yellow] deleting all existing tenant data...")
            try:
                result = await client.delete_tenant()
                console.print(f"  [green]Tenant deleted.[/green] ({result.get('status', 'ok')})")
                await client.create_tenant()
                console.print("  [green]Tenant recreated.[/green]")
            except Exception as e:
                console.print(f"  [red]Tenant deletion failed:[/red] {e}")
                return 1

        if args.skip_ingestion:
            console.print("  [yellow]Skipped[/yellow] (--skip-ingestion)")
        else:
            orchestrator = IngestionOrchestrator(client, config.ingestion, config.hydradb)
            ingestion_report = await orchestrator.run()
            console.print(f"  {ingestion_report.summary()}")

        # ── 4. Run single-turn queries ──────────────────────────────────
        console.print("\n[bold]4/6  Running single-turn benchmark queries...[/bold]")
        runner = BenchmarkRunner(client, config.evaluation)

        if config.hf_dataset.enabled and config.hf_dataset.mode == "qa":
            hf = config.hf_dataset
            if not hf.repo_id:
                console.print("[red]hf_dataset.repo_id is required in qa mode.[/red]")
                return 1
            try:
                test_samples = load_qa_dataset(
                    repo_id=hf.repo_id,
                    split=hf.split,
                    config_name=hf.config_name or None,
                    column_map=hf.column_map or None,
                    max_samples=hf.max_samples,
                    id_prefix=hf.id_prefix,
                    save_path=hf.save_qa_path,
                )
            except Exception as e:
                console.print(f"  [red]HF Q&A load failed:[/red] {e}")
                return 1
        else:
            try:
                test_samples = runner.load_test_dataset(config.evaluation.test_dataset_path)
            except FileNotFoundError as e:
                console.print(f"  [red]Dataset error:[/red] {e}")
                return 1

        if args.limit:
            test_samples = test_samples[: args.limit]
            console.print(f"  [yellow]--limit {args.limit}:[/yellow] running first {len(test_samples)} sample(s)")

        new_samples = await runner.run(test_samples, checkpoint=checkpoint)
        restored_samples = checkpoint.restore(test_samples)
        # Merge: restored (already-done) + new results, preserving original order
        new_by_id = {s.test_sample.id: s for s in new_samples}
        benchmark_samples = [
            new_by_id.get(ts.id) or next((r for r in restored_samples if r.test_sample.id == ts.id), None)
            for ts in test_samples
        ]
        benchmark_samples = [s for s in benchmark_samples if s is not None]
        error_count = sum(1 for s in benchmark_samples if s.error)
        console.print(
            f"  Completed: [bold]{len(benchmark_samples)}[/bold] queries "
            f"([red]{error_count}[/red] errors)"
        )

        # ── 4b. (Optional) Multi-turn queries ──────────────────────────
        multi_turn_samples = []
        run_multi_turn = args.multi_turn or config.evaluation.multi_turn.enabled
        if run_multi_turn:
            console.print("\n[bold]4b/6  Running multi-turn conversations...[/bold]")
            try:
                conversations = load_multi_turn_dataset(
                    config.evaluation.multi_turn.dataset_path
                )
                mt_runner = MultiTurnRunner(client, config.evaluation)
                multi_turn_samples = await mt_runner.run(conversations)
                mt_errors = sum(1 for s in multi_turn_samples if s.error)
                console.print(
                    f"  Completed: [bold]{len(multi_turn_samples)}[/bold] conversations "
                    f"([red]{mt_errors}[/red] errors)"
                )
            except FileNotFoundError as e:
                console.print(f"  [yellow]Multi-turn dataset not found:[/yellow] {e}")

    # ── 5. RAGAS evaluation ─────────────────────────────────────────────
    console.print("\n[bold]5/6  Running RAGAS evaluation...[/bold]")
    console.print("  [dim]Calls OpenAI for LLM-based metrics — see cost section in HTML report[/dim]")

    evaluator = RAGASEvaluator(config.ragas)

    if args.save_prompts:
        evaluator.save_prompts(config.evaluation.metrics, config.output_dir)
        console.print(f"  [blue]Prompts saved to {config.output_dir}/prompts/[/blue]")

    aggregate_scores, per_sample_scores, token_usage = evaluator.evaluate(
        samples=benchmark_samples,
        metric_names=config.evaluation.metrics,
        aspect_configs=config.evaluation.aspect_critics,
        scored_criteria_configs=config.evaluation.scored_criteria,
        include_reasons=config.reporting.include_reasons,
    )
    _print_scores("Single-Turn RAGAS Scores", aggregate_scores)

    if token_usage.total_tokens > 0:
        console.print(
            f"  Tokens: [bold]{token_usage.total_tokens:,}[/bold] "
            f"(in={token_usage.input_tokens:,}, out={token_usage.output_tokens:,}) "
            f"Cost (judge LLM): [bold]${token_usage.actual_cost_usd:.4f}[/bold]"
        )

    multi_turn_scores: dict[str, float] = {}
    mt_token_usage = TokenUsageResult()
    if run_multi_turn and multi_turn_samples:
        console.print("\n  [dim]Running multi-turn evaluation...[/dim]")
        evaluator_input = multi_turn_samples_to_evaluator_input(multi_turn_samples)
        if evaluator_input:
            multi_turn_scores, mt_token_usage = evaluator.evaluate_multi_turn(
                conversations=evaluator_input,
                metric_names=config.evaluation.multi_turn.metrics,
            )
            _print_scores("Multi-Turn RAGAS Scores", multi_turn_scores)

    # ── 6. Generate reports ─────────────────────────────────────────────
    console.print("\n[bold]6/6  Generating reports...[/bold]")

    latency_stats = _compute_latency_stats(benchmark_samples)

    combined_token_usage = TokenUsageResult(
        input_tokens=token_usage.input_tokens + mt_token_usage.input_tokens,
        output_tokens=token_usage.output_tokens + mt_token_usage.output_tokens,
        total_tokens=token_usage.total_tokens + mt_token_usage.total_tokens,
        actual_cost_usd=round(
            token_usage.actual_cost_usd + mt_token_usage.actual_cost_usd, 6
        ),
        model=token_usage.model or mt_token_usage.model,
    )

    result = BenchmarkResult(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        benchmark_name=config.name,
        config_snapshot=config.model_dump(exclude={"hydradb": {"api_key"}}),
        samples=benchmark_samples,
        multi_turn_samples=multi_turn_samples,
        ragas_scores=aggregate_scores,
        multi_turn_scores=multi_turn_scores,
        per_sample_scores=per_sample_scores,
        latency_stats=latency_stats,
        token_usage=combined_token_usage,
        error_count=error_count,
        evaluated_count=len(per_sample_scores),
    )

    reporter = BenchmarkReporter(config.reporting)
    output_paths = reporter.generate(result, output_dir=config.output_dir)
    for path in output_paths:
        console.print(f"  [blue]->[/blue] {path}")

    checkpoint.delete()

    # ── 6b. Slack notification ──────────────────────────────────────────
    if config.slack.enabled:
        console.print("\n[bold]6b/6  Sending reports to Slack...[/bold]")
        try:
            await slack_send(config.slack, result, output_paths)
            console.print("  [green]Sent to Slack[/green]")
        except Exception as e:
            console.print(f"  [red]Slack notification failed:[/red] {e}")

    return 0


async def _main_async(args: argparse.Namespace) -> int:
    console.print(Panel.fit(
        "[bold cyan]HydraDB Benchmark Framework[/bold cyan]\n"
        "[dim]Full RAGAS feature coverage — v0.4.x[/dim]",
        border_style="cyan",
    ))

    # ── 0. Fast paths: no config/API keys needed ────────────────────────
    if args.print_hf_info:
        print_dataset_info(args.print_hf_info, split="test")
        return 0

    if args.extract_qa_corpus:
        repo_id, output_dir = args.extract_qa_corpus
        extract_qa_corpus(
            repo_id=repo_id,
            split=args.hf_split or "train",
            context_column=args.context_column,
            group_by_column=args.group_by,
            output_dir=output_dir,
        )
        console.print(
            f"\n[bold]Next steps:[/bold]\n"
            f"  1. Set ingestion.documents_dir: \"{output_dir}\" in your config YAML\n"
            f"  2. Run: python run_benchmark.py --config config/benchmark.yaml\n"
            f"     (without --skip-ingestion so docs are uploaded first)"
        )
        return 0

    # ── 1. Load config ──────────────────────────────────────────────────
    console.print("\n[bold]1/6  Loading configuration...[/bold]")
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        return 1

    # CLI flags override config
    if args.hf_dataset:
        config.hf_dataset.enabled = True
        config.hf_dataset.repo_id = args.hf_dataset
    if args.hf_mode:
        config.hf_dataset.mode = args.hf_mode

    run_id = args.resume or args.run_id or config.run_id or str(uuid.uuid4())[:8]
    if args.resume:
        console.print(f"  [yellow]Resuming run:[/yellow] [bold]{run_id}[/bold]")
    console.print(f"  Run ID     : [bold]{run_id}[/bold]")
    console.print(f"  Tenant     : [bold]{config.hydradb.tenant_id}[/bold]")
    console.print(f"  Sub-tenant : [bold]{config.hydradb.sub_tenant_id or '(default)'}[/bold]")
    endpoint_label = config.evaluation.search_endpoint
    if config.evaluation.search_endpoint == "full_recall":
        endpoint_label += f" ({config.evaluation.retrieve_mode} mode)"
    console.print(f"  Endpoint   : [bold]{endpoint_label}[/bold]")
    console.print(f"  Metrics    : {', '.join(config.evaluation.metrics)}")
    if config.evaluation.aspect_critics:
        console.print(
            f"  Aspects    : {', '.join(a.name for a in config.evaluation.aspect_critics)}"
        )
    if config.evaluation.scored_criteria:
        console.print(
            f"  Scored     : {', '.join(s.name for s in config.evaluation.scored_criteria)}"
        )
    if config.hf_dataset.enabled:
        console.print(
            f"  HF Dataset : [bold]{config.hf_dataset.repo_id}[/bold] "
            f"(mode={config.hf_dataset.mode}, split={config.hf_dataset.split})"
        )

    # ── 1b. Validate LLM API key before spending time on ingestion/queries ─
    console.print("\n[bold]1b/6  Validating LLM API key...[/bold]")
    key_error = _validate_llm_api_key(config.ragas)
    if key_error:
        console.print(f"[red]API key validation failed:[/red] {key_error}")
        console.print(
            "[yellow]Fix your API key in .env before running the benchmark.[/yellow]\n"
            "  OpenAI:     OPENAI_API_KEY=sk-...\n"
            "  OpenRouter: OPENROUTER_API_KEY=sk-or-..."
        )
        return 1
    console.print("  [green]API key OK[/green]")

    # ── 1c. HuggingFace corpus — download docs, override ingestion dir ───
    if config.hf_dataset.enabled and config.hf_dataset.mode == "corpus":
        hf = config.hf_dataset
        if not hf.repo_id:
            console.print("[red]hf_dataset.repo_id is required in corpus mode.[/red]")
            return 1
        console.print(f"\n[bold]HF Corpus:[/bold] downloading {hf.repo_id!r} -> {hf.corpus_output_dir}")
        try:
            load_corpus_as_documents(
                repo_id=hf.repo_id,
                split=hf.split,
                text_column=hf.text_column,
                title_column=hf.title_column,
                max_docs=hf.max_docs,
                output_dir=hf.corpus_output_dir,
            )
        except Exception as e:
            console.print(f"[red]HF corpus download failed:[/red] {e}")
            return 1
        # Override ingestion dir so HydraDB uploads these docs
        config.ingestion.documents_dir = hf.corpus_output_dir

    # ── 2. (Optional) Generate testset from documents ───────────────────
    run_testset_gen = args.generate_testset or config.testset_generation.enabled
    if run_testset_gen:
        console.print("\n[bold]2/6  Generating testset from documents...[/bold]")
        gen = TestsetGeneratorWrapper(config.ragas, config.testset_generation)
        try:
            gen.generate(
                documents_dir=config.ingestion.documents_dir,
                extensions=config.ingestion.file_extensions,
                save=True,
            )
        except Exception as e:
            console.print(f"  [red]Testset generation failed:[/red] {e}")
            console.print("  [yellow]Falling back to existing test_dataset.json[/yellow]")
    else:
        console.print("\n[bold]2/6  Testset generation skipped[/bold] [dim](--generate-testset to enable)[/dim]")

    # ── 3–6. Run dataset pipeline (single or multi-dataset) ─────────────
    if config.datasets:
        datasets_to_run = config.datasets
        if args.dataset:
            datasets_to_run = [
                d for d in config.datasets
                if d.name.lower() == args.dataset.lower()
                or d.sub_tenant_id.lower() == args.dataset.lower()
            ]
            if not datasets_to_run:
                names = ", ".join(d.name for d in config.datasets)
                console.print(f"[red]--dataset {args.dataset!r} not found. Available: {names}[/red]")
                return 1

        console.print(
            f"\n[bold cyan]Multi-dataset mode:[/bold cyan] "
            f"{len(datasets_to_run)} dataset(s) — "
            + ", ".join(f"[bold]{d.name}[/bold]" for d in datasets_to_run)
        )
        for i, dataset in enumerate(datasets_to_run, 1):
            console.print(
                f"\n[bold]---  Dataset {i}/{len(config.datasets)}: "
                f"{dataset.name}  ---[/bold]"
            )
            # Override per-dataset fields on a copy of the config
            dataset_config = config.model_copy(deep=True)
            dataset_config.name = f"{config.name} — {dataset.name}"
            dataset_config.hydradb.sub_tenant_id = dataset.sub_tenant_id
            dataset_config.ingestion.documents_dir = dataset.documents_dir
            dataset_config.evaluation.test_dataset_path = dataset.test_dataset_path

            rc = await _run_dataset_pipeline(dataset_config, args, run_id, dataset_name=dataset.name)
            if rc != 0:
                console.print(f"  [red]Dataset {dataset.name} failed — continuing...[/red]")

        console.print("\n[bold green]All datasets complete![/bold green]")
    else:
        rc = await _run_dataset_pipeline(config, args, run_id)
        if rc != 0:
            return rc
        console.print("\n[bold green]Benchmark complete![/bold green]")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HydraDB benchmark with full RAGAS feature coverage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", default="config/benchmark.yaml",
        help="Path to benchmark.yaml (default: config/benchmark.yaml)",
    )
    parser.add_argument(
        "--skip-ingestion", action="store_true",
        help="Skip document upload (use when docs are already indexed)",
    )
    parser.add_argument(
        "--reset-tenant", action="store_true",
        help="Delete all existing tenant data then re-ingest from scratch (irreversible)",
    )
    parser.add_argument(
        "--generate-testset", action="store_true",
        help="Auto-generate Q&A pairs from documents using RAGAS TestsetGenerator",
    )
    parser.add_argument(
        "--multi-turn", action="store_true",
        help="Also run multi-turn conversation evaluation",
    )
    parser.add_argument(
        "--save-prompts", action="store_true",
        help="Dump metric prompt instructions to reports/prompts/ for inspection",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Custom run ID (auto-generated if not provided)",
    )
    parser.add_argument(
        "--hf-dataset", default=None, metavar="REPO_ID",
        help="HuggingFace dataset repo ID to use (overrides hf_dataset.repo_id in config)",
    )
    parser.add_argument(
        "--hf-mode", default=None, choices=["qa", "corpus"],
        help="HF dataset mode: 'qa' (evaluation pairs) or 'corpus' (documents for ingestion)",
    )
    parser.add_argument(
        "--print-hf-info", default=None, metavar="REPO_ID",
        help="Print schema/preview of a HuggingFace dataset and exit (useful before benchmarking)",
    )
    parser.add_argument(
        "--extract-qa-corpus", default=None, nargs=2, metavar=("REPO_ID", "OUTPUT_DIR"),
        help="Extract context passages from any HF Q&A dataset as .txt files for HydraDB ingestion. "
             "Example: --extract-qa-corpus PatronusAI/financebench ./data/corpus",
    )
    parser.add_argument(
        "--context-column", default=None, metavar="COL",
        help="Column containing context passages (auto-detected if not set)",
    )
    parser.add_argument(
        "--group-by", default=None, metavar="COL",
        help="Group passages by this column when extracting corpus (e.g. 'doc_name'). "
             "Creates one file per unique value instead of one per row.",
    )
    parser.add_argument(
        "--hf-split", default=None, metavar="SPLIT",
        help="HuggingFace dataset split to use (default: auto-detect or 'train')",
    )
    parser.add_argument(
        "--dataset", default=None, metavar="NAME",
        help="Run only one dataset from a multi-dataset config (match by name or sub_tenant_id). "
             "Example: --dataset PrivacyQA",
    )
    parser.add_argument(
        "--resume", default=None, metavar="RUN_ID",
        help="Resume an interrupted run by its run ID (skips already-completed samples)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Run only the first N samples (useful for quick smoke tests)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose/debug logging",
    )
    args = parser.parse_args()
    _setup_logging(args.verbose)
    sys.exit(asyncio.run(_main_async(args)))


if __name__ == "__main__":
    main()
