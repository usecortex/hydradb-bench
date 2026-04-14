#!/usr/bin/env python3
"""Generate weekly release notes for HydraDB.

Fetches merged PRs from GitHub, optionally enriches with Slack context,
uses OpenAI for semantic analysis, and outputs impact-driven release notes.

Usage:
    python generate_release_notes.py
    python generate_release_notes.py --days 14 --verbose
    python generate_release_notes.py --dry-run
    python generate_release_notes.py --repos usecortex/cortex-application
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from release_notes.analyzer import analyze_changes
from release_notes.formatter import format_release_notes, save_release_notes
from release_notes.github_collector import DEFAULT_REPOS, fetch_merged_prs
from release_notes.models import EnrichedChange
from release_notes.slack_collector import fetch_slack_messages, match_slack_context

console = Console()
logger = logging.getLogger("release_notes")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate weekly release notes for HydraDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables:
  GITHUB_TOKEN     Required. GitHub personal access token with repo read access.
  OPENAI_API_KEY   Required. OpenAI API key for semantic analysis.
  SLACK_BOT_TOKEN  Optional. Slack bot token for context enrichment.

Examples:
  python generate_release_notes.py                    # Default: last 7 days, both repos
  python generate_release_notes.py --days 14          # Last 14 days
  python generate_release_notes.py --dry-run          # Show PRs without generating notes
  python generate_release_notes.py --skip-slack       # Skip Slack enrichment
  python generate_release_notes.py --verbose          # Debug logging
        """,
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Lookback window in days (default: 7)",
    )
    parser.add_argument(
        "--repos",
        type=str,
        default=None,
        help="Comma-separated list of repos (default: cortex-application,cortex-ingestion)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports",
        help="Output directory for release notes (default: reports/)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model for analysis (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show PRs found but don't generate release notes",
    )
    parser.add_argument(
        "--skip-slack",
        action="store_true",
        help="Skip Slack context enrichment",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    """Main async entry point."""
    repos = args.repos.split(",") if args.repos else DEFAULT_REPOS

    # Step 1: Fetch merged PRs from GitHub
    console.print(f"\n[bold blue]Fetching merged PRs from last {args.days} days...[/]")
    console.print(f"  Repos: {', '.join(repos)}")

    try:
        prs = await fetch_merged_prs(repos=repos, days=args.days)
    except RuntimeError as e:
        console.print(f"[bold red]Error:[/] {e}")
        return 1
    except Exception as e:
        console.print(f"[bold red]Failed to fetch PRs:[/] {e}")
        return 1

    if not prs:
        console.print("[yellow]No merged PRs found in the given time window.[/]")
        return 0

    # Display found PRs
    table = Table(title=f"Merged PRs ({len(prs)} found)")
    table.add_column("PR", style="cyan", width=8)
    table.add_column("Repository", style="dim", width=30)
    table.add_column("Title", width=50)
    table.add_column("Author", style="green", width=15)
    table.add_column("Merged", style="dim", width=12)

    for pr in prs:
        table.add_row(
            f"#{pr.number}",
            pr.repo_name.split("/")[-1],
            pr.title[:50],
            pr.author,
            pr.merged_at.strftime("%Y-%m-%d"),
        )

    console.print(table)

    if args.dry_run:
        console.print("\n[yellow]Dry run — skipping analysis and generation.[/]")
        return 0

    # Step 2: Optionally fetch Slack context
    slack_messages = []
    if not args.skip_slack:
        console.print("\n[bold blue]Fetching Slack context...[/]")
        slack_messages = await fetch_slack_messages(days=args.days)
        if slack_messages:
            console.print(f"  Found {len(slack_messages)} Slack messages")
        else:
            console.print("  [dim]No Slack messages (token not set or no messages found)[/]")

    # Step 3: Enrich PRs with Slack context
    enriched: list[EnrichedChange] = []
    for pr in prs:
        slack_ctx = match_slack_context(pr, slack_messages)
        enriched.append(EnrichedChange(pr=pr, slack_context=slack_ctx))

    # Step 4: Semantic analysis via OpenAI
    console.print(f"\n[bold blue]Analyzing {len(enriched)} PRs with OpenAI ({args.model})...[/]")
    try:
        notes = analyze_changes(enriched, days=args.days, model=args.model)
    except RuntimeError as e:
        console.print(f"[bold red]Error:[/] {e}")
        return 1
    except Exception as e:
        console.print(f"[bold red]Analysis failed:[/] {e}")
        return 1

    # Step 5: Format and save
    filepath = save_release_notes(notes, output_dir=args.output)

    # Display results
    content = format_release_notes(notes)
    console.print()
    console.print(Panel(content, title="Release Notes", border_style="green"))

    # Summary
    total = len(notes.features) + len(notes.improvements) + len(notes.fixes) + len(notes.internal_changes)
    console.print(f"\n[bold green]Done![/] {total} significant changes from {len(prs)} PRs")
    console.print(f"  Features:     {len(notes.features)}")
    console.print(f"  Improvements: {len(notes.improvements)}")
    console.print(f"  Fixes:        {len(notes.fixes)}")
    console.print(f"  Internal:     {len(notes.internal_changes)}")
    console.print(f"\n  Saved to: [bold]{filepath}[/]")

    return 0


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    exit_code = asyncio.run(run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
