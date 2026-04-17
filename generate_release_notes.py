#!/usr/bin/env python3
"""Generate weekly release notes for HydraDB by scanning merged PRs across all repos.

Usage:
    python generate_release_notes.py --days 7
    python generate_release_notes.py --days 7 --dry-run   # skip AI summarization

Requires:
    GITHUB_TOKEN  - GitHub personal access token with repo read access
    OPENAI_API_KEY - OpenAI API key (optional if --dry-run)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPOS: list[dict[str, str]] = [
    {"owner": "usecortex", "name": "cortex-application", "display": "Core API"},
    {"owner": "usecortex", "name": "cortex-ingestion", "display": "Ingestion Pipeline"},
    {"owner": "usecortex", "name": "cortex-dashboard", "display": "Dashboard"},
    {"owner": "usecortex", "name": "hydradb-on-prem-infra", "display": "On-Prem Infrastructure"},
    {"owner": "usecortex", "name": "hydradb-cli", "display": "CLI"},
    {"owner": "usecortex", "name": "hydradb-mcp", "display": "MCP Server"},
    {"owner": "usecortex", "name": "hydradb-claude-code", "display": "Claude Code Integration"},
    {"owner": "usecortex", "name": "hydradb-bench", "display": "Benchmarks"},
    {"owner": "usecortex", "name": "python-sdk", "display": "Python SDK"},
    {"owner": "usecortex", "name": "ts-sdk", "display": "TypeScript SDK"},
    {"owner": "usecortex", "name": "mintlify-docs", "display": "Documentation"},
    {"owner": "usecortex", "name": "docs", "display": "Docs (legacy)"},
    {"owner": "usecortex", "name": "openclaw-hydradb", "display": "OpenClaw"},
]

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Features": ["feat", "feature", "add", "new"],
    "Bug Fixes": ["fix", "bug", "hotfix", "patch", "resolve"],
    "Performance": ["perf", "optim", "speed", "latency", "cache"],
    "Security": ["security", "auth", "encrypt", "vulnerability", "cve"],
    "Infrastructure": ["infra", "deploy", "ci", "cd", "docker", "helm", "k8s", "argo"],
    "Documentation": ["doc", "readme", "guide", "cookbook"],
    "Chores": ["chore", "bump", "refactor", "cleanup", "lint", "format", "revert"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fetch_merged_prs(owner: str, name: str, since: datetime) -> list[dict]:
    """Fetch merged PRs from a GitHub repo using the gh CLI."""
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print(f"  WARNING: GITHUB_TOKEN not set, skipping {owner}/{name}", file=sys.stderr)
        return []

    cmd = [
        "gh", "pr", "list",
        "--repo", f"{owner}/{name}",
        "--state", "merged",
        "--json", "number,title,author,mergedAt,url,body,labels",
        "--limit", "100",
    ]
    env = {**os.environ, "GH_TOKEN": token}
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)
        if result.returncode != 0:
            print(f"  WARNING: gh failed for {owner}/{name}: {result.stderr.strip()}", file=sys.stderr)
            return []
        prs = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(f"  WARNING: error fetching {owner}/{name}: {exc}", file=sys.stderr)
        return []

    # Filter to the time window
    recent = []
    for pr in prs:
        merged_at = datetime.fromisoformat(pr["mergedAt"].replace("Z", "+00:00"))
        if merged_at >= since:
            pr["_merged_at"] = merged_at
            recent.append(pr)
    return recent


def categorize_pr(pr: dict) -> str:
    """Categorize a PR based on its title."""
    title_lower = pr["title"].lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in title_lower:
                return category
    return "Other"


def generate_ai_summary(categorized: dict, dry_run: bool = False) -> str | None:
    """Use OpenAI to generate a polished executive summary."""
    if dry_run:
        return None

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("  WARNING: OPENAI_API_KEY not set, skipping AI summary", file=sys.stderr)
        return None

    try:
        from openai import OpenAI
    except ImportError:
        print("  WARNING: openai package not installed, skipping AI summary", file=sys.stderr)
        return None

    # Build a compact representation of the PRs
    pr_list_text = ""
    for category, repos in categorized.items():
        pr_list_text += f"\n## {category}\n"
        for repo_display, prs in repos.items():
            for pr in prs:
                pr_list_text += f"- [{repo_display}] {pr['title']} (#{pr['number']})\n"

    prompt = f"""You are a technical writer for HydraDB, a vector database product.
Write a concise executive summary (3-5 paragraphs) of this week's release highlights.
Focus on user-facing impact. Group related changes together. Use professional tone.
Do NOT list every PR -- synthesize the key themes and improvements.

PRs merged this week:
{pr_list_text}
"""

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=1000,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HydraDB weekly release notes")
    parser.add_argument("--days", type=int, default=7, help="Look-back window in days (default: 7)")
    parser.add_argument("--dry-run", action="store_true", help="Skip AI summarization")
    parser.add_argument("--output-dir", default="reports", help="Output directory (default: reports)")
    args = parser.parse_args()

    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"Generating release notes for {args.days}-day window ending {today}")
    print(f"Cutoff: {since.isoformat()}")
    print()

    # Fetch PRs from all repos
    all_prs: list[tuple[dict, dict]] = []  # (repo_config, pr)
    for repo in REPOS:
        slug = f"{repo['owner']}/{repo['name']}"
        print(f"Fetching {slug}...")
        prs = fetch_merged_prs(repo["owner"], repo["name"], since)
        if prs:
            print(f"  Found {len(prs)} merged PRs")
            for pr in prs:
                all_prs.append((repo, pr))
        else:
            print(f"  No merged PRs in window")

    if not all_prs:
        print("\nNo merged PRs found in the time window. Nothing to report.")
        sys.exit(0)

    print(f"\nTotal: {len(all_prs)} merged PRs across {len({r['name'] for r, _ in all_prs})} repos")

    # Categorize
    categorized: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for repo, pr in all_prs:
        category = categorize_pr(pr)
        categorized[category][repo["display"]].append(pr)

    # Generate AI summary
    ai_summary = generate_ai_summary(categorized, dry_run=args.dry_run)

    # Build markdown
    lines: list[str] = []
    lines.append(f"# HydraDB Release Notes -- Week of {today}")
    lines.append("")
    lines.append(f"**Period:** {since.strftime('%B %d')} -- {datetime.now(timezone.utc).strftime('%B %d, %Y')}")
    lines.append(f"**Total PRs Merged:** {len(all_prs)}")
    active_repos = sorted({r["display"] for r, _ in all_prs})
    lines.append(f"**Active Repositories:** {', '.join(active_repos)}")
    lines.append("")

    if ai_summary:
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(ai_summary)
        lines.append("")

    lines.append("---")
    lines.append("")

    # Ordered categories
    category_order = [
        "Features", "Bug Fixes", "Performance", "Security",
        "Infrastructure", "Documentation", "Chores", "Other",
    ]
    for category in category_order:
        if category not in categorized:
            continue
        repos = categorized[category]
        total = sum(len(prs) for prs in repos.values())
        lines.append(f"## {category} ({total})")
        lines.append("")
        for repo_display in sorted(repos.keys()):
            prs = repos[repo_display]
            lines.append(f"### {repo_display}")
            lines.append("")
            for pr in sorted(prs, key=lambda p: p["number"], reverse=True):
                author = pr["author"].get("login", "unknown")
                url = pr["url"]
                lines.append(f"- **[#{pr['number']}]({url})** {pr['title']} _(by @{author})_")
            lines.append("")

    # Stats
    lines.append("---")
    lines.append("")
    lines.append("## Contributors")
    lines.append("")
    contributors: dict[str, int] = defaultdict(int)
    for _, pr in all_prs:
        author = pr["author"].get("login", "unknown")
        if not pr["author"].get("is_bot", False):
            contributors[author] += 1
    for author, count in sorted(contributors.items(), key=lambda x: -x[1]):
        lines.append(f"- @{author} ({count} PRs)")
    lines.append("")

    # Bot contributions
    bot_count = sum(1 for _, pr in all_prs if pr["author"].get("is_bot", False))
    if bot_count:
        lines.append(f"- Automated (Vorflux bot): {bot_count} PRs")
        lines.append("")

    content = "\n".join(lines)

    # Write output
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"release-notes-{today}.md"
    out_path.write_text(content, encoding="utf-8")
    print(f"\nRelease notes written to: {out_path}")
    print(f"Length: {len(content)} chars, {len(lines)} lines")


if __name__ == "__main__":
    main()
