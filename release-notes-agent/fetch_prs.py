#!/usr/bin/env python3
"""
Fetch merged PRs from HydraDB product repositories.

Uses the GitHub CLI (`gh`) which must be authenticated.
Outputs structured JSON to stdout for consumption by the release notes agent.

Usage:
    python fetch_prs.py                     # last 7 days, default repos
    python fetch_prs.py --days 14           # last 14 days
    python fetch_prs.py --repos cortex-application cortex-ingestion
    python fetch_prs.py --include-infra     # include infra repos too
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

# Ensure imports work regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_REPOS,
    GITHUB_ORG,
    INFRA_REPOS,
    NOISE_PREFIXES,
    NOISE_SUBSTRINGS,
    REFACTOR_KEYWORDS,
)


def run_gh(args: list[str], default: str = "[]") -> str:
    """Run a gh CLI command and return stdout. Returns `default` on failure."""
    result = subprocess.run(
        ["gh"] + args,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        print(f"[WARN] gh command failed: gh {' '.join(args)}", file=sys.stderr)
        print(f"  stderr: {result.stderr.strip()}", file=sys.stderr)
        return default
    return result.stdout


def fetch_commits_for_pr(repo: str, pr_number: int) -> list[dict[str, Any]]:
    """Fetch commits for a specific PR (separate call to avoid GraphQL complexity limits)."""
    raw = run_gh(
        ["pr", "view", str(pr_number), "--repo", f"{GITHUB_ORG}/{repo}", "--json", "commits"],
        default='{"commits": []}',
    )
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data.get("commits", [])
        return []
    except (json.JSONDecodeError, AttributeError):
        return []


def fetch_merged_prs(repo: str, since: datetime) -> list[dict[str, Any]]:
    """Fetch PRs merged after `since` from a given repo."""
    # Fetch PR list without commits (avoids GraphQL complexity limit)
    raw = run_gh([
        "pr", "list",
        "--repo", f"{GITHUB_ORG}/{repo}",
        "--state", "merged",
        "--json", "number,title,body,mergedAt,author,labels,headRefName,url",
        "--limit", "50",
    ])
    prs = json.loads(raw)

    filtered = []
    for pr in prs:
        merged_at_str = pr.get("mergedAt", "")
        if not merged_at_str:
            continue
        merged_at = datetime.fromisoformat(merged_at_str.replace("Z", "+00:00"))
        if merged_at >= since:
            pr["repo"] = repo
            pr["merged_at_parsed"] = merged_at.isoformat()
            # Fetch commits separately for each qualifying PR
            pr["commits"] = fetch_commits_for_pr(repo, pr["number"])
            filtered.append(pr)

    return filtered


def is_noise(pr: dict[str, Any]) -> bool:
    """Check if a PR is noise that should be excluded."""
    title = pr.get("title", "").strip()
    title_lower = title.lower()

    # Check noise prefixes (case-insensitive)
    for prefix in NOISE_PREFIXES:
        if title_lower.startswith(prefix.lower()):
            return True

    # Check noise substrings (case-insensitive)
    for substring in NOISE_SUBSTRINGS:
        if substring.lower() in title_lower:
            return True

    # Draft/WIP
    if title_lower.startswith("wip") or title_lower.startswith("[wip]") or title_lower.startswith("draft"):
        return True

    # Single-word titles like "Staging" are almost always noise
    if len(title.split()) <= 1 and not title_lower.startswith("feat"):
        return True

    return False


def is_pure_refactor(pr: dict[str, Any]) -> bool:
    """Check if a PR is a pure refactor with no user/system impact."""
    title = pr.get("title", "").strip()
    title_lower = title.lower()
    body = pr.get("body", "") or ""

    is_refactor = any(kw.lower() in title_lower for kw in REFACTOR_KEYWORDS)
    if not is_refactor:
        return False

    # If the body mentions impact, performance, fix, or user-facing changes, keep it
    impact_signals = [
        "impact", "performance", "latency", "throughput", "user",
        "customer", "breaking", "fix", "bug", "security", "reliability",
    ]
    body_lower = body.lower()
    for signal in impact_signals:
        if signal in body_lower:
            return False

    return True


def extract_release_notes_section(body: str) -> str | None:
    """Extract the '## Release Notes' section from a PR body if present."""
    if not body:
        return None

    pattern = r"##\s*Release\s*Notes?\s*\n(.*?)(?=\n##\s|\Z)"
    match = re.search(pattern, body, re.IGNORECASE | re.DOTALL)
    if match:
        content = match.group(1).strip()
        if content and content != "-" and len(content) > 5:
            return content
    return None


def extract_summary_section(body: str) -> str | None:
    """Extract the '## Summary' section from a PR body."""
    if not body:
        return None

    pattern = r"##\s*Summary\s*\n(.*?)(?=\n##\s|\Z)"
    match = re.search(pattern, body, re.IGNORECASE | re.DOTALL)
    if match:
        content = match.group(1).strip()
        if content and len(content) > 5:
            return content
    return None


def extract_key_changes_section(body: str) -> str | None:
    """Extract the '## Key Changes' section from a PR body."""
    if not body:
        return None

    pattern = r"##\s*Key\s*Changes?\s*\n(.*?)(?=\n##\s|\Z)"
    match = re.search(pattern, body, re.IGNORECASE | re.DOTALL)
    if match:
        content = match.group(1).strip()
        if content and len(content) > 5:
            return content
    return None


def extract_why_it_matters_section(body: str) -> str | None:
    """Extract the '## Why it matters' section from a PR body."""
    if not body:
        return None

    pattern = r"##\s*Why\s*[Ii]t\s*[Mm]atters?\s*\n(.*?)(?=\n##\s|\Z)"
    match = re.search(pattern, body, re.IGNORECASE | re.DOTALL)
    if match:
        content = match.group(1).strip()
        if content and len(content) > 5:
            return content
    return None


def classify_pr(title: str, body: str) -> str:
    """Classify a PR into a release notes category based on title and body."""
    title_lower = title.lower().strip()
    body_lower = (body or "").lower()

    # Feature
    if any(title_lower.startswith(p) for p in ["feat:", "feat/", "feature:"]):
        return "feature"

    # Fix
    if any(title_lower.startswith(p) for p in ["fix:", "fix/", "bugfix:", "hotfix:"]):
        return "fix"

    # Performance
    if any(title_lower.startswith(p) for p in ["perf:", "performance:"]):
        return "improvement"

    # Internal / chore
    if any(title_lower.startswith(p) for p in ["chore:", "refactor:", "build:"]):
        return "internal"

    # Docs
    if any(title_lower.startswith(p) for p in ["docs:", "doc:"]):
        return "docs"

    # Infer from body or title keywords
    if "improve" in title_lower or "enhance" in title_lower or "optimize" in title_lower:
        return "improvement"
    if "fix" in title_lower:
        return "fix"
    if "feat" in title_lower or title_lower.startswith("add ") or title_lower.startswith("add:"):
        return "feature"

    # Check body for signals
    if any(kw in body_lower for kw in ["new feature", "adds support", "introduces"]):
        return "feature"
    if any(kw in body_lower for kw in ["fixes", "resolves", "bug", "regression"]):
        return "fix"

    return "other"


def build_pr_summary(pr: dict[str, Any]) -> dict[str, Any]:
    """Build a structured summary of a PR for the release notes agent."""
    body = pr.get("body", "") or ""
    title = pr.get("title", "")

    # Extract commit messages (first line only)
    commits = pr.get("commits", []) or []
    commit_messages = []
    for c in commits:
        headline = c.get("messageHeadline", "")
        if headline and not headline.startswith("Merge"):
            commit_messages.append(headline)

    category = classify_pr(title, body)

    author = pr.get("author", {})
    author_login = author.get("login", "unknown") if author else "unknown"

    return {
        "repo": pr.get("repo", ""),
        "number": pr.get("number"),
        "title": title,
        "url": pr.get("url", ""),
        "author": author_login,
        "merged_at": pr.get("merged_at_parsed", ""),
        "branch": pr.get("headRefName", ""),
        "category": category,
        "labels": [l.get("name", "") for l in (pr.get("labels") or [])],
        "release_notes": extract_release_notes_section(body),
        "summary": extract_summary_section(body),
        "key_changes": extract_key_changes_section(body),
        "why_it_matters": extract_why_it_matters_section(body),
        "commit_messages": commit_messages,
        "body_preview": body[:2000] if body else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch merged PRs for release notes")
    parser.add_argument("--days", type=int, default=DEFAULT_LOOKBACK_DAYS,
                        help=f"Lookback window in days (default: {DEFAULT_LOOKBACK_DAYS})")
    parser.add_argument("--repos", nargs="+", default=None,
                        help="Specific repos to fetch (default: core + product)")
    parser.add_argument("--include-infra", action="store_true",
                        help="Include infra repos")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: stdout)")
    args = parser.parse_args()

    repos = args.repos or DEFAULT_REPOS
    if args.include_infra:
        repos = repos + INFRA_REPOS

    since = datetime.now(timezone.utc) - timedelta(days=args.days)

    print(f"Fetching PRs merged since {since.isoformat()} from {len(repos)} repos...",
          file=sys.stderr)

    all_prs: list[dict[str, Any]] = []
    for repo in repos:
        print(f"  Fetching {GITHUB_ORG}/{repo}...", file=sys.stderr)
        prs = fetch_merged_prs(repo, since)
        for pr in prs:
            if is_noise(pr):
                print(f"    [SKIP noise] #{pr['number']} {pr['title']}", file=sys.stderr)
                continue
            if is_pure_refactor(pr):
                print(f"    [SKIP refactor] #{pr['number']} {pr['title']}", file=sys.stderr)
                continue
            all_prs.append(build_pr_summary(pr))

    # Sort by merged_at descending
    all_prs.sort(key=lambda p: p.get("merged_at", ""), reverse=True)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": args.days,
        "since": since.isoformat(),
        "repos_scanned": repos,
        "total_prs": len(all_prs),
        "prs": all_prs,
    }

    output = json.dumps(result, indent=2, default=str)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Wrote {len(all_prs)} PRs to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
