#!/usr/bin/env python3
"""
Main orchestrator for the HydraDB Release Notes Agent.

This script is called by the Vorflux scheduled session. It:
1. Fetches merged PRs from all product repos
2. Outputs the structured PR data as JSON
3. The Vorflux agent (Claude) then analyzes this data and generates release notes
4. The agent calls post_to_slack.py to deliver the notes

Usage:
    python generate.py                  # default: last 7 days
    python generate.py --days 14        # custom lookback
    python generate.py --include-infra  # include infra repos
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

# Ensure imports work regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch_prs import fetch_merged_prs, is_noise, is_pure_refactor, build_pr_summary
from config import DEFAULT_LOOKBACK_DAYS, DEFAULT_REPOS, INFRA_REPOS, GITHUB_ORG


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate release notes data")
    parser.add_argument("--days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--include-infra", action="store_true")
    args = parser.parse_args()

    repos = DEFAULT_REPOS[:]
    if args.include_infra:
        repos += INFRA_REPOS

    since = datetime.now(timezone.utc) - timedelta(days=args.days)

    print(f"=== HydraDB Release Notes Agent ===", file=sys.stderr)
    print(f"Lookback: {args.days} days (since {since.date().isoformat()})", file=sys.stderr)
    print(f"Repos: {len(repos)}", file=sys.stderr)
    print(file=sys.stderr)

    all_prs = []
    repo_stats = {}
    for repo in repos:
        prs = fetch_merged_prs(repo, since)
        included = []
        skipped_noise = 0
        skipped_refactor = 0
        for pr in prs:
            if is_noise(pr):
                skipped_noise += 1
                continue
            if is_pure_refactor(pr):
                skipped_refactor += 1
                continue
            included.append(build_pr_summary(pr))

        repo_stats[repo] = {
            "total_merged": len(prs),
            "included": len(included),
            "skipped_noise": skipped_noise,
            "skipped_refactor": skipped_refactor,
        }
        all_prs.extend(included)

    all_prs.sort(key=lambda p: p.get("merged_at", ""), reverse=True)

    # Deduplicate cross-repo PRs that represent the same logical change.
    # Group by normalized title; keep the one from the higher-priority repo
    # (core > product) or the one with the richer body.
    seen_titles: dict[str, int] = {}  # normalized_title -> index in deduped list
    deduped: list[dict[str, Any]] = []
    for pr in all_prs:
        # Normalize: lowercase, strip prefix (feat:, fix:, etc.), strip whitespace
        raw_title = pr["title"].lower().strip()
        # Remove conventional-commit prefix
        normalized = re.sub(r"^(feat|fix|chore|perf|refactor|docs|build|ci|test)[:/]\s*", "", raw_title).strip()
        if normalized in seen_titles:
            # Merge: add the other repo as a cross-reference
            idx = seen_titles[normalized]
            existing = deduped[idx]
            other_repos = existing.get("also_in_repos", [])
            other_repos.append({"repo": pr["repo"], "number": pr["number"], "url": pr["url"]})
            existing["also_in_repos"] = other_repos
        else:
            seen_titles[normalized] = len(deduped)
            pr["also_in_repos"] = []
            deduped.append(pr)

    all_prs = deduped

    total_before_dedup = sum(s["included"] for s in repo_stats.values())
    total_after_dedup = len(all_prs)

    # Print stats to stderr
    print(f"--- Scan Results ---", file=sys.stderr)
    for repo, stats in repo_stats.items():
        if stats["total_merged"] > 0:
            print(f"  {repo}: {stats['included']}/{stats['total_merged']} PRs included "
                  f"({stats['skipped_noise']} noise, {stats['skipped_refactor']} refactors skipped)",
                  file=sys.stderr)
    if total_before_dedup != total_after_dedup:
        print(f"\nDeduplicated: {total_before_dedup} -> {total_after_dedup} "
              f"({total_before_dedup - total_after_dedup} cross-repo duplicates merged)",
              file=sys.stderr)
    print(f"Total PRs for release notes: {total_after_dedup}", file=sys.stderr)

    # Output the structured data as JSON to stdout
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "week_ending": datetime.now(timezone.utc).date().isoformat(),
        "lookback_days": args.days,
        "since": since.isoformat(),
        "repos_scanned": repos,
        "repo_stats": repo_stats,
        "total_prs_before_dedup": total_before_dedup,
        "total_prs": len(all_prs),
        "prs": all_prs,
    }

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
