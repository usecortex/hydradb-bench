"""GitHub data collector — fetches merged PRs from target repositories."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime, timedelta, timezone

import httpx

from .models import MergedPR

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"

DEFAULT_REPOS = [
    "usecortex/cortex-application",
    "usecortex/cortex-ingestion",
]


def _get_github_token() -> str:
    """Read GITHUB_TOKEN from environment."""
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN environment variable is required. "
            "Set it to a GitHub personal access token with repo read access."
        )
    return token


def _parse_section(body: str | None, heading: str) -> str:
    """Extract content under a markdown ## heading from a PR body.

    Returns the text between the given heading and the next heading (or end of string).
    """
    if not body:
        return ""
    # Match ## heading (case-insensitive), capture until next ## or end
    pattern = rf"##\s*{re.escape(heading)}\s*\n(.*?)(?=\n##\s|\Z)"
    match = re.search(pattern, body, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


async def _handle_rate_limit(response: httpx.Response) -> None:
    """Sleep until the rate-limit reset time if we've been throttled."""
    if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
        reset_ts = int(response.headers.get("X-RateLimit-Reset", "0"))
        wait = max(reset_ts - int(datetime.now(timezone.utc).timestamp()), 1)
        logger.warning("GitHub rate limit hit — sleeping %d seconds", wait)
        await asyncio.sleep(wait)


async def fetch_merged_prs(
    repos: list[str] | None = None,
    days: int = 7,
    github_token: str | None = None,
) -> list[MergedPR]:
    """Fetch PRs merged into main in the last `days` days from the given repos.

    Args:
        repos: List of "owner/repo" strings. Defaults to DEFAULT_REPOS.
        days: Lookback window in days.
        github_token: GitHub token. Reads from GITHUB_TOKEN env var if not provided.

    Returns:
        List of MergedPR objects, sorted by merged_at descending.
    """
    repos = repos or DEFAULT_REPOS
    token = github_token or _get_github_token()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    async with httpx.AsyncClient(
        base_url=GITHUB_API_BASE,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=30.0,
    ) as client:

        async def _fetch_one(repo: str) -> list[MergedPR]:
            try:
                return await _fetch_repo_prs(client, repo, cutoff)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 403:
                    await _handle_rate_limit(exc.response)
                    # Retry once after rate-limit sleep
                    return await _fetch_repo_prs(client, repo, cutoff)
                else:
                    logger.error("Failed to fetch PRs from %s: %s", repo, exc)
                    raise
            except httpx.HTTPError as exc:
                logger.error("Network error fetching PRs from %s: %s", repo, exc)
                raise

        results = await asyncio.gather(*[_fetch_one(repo) for repo in repos])

    all_prs: list[MergedPR] = [pr for repo_prs in results for pr in repo_prs]

    # Sort by merged_at descending (most recent first)
    all_prs.sort(key=lambda pr: pr.merged_at, reverse=True)
    return all_prs


async def _fetch_repo_prs(
    client: httpx.AsyncClient,
    repo: str,
    cutoff: datetime,
) -> list[MergedPR]:
    """Fetch merged PRs for a single repo since cutoff date.

    Uses the GitHub search API with ``merged:>YYYY-MM-DD`` so the server
    filters by merge date directly, avoiding issues with ``sort=updated``
    where recently-commented old PRs can appear alongside newer ones.
    """
    prs: list[MergedPR] = []
    page = 1
    per_page = 100
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    query = f"repo:{repo} is:pr is:merged base:main merged:>{cutoff_str}"

    while True:
        response = await client.get(
            "/search/issues",
            params={
                "q": query,
                "sort": "updated",
                "order": "desc",
                "per_page": per_page,
                "page": page,
            },
        )
        if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
            await _handle_rate_limit(response)
            continue  # Retry the same page
        response.raise_for_status()
        result = response.json()
        items = result.get("items", [])

        if not items:
            break

        for item in items:
            # The search API returns issue-shaped objects; fetch the full PR
            # to get merged_at and draft status.
            pr_detail = await _fetch_pr_detail(client, repo, item["number"])
            if pr_detail is None:
                continue

            # Skip drafts and WIP
            if pr_detail.get("draft", False):
                continue
            title = pr_detail.get("title", "")
            if title.upper().startswith("WIP") or title.upper().startswith("[WIP]"):
                continue

            merged_at_str = pr_detail.get("merged_at")
            if not merged_at_str:
                continue
            merged_at = datetime.fromisoformat(merged_at_str.replace("Z", "+00:00"))

            # Fetch commits for this PR
            commits = await _fetch_pr_commits(client, repo, pr_detail["number"])

            body = pr_detail.get("body") or ""
            labels = [label["name"] for label in pr_detail.get("labels", [])]

            pr = MergedPR(
                number=pr_detail["number"],
                title=title,
                url=pr_detail.get("html_url", ""),
                author=pr_detail.get("user", {}).get("login", "unknown"),
                body=body,
                release_notes_section=_parse_section(body, "Release Notes"),
                summary_section=_parse_section(body, "Summary"),
                why_it_matters_section=_parse_section(body, "Why it matters"),
                commits=commits,
                labels=labels,
                merged_at=merged_at,
                repo_name=repo,
            )
            prs.append(pr)
            logger.debug("Found PR #%d: %s (%s)", pr.number, pr.title, repo)

        if len(items) < per_page:
            break

        page += 1

    logger.info("Fetched %d merged PRs from %s", len(prs), repo)
    return prs


async def _fetch_pr_detail(
    client: httpx.AsyncClient,
    repo: str,
    pr_number: int,
) -> dict | None:
    """Fetch full PR detail (needed for merged_at, draft, body, labels)."""
    try:
        response = await client.get(f"/repos/{repo}/pulls/{pr_number}")
        response.raise_for_status()
        return response.json()
    except Exception:
        logger.warning("Failed to fetch PR #%d detail from %s", pr_number, repo)
        return None


async def _fetch_pr_commits(
    client: httpx.AsyncClient,
    repo: str,
    pr_number: int,
) -> list[str]:
    """Fetch commit messages for a PR."""
    try:
        response = await client.get(
            f"/repos/{repo}/pulls/{pr_number}/commits",
            params={"per_page": 100},
        )
        response.raise_for_status()
        commits_data = response.json()
        return [
            c.get("commit", {}).get("message", "").split("\n")[0]
            for c in commits_data
            if c.get("commit", {}).get("message")
        ]
    except Exception:
        logger.warning("Failed to fetch commits for PR #%d in %s", pr_number, repo)
        return []
