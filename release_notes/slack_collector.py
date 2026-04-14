"""Slack context collector — fetches and matches Slack discussions to PRs."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone

import httpx

from .models import MergedPR, SlackContext, SlackMessage, extract_keywords

logger = logging.getLogger(__name__)

SLACK_API_BASE = "https://slack.com/api"

DEFAULT_CHANNELS = ["engineering", "github-hydradb"]


def _get_slack_token() -> str | None:
    """Read SLACK_BOT_TOKEN from environment. Returns None if not set."""
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        logger.info("SLACK_BOT_TOKEN not set — skipping Slack context enrichment")
        return None
    return token


async def fetch_slack_messages(
    channels: list[str] | None = None,
    days: int = 7,
    slack_token: str | None = None,
) -> list[SlackMessage]:
    """Fetch messages from Slack channels for the given time window.

    Returns an empty list if SLACK_BOT_TOKEN is not configured.
    """
    token = slack_token or _get_slack_token()
    if not token:
        return []

    channels = channels or DEFAULT_CHANNELS
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    oldest_ts = str(cutoff.timestamp())

    all_messages: list[SlackMessage] = []

    async with httpx.AsyncClient(
        base_url=SLACK_API_BASE,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30.0,
    ) as client:
        # First, resolve channel names to IDs
        channel_map = await _resolve_channel_ids(client, channels)

        for channel_name, channel_id in channel_map.items():
            try:
                messages = await _fetch_channel_history(client, channel_id, channel_name, oldest_ts)
                all_messages.extend(messages)
            except Exception:
                logger.warning(
                    "Failed to fetch history for #%s — skipping channel",
                    channel_name,
                    exc_info=True,
                )

    logger.info("Fetched %d Slack messages from %d channels", len(all_messages), len(channel_map))
    return all_messages


async def _resolve_channel_ids(
    client: httpx.AsyncClient,
    channel_names: list[str],
) -> dict[str, str]:
    """Map channel names to Slack channel IDs."""
    channel_map: dict[str, str] = {}
    cursor = ""

    try:
        while True:
            params: dict[str, str | int] = {
                "types": "public_channel,private_channel",
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor

            response = await client.get("/conversations.list", params=params)
            data = response.json()

            if not data.get("ok"):
                logger.warning("Slack conversations.list failed: %s", data.get("error", "unknown"))
                break

            for ch in data.get("channels", []):
                name = ch.get("name", "")
                if name in channel_names:
                    channel_map[name] = ch["id"]

            # Stop if we found all channels or no more pages
            if len(channel_map) == len(channel_names):
                break
            cursor = data.get("response_metadata", {}).get("next_cursor", "")
            if not cursor:
                break
    except Exception:
        logger.warning("Error resolving Slack channel IDs", exc_info=True)

    return channel_map


async def _fetch_channel_history(
    client: httpx.AsyncClient,
    channel_id: str,
    channel_name: str,
    oldest_ts: str,
) -> list[SlackMessage]:
    """Fetch message history for a single channel."""
    messages: list[SlackMessage] = []
    cursor = ""

    while True:
        params: dict[str, str | int] = {
            "channel": channel_id,
            "oldest": oldest_ts,
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor

        response = await client.get("/conversations.history", params=params)
        data = response.json()

        if not data.get("ok"):
            logger.warning(
                "Slack conversations.history failed for #%s: %s",
                channel_name,
                data.get("error", "unknown"),
            )
            break

        for msg in data.get("messages", []):
            text = msg.get("text", "")
            if not text:
                continue

            # Fetch thread replies if this message has a thread
            thread_replies: list[str] = []
            if msg.get("reply_count", 0) > 0:
                thread_replies = await _fetch_thread_replies(client, channel_id, msg["ts"])

            messages.append(
                SlackMessage(
                    text=text,
                    timestamp=msg.get("ts", ""),
                    user=msg.get("user", ""),
                    thread_replies=thread_replies,
                    channel=channel_name,
                )
            )

        cursor = data.get("response_metadata", {}).get("next_cursor", "")
        if not cursor:
            break

    return messages


async def _fetch_thread_replies(
    client: httpx.AsyncClient,
    channel_id: str,
    thread_ts: str,
) -> list[str]:
    """Fetch reply texts for a thread."""
    try:
        response = await client.get(
            "/conversations.replies",
            params={"channel": channel_id, "ts": thread_ts, "limit": 100},
        )
        data = response.json()
        if not data.get("ok"):
            return []
        # Skip the first message (it's the parent)
        return [msg.get("text", "") for msg in data.get("messages", [])[1:] if msg.get("text")]
    except Exception:
        logger.warning("Failed to fetch thread replies for ts=%s", thread_ts)
        return []


def match_slack_context(
    pr: MergedPR,
    messages: list[SlackMessage],
    repo_name: str | None = None,
) -> SlackContext:
    """Match Slack messages to a PR using URL, branch, and keyword matching.

    Args:
        pr: The pull request to match against.
        messages: Candidate Slack messages.
        repo_name: When provided, PR-number matches require the repo short
            name (e.g. ``"cortex-application"``) to also appear in the message
            text to avoid cross-repo false positives.

    Matching strategies (in priority order):
    1. PR URL appears in message text
    2. PR number reference with repo-name check (when *repo_name* is given)
    3. Significant keyword overlap between PR title/body and message
    """
    if not messages:
        return SlackContext()

    matched_messages: list[SlackMessage] = []
    matched_keywords: set[str] = set()

    # Derive the short repo name (e.g. "cortex-application") for PR-number matching.
    # Always split on "/" so that a full "owner/repo" string passed via repo_name
    # is reduced to just the repo part, matching what appears in Slack messages.
    repo_short = (repo_name or pr.repo_name).split("/")[-1]

    # Extract keywords from PR title (words 4+ chars, lowercased)
    pr_keywords = extract_keywords(pr.title + " " + pr.summary_section)

    for msg in messages:
        all_text = msg.text + " " + " ".join(msg.thread_replies)

        # Strategy 1: PR URL match
        if pr.url and pr.url in all_text:
            matched_messages.append(msg)
            matched_keywords.add("url_match")
            continue

        # Strategy 2: PR number match (e.g., #123 or PR-123)
        # Require the repo short name to also appear in the message to
        # prevent cross-repo false positives.
        pr_ref_pattern = rf"(?:#|PR[- ]?){pr.number}\b"
        if re.search(pr_ref_pattern, all_text, re.IGNORECASE) and repo_short.lower() in all_text.lower():
            matched_messages.append(msg)
            matched_keywords.add(f"pr_ref_#{pr.number}")
            continue

        # Strategy 3: Keyword overlap (need at least 2 matching keywords)
        msg_keywords = extract_keywords(all_text)
        overlap = pr_keywords & msg_keywords
        if len(overlap) >= 2:
            matched_messages.append(msg)
            matched_keywords.update(overlap)

    return SlackContext(
        messages=matched_messages,
        matched_keywords=sorted(matched_keywords),
    )
