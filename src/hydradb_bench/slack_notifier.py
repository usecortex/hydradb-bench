"""Send benchmark report files to a Slack DM when a run completes."""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

from .models import BenchmarkResult, SlackConfig

logger = logging.getLogger(__name__)

_SLACK_POST_URL = "https://slack.com/api/chat.postMessage"
_SLACK_CONV_OPEN_URL = "https://slack.com/api/conversations.open"
_SLACK_UPLOAD_URL = "https://slack.com/api/files.getUploadURLExternal"
_SLACK_COMPLETE_URL = "https://slack.com/api/files.completeUploadExternal"
_SLACK_FILES_INFO_URL = "https://slack.com/api/files.info"


async def _open_dm_channel(client: httpx.AsyncClient, token: str, user_id: str) -> str:
    """Open a DM with the given user — returns the DM channel ID."""
    r = await client.post(
        _SLACK_CONV_OPEN_URL,
        headers={"Authorization": f"Bearer {token}"},
        json={"users": user_id},
    )
    r.raise_for_status()
    body = r.json()
    if body.get("ok"):
        return body["channel"]["id"]
    raise RuntimeError(f"conversations.open failed: {body.get('error')}")


async def _upload_and_share(
    token: str,
    channel: str,
    file_path: Path,
) -> bool:
    """
    Upload a file via v2 API, then share it to the DM channel as a visible message.
    Steps:
      1. getUploadURLExternal  — get upload URL + file_id
      2. PUT content to upload URL
      3. completeUploadExternal — finalize (no channel yet)
      4. files.info — get permalink
      5. chat.postMessage — post permalink as a button so it appears in chat
    """
    content = file_path.read_bytes()
    filename = file_path.name

    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        headers = {"Authorization": f"Bearer {token}"}

        # Step 1 — request upload URL
        r = await client.post(
            _SLACK_UPLOAD_URL,
            headers=headers,
            data={"filename": filename, "length": len(content)},
        )
        r.raise_for_status()
        body = r.json()
        if not body.get("ok"):
            logger.error("getUploadURLExternal failed for %s: %s", filename, body.get("error"))
            return False
        upload_url: str = body["upload_url"]
        file_id: str = body["file_id"]

        # Step 2 — PUT file bytes
        put_r = await client.put(upload_url, content=content)
        put_r.raise_for_status()

        # Step 3 — complete upload (share to channel with initial_comment forces it into chat)
        complete_r = await client.post(
            _SLACK_COMPLETE_URL,
            headers=headers,
            json={
                "files": [{"id": file_id, "title": filename}],
                "channel_id": channel,
                "initial_comment": f"📎 `{filename}`",
            },
        )
        complete_r.raise_for_status()
        complete_body = complete_r.json()
        if not complete_body.get("ok"):
            logger.error("completeUploadExternal failed for %s: %s", filename, complete_body.get("error"))
            return False

        # Step 4 — get file info for permalink
        info_r = await client.get(
            _SLACK_FILES_INFO_URL,
            headers=headers,
            params={"file": file_id},
        )
        info_r.raise_for_status()
        info_body = info_r.json()
        if not info_body.get("ok"):
            # File uploaded but can't get permalink — still counts as success
            logger.warning("files.info failed for %s: %s", filename, info_body.get("error"))
            return True

        permalink = info_body.get("file", {}).get("permalink", "")

        # Step 5 — post a button message with the file link so it's visible in chat
        if permalink:
            msg_r = await client.post(
                _SLACK_POST_URL,
                headers=headers,
                json={
                    "channel": channel,
                    "blocks": [
                        {
                            "type": "section",
                            "text": {"type": "mrkdwn", "text": f"*{filename}*"},
                            "accessory": {
                                "type": "button",
                                "text": {"type": "plain_text", "text": "Download"},
                                "url": permalink,
                                "action_id": f"download_{file_id}",
                            },
                        }
                    ],
                },
            )
            msg_r.raise_for_status()

    return True


def _build_summary_message(result: BenchmarkResult) -> str:
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

    lines.append(
        f"*Samples:* {len(result.samples)} queried, {result.error_count} errors"
    )

    if result.latency_stats:
        p50 = result.latency_stats.get("p50", 0)
        p95 = result.latency_stats.get("p95", 0)
        lines.append(f"*Latency (HydraDB):* p50={p50:.0f}ms  p95={p95:.0f}ms")

    if result.token_usage.total_tokens > 0:
        lines.append(
            f"*Judge LLM Cost:* ${result.token_usage.actual_cost_usd:.4f} "
            f"({result.token_usage.total_tokens:,} tokens)"
        )

    lines.append("\n_Reports attached below._")
    return "\n".join(lines)


async def send_results(
    slack_config: SlackConfig,
    result: BenchmarkResult,
    report_paths: list[str],
) -> None:
    """Post a summary message and upload all report files to the configured Slack DM."""
    if not slack_config.enabled:
        return

    token = slack_config.bot_token
    user_id = slack_config.user_id

    if not token or not user_id:
        logger.warning(
            "Slack notifications enabled but bot_token or user_id is empty — skipping."
        )
        return

    async with httpx.AsyncClient(timeout=60) as client:
        channel = await _open_dm_channel(client, token, user_id)

        # Post summary text message
        summary = _build_summary_message(result)
        post_r = await client.post(
            _SLACK_POST_URL,
            headers={"Authorization": f"Bearer {token}"},
            json={"channel": channel, "text": summary},
        )
        post_r.raise_for_status()
        post_body = post_r.json()
        if not post_body.get("ok"):
            logger.error("Slack chat.postMessage failed: %s", post_body.get("error"))

    # Upload and share each report file
    for path_str in report_paths:
        path = Path(path_str)
        if not path.exists():
            logger.warning("Report file not found, skipping: %s", path)
            continue
        try:
            ok = await _upload_and_share(token, channel, path)
            if ok:
                logger.info("Uploaded to Slack: %s", path.name)
            else:
                logger.warning("Slack upload returned not-ok for: %s", path.name)
        except Exception as e:
            logger.error("Failed to upload %s to Slack: %s", path.name, e)
