#!/usr/bin/env python3
"""
Post release notes to Slack via webhook.

Converts standard markdown to Slack mrkdwn format before posting.

Usage:
    python post_to_slack.py --file release_notes.md
    python post_to_slack.py --text "Release notes content here"
    echo "notes" | python post_to_slack.py --stdin

Requires SLACK_WEBHOOK_URL environment variable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
import urllib.error


def markdown_to_slack_mrkdwn(text: str) -> str:
    """Convert standard markdown to Slack mrkdwn format.

    Key differences:
    - ## Headers -> *Bold text* (Slack has no header syntax in mrkdwn)
    - **bold** -> *bold* (Slack uses single asterisks)
    - `code` stays the same
    - - bullets stay the same
    - Links [text](url) -> <url|text>
    """
    lines = text.split("\n")
    converted = []

    for line in lines:
        # Convert ## headers to bold text with emoji preserved
        if line.startswith("## "):
            header_text = line[3:].strip()
            converted.append(f"\n*{header_text}*")
            continue

        # Convert ### sub-headers to bold
        if line.startswith("### "):
            header_text = line[4:].strip()
            converted.append(f"*{header_text}*")
            continue

        # Convert **bold** to *bold* (but not inside code blocks)
        if "```" not in line:
            line = re.sub(r"\*\*(.+?)\*\*", r"*\1*", line)

        # Convert markdown links [text](url) to Slack links <url|text>
        line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", line)

        converted.append(line)

    return "\n".join(converted)


def post_to_slack(text: str, webhook_url: str) -> bool:
    """Post a message to Slack via incoming webhook.

    Slack webhooks have a 40,000 character limit per message.
    If the text exceeds that, it's split into multiple messages.
    """
    MAX_LEN = 39_000  # leave buffer for JSON encoding overhead

    # Convert markdown to Slack mrkdwn
    text = markdown_to_slack_mrkdwn(text)

    chunks = []
    if len(text) <= MAX_LEN:
        chunks = [text]
    else:
        # Split at section boundaries (*Header* lines)
        sections = re.split(r"(\n\*[^\n]+\*\n)", text)
        current_chunk = ""
        for section in sections:
            if len(current_chunk) + len(section) > MAX_LEN:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = section
            else:
                current_chunk += section
        if current_chunk:
            chunks.append(current_chunk)

    success = True
    for i, chunk in enumerate(chunks):
        payload = json.dumps({"text": chunk}).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status != 200:
                    print(f"[ERROR] Slack returned status {resp.status}", file=sys.stderr)
                    success = False
                else:
                    part = f" (part {i+1}/{len(chunks)})" if len(chunks) > 1 else ""
                    print(f"[OK] Posted to Slack{part}", file=sys.stderr)
        except urllib.error.HTTPError as e:
            print(f"[ERROR] Slack HTTP error: {e.code} {e.reason}", file=sys.stderr)
            success = False
        except urllib.error.URLError as e:
            print(f"[ERROR] Slack URL error: {e.reason}", file=sys.stderr)
            success = False

    return success


def main() -> None:
    parser = argparse.ArgumentParser(description="Post release notes to Slack")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to markdown file")
    group.add_argument("--text", type=str, help="Release notes text")
    group.add_argument("--stdin", action="store_true", help="Read from stdin")
    args = parser.parse_args()

    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("[ERROR] SLACK_WEBHOOK_URL environment variable not set", file=sys.stderr)
        sys.exit(1)

    if args.file:
        with open(args.file) as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    if not text.strip():
        print("[WARN] Empty release notes, nothing to post", file=sys.stderr)
        sys.exit(0)

    ok = post_to_slack(text, webhook_url)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
