"""Semantic analyzer — uses OpenAI to classify and summarize changes."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone

from openai import OpenAI

from .models import (
    AnalyzedChange,
    ChangeCategory,
    EnrichedChange,
    ImpactType,
    ReleaseNotes,
    extract_keywords,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a release notes analyst for HydraDB, a vector database product.
Your job is to analyze pull requests and produce clear, impact-driven release notes.

Think like you are explaining changes to leadership or customers:
"What changed and why should they care?"

For each PR, you must determine:
1. **What was done** — Be specific and concrete. Avoid vague statements.
2. **Why it matters** — What problem was solved or capability unlocked.
3. **Impact** — Classify as user-facing (new feature, better UX, performance) or
   organizational/system (reliability, scalability, developer velocity, cost efficiency).
4. **Category** — One of: new_feature, improvement, bug_fix, internal.
5. **Significance** — Is this worth including in release notes? Filter out:
   - Pure refactors with no user/system impact
   - Trivial config changes, typo fixes, formatting
   - CI/build pipeline tweaks with no impact
   - Dependency bumps with no behavioral change

PRIORITY: If the PR has a "Release Notes" section, use that content as the primary
source. It was written by the author specifically for release notes.

Writing guidelines:
- Be SPECIFIC: "Reduced query latency by ~35% for vector search" NOT "Improved performance"
- Be IMPACT-FIRST: Lead with what matters to users/org
- Be CONCISE: One clear sentence per field
- NO engineering jargon unless necessary
- NO duplication across entries
"""

ANALYSIS_PROMPT = """\
Analyze the following pull request and return a JSON object.

PR #{number} from {repo}:
- Title: {title}
- Author: {author}
- Merged: {merged_at}
- Labels: {labels}

{release_notes_block}

Description:
{body}

Commits:
{commits}

{slack_context_block}

Return a JSON object with these exact fields:
{{
  "what_was_done": "Specific description of the change",
  "why_it_matters": "Problem solved or capability unlocked",
  "impact_description": "Concrete impact statement",
  "impact_type": "user_facing" or "organizational",
  "category": "new_feature" or "improvement" or "bug_fix" or "internal",
  "is_significant": true or false
}}

If is_significant is false, still fill in the other fields but they can be brief.
Return ONLY valid JSON, no markdown fences or extra text.
"""


def _get_openai_client() -> OpenAI:
    """Create an OpenAI client from environment."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required for semantic analysis.")
    return OpenAI(api_key=api_key)


def analyze_changes(
    changes: list[EnrichedChange],
    days: int = 7,
    model: str = "gpt-4o-mini",
) -> ReleaseNotes:
    """Analyze a list of enriched changes and produce categorized release notes.

    Args:
        changes: List of PRs enriched with optional Slack context.
        days: Lookback window (used for date range in output).
        model: OpenAI model to use for analysis.

    Returns:
        ReleaseNotes with changes categorized into features, improvements, fixes, internal.
    """
    if not changes:
        now = datetime.now(timezone.utc)
        return ReleaseNotes(
            date_range_start=now - timedelta(days=days),
            date_range_end=now,
        )

    client = _get_openai_client()
    analyzed: list[AnalyzedChange] = []

    for change in changes:
        try:
            result = _analyze_single_change(client, change, model)
            if result:
                analyzed.append(result)
        except Exception:
            logger.warning(
                "Failed to analyze PR #%d from %s — skipping",
                change.pr.number,
                change.pr.repo_name,
                exc_info=True,
            )

    # Deduplicate related changes (e.g., a feature PR + its follow-up fix)
    analyzed = _deduplicate(analyzed)

    # Categorize into buckets
    now = datetime.now(timezone.utc)
    notes = ReleaseNotes(
        date_range_start=now - timedelta(days=days),
        date_range_end=now,
        raw_prs=[c.pr for c in changes],
    )

    for change in analyzed:
        if not change.is_significant:
            continue
        if change.category == ChangeCategory.NEW_FEATURE:
            notes.features.append(change)
        elif change.category == ChangeCategory.IMPROVEMENT:
            notes.improvements.append(change)
        elif change.category == ChangeCategory.BUG_FIX:
            notes.fixes.append(change)
        elif change.category == ChangeCategory.INTERNAL:
            notes.internal_changes.append(change)

    logger.info(
        "Analysis complete: %d significant changes from %d PRs (%d features, %d improvements, %d fixes, %d internal)",
        notes.significant_count,
        len(changes),
        len(notes.features),
        len(notes.improvements),
        len(notes.fixes),
        len(notes.internal_changes),
    )
    return notes


def _analyze_single_change(
    client: OpenAI,
    change: EnrichedChange,
    model: str,
) -> AnalyzedChange | None:
    """Analyze a single PR using OpenAI."""
    pr = change.pr

    # Build the release notes block (highest priority source)
    release_notes_block = ""
    if pr.release_notes_section:
        release_notes_block = f"RELEASE NOTES (author-provided, highest priority):\n{pr.release_notes_section}"
    elif pr.summary_section or pr.why_it_matters_section:
        parts = []
        if pr.summary_section:
            parts.append(f"Summary: {pr.summary_section}")
        if pr.why_it_matters_section:
            parts.append(f"Why it matters: {pr.why_it_matters_section}")
        release_notes_block = "\n".join(parts)

    # Build Slack context block
    slack_context_block = ""
    if change.slack_context.messages:
        slack_texts = [f"- {msg.text[:300]}" for msg in change.slack_context.messages[:5]]
        slack_context_block = "Related Slack discussions:\n" + "\n".join(slack_texts)

    # Format commits
    commits_text = "\n".join(f"- {c}" for c in pr.commits[:10]) if pr.commits else "(no commits)"

    prompt = ANALYSIS_PROMPT.format(
        number=pr.number,
        repo=pr.repo_name,
        title=pr.title,
        author=pr.author,
        merged_at=pr.merged_at.strftime("%Y-%m-%d"),
        labels=", ".join(pr.labels) if pr.labels else "(none)",
        release_notes_block=release_notes_block,
        body=pr.body[:2000] if pr.body else "(no description)",
        commits=commits_text,
        slack_context_block=slack_context_block,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=500,
    )

    content = response.choices[0].message.content
    if not content:
        logger.warning("Empty response from OpenAI for PR #%d", pr.number)
        return None

    content = _strip_markdown_fences(content)

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Failed to parse OpenAI response for PR #%d: %s", pr.number, content[:200])
        return None

    return AnalyzedChange(
        what_was_done=data.get("what_was_done", pr.title),
        why_it_matters=data.get("why_it_matters", ""),
        impact_description=data.get("impact_description", ""),
        impact_type=ImpactType(data.get("impact_type", "organizational")),
        category=ChangeCategory(data.get("category", "internal")),
        pr_url=pr.url,
        pr_number=pr.number,
        repo_name=pr.repo_name,
        is_significant=data.get("is_significant", True),
    )


def _strip_markdown_fences(content: str) -> str:
    """Remove markdown code fences wrapping a string (e.g. ```json ... ```)."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


def _deduplicate(changes: list[AnalyzedChange]) -> list[AnalyzedChange]:
    """Remove near-duplicate entries (e.g., a feature + its immediate hotfix).

    Groups by similar what_was_done text and keeps the most significant entry.
    Simple heuristic: if two changes from the same repo share 3+ keywords
    in what_was_done, keep only the one with the higher-priority category.
    """
    if len(changes) <= 1:
        return changes

    category_priority = {
        ChangeCategory.NEW_FEATURE: 0,
        ChangeCategory.BUG_FIX: 1,
        ChangeCategory.IMPROVEMENT: 2,
        ChangeCategory.INTERNAL: 3,
    }

    # Extract keywords for each change
    change_keywords: list[set[str]] = [extract_keywords(c.what_was_done) for c in changes]

    # Mark duplicates
    keep = [True] * len(changes)
    for i in range(len(changes)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(changes)):
            if not keep[j]:
                continue
            if changes[i].repo_name != changes[j].repo_name:
                continue
            overlap = change_keywords[i] & change_keywords[j]
            if len(overlap) >= 3:
                # Keep the higher-priority one
                pri_i = category_priority.get(changes[i].category, 99)
                pri_j = category_priority.get(changes[j].category, 99)
                if pri_i <= pri_j:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    result = [c for c, k in zip(changes, keep, strict=True) if k]
    if len(result) < len(changes):
        logger.info("Deduplicated %d → %d changes", len(changes), len(result))
    return result
