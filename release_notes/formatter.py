"""Markdown formatter — renders ReleaseNotes into the final output document."""

from __future__ import annotations

import logging
from pathlib import Path

from .models import AnalyzedChange, ImpactType, ReleaseNotes

logger = logging.getLogger(__name__)


def format_release_notes(notes: ReleaseNotes) -> str:
    """Render ReleaseNotes to a markdown string following the strict output format.

    Sections with no entries are omitted. User-facing changes are highlighted.
    Each entry includes a PR link reference.
    """
    start = notes.date_range_start.strftime("%B %d, %Y")
    end = notes.date_range_end.strftime("%B %d, %Y")
    lines: list[str] = [f"# HydraDB Weekly Release Notes — {start} to {end}", ""]

    has_content = False

    # New Features
    if notes.features:
        has_content = True
        lines.append("## New Features")
        for change in notes.features:
            lines.append(_format_entry(change))
        lines.append("")

    # Improvements
    if notes.improvements:
        has_content = True
        lines.append("## Improvements")
        for change in notes.improvements:
            lines.append(_format_entry(change))
        lines.append("")

    # Fixes
    if notes.fixes:
        has_content = True
        lines.append("## Fixes")
        for change in notes.fixes:
            lines.append(_format_entry(change))
        lines.append("")

    # Internal Changes (optional)
    if notes.internal_changes:
        has_content = True
        lines.append("## Internal Changes")
        for change in notes.internal_changes:
            lines.append(_format_internal_entry(change))
        lines.append("")

    if not has_content:
        lines.append("_No significant changes this week._")
        lines.append("")

    # Footer with PR count
    total_prs = len(notes.raw_prs)
    significant = len(notes.features) + len(notes.improvements) + len(notes.fixes) + len(notes.internal_changes)
    lines.append("---")
    lines.append(f"_Generated from {total_prs} merged PRs — {significant} significant changes highlighted._")
    lines.append("")

    return "\n".join(lines)


def _format_entry(change: AnalyzedChange) -> str:
    """Format a single release notes entry with impact highlighting.

    Format: - [User-Facing]? <What was done> -- <Why it matters> -- <Impact> (PR #N)
    """
    parts: list[str] = []

    # User-facing highlight
    if change.impact_type == ImpactType.USER_FACING:
        parts.append("**[User-Facing]**")

    parts.append(change.what_was_done)

    if change.why_it_matters:
        parts.append(f"— {change.why_it_matters}")

    if change.impact_description:
        parts.append(f"— {change.impact_description}")

    entry = " ".join(parts)

    # PR reference link
    pr_ref = f"([PR #{change.pr_number}]({change.pr_url}))"

    return f"- {entry} {pr_ref}"


def _format_internal_entry(change: AnalyzedChange) -> str:
    """Format an internal change entry (simpler format, no impact type highlight).

    Format: - <What was done> -- <Why it matters> (PR #N)
    """
    parts = [change.what_was_done]

    if change.why_it_matters:
        parts.append(f"— {change.why_it_matters}")

    entry = " ".join(parts)
    pr_ref = f"([PR #{change.pr_number}]({change.pr_url}))"

    return f"- {entry} {pr_ref}"


def save_release_notes(
    notes: ReleaseNotes,
    output_dir: str = "reports",
) -> Path:
    """Format and save release notes to a markdown file.

    Args:
        notes: The release notes to save.
        output_dir: Directory to save the file in.

    Returns:
        Path to the saved file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    date_str = notes.date_range_end.strftime("%Y-%m-%d")
    filename = f"release-notes-{date_str}.md"
    filepath = output_path / filename

    content = format_release_notes(notes)
    filepath.write_text(content, encoding="utf-8")

    logger.info("Release notes saved to %s", filepath)
    return filepath
