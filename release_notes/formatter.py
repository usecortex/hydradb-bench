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
    has_content |= _render_section(lines, "## New Features", notes.features)
    has_content |= _render_section(lines, "## Improvements", notes.improvements)
    has_content |= _render_section(lines, "## Fixes", notes.fixes)
    has_content |= _render_section(lines, "## Internal Changes", notes.internal_changes, include_impact=False)

    if not has_content:
        lines.append("_No significant changes this week._")
        lines.append("")

    # Footer with PR count
    total_prs = len(notes.raw_prs)
    lines.append("---")
    lines.append(
        f"_Generated from {total_prs} merged PRs — {notes.significant_count} significant changes highlighted._"
    )
    lines.append("")

    return "\n".join(lines)


def _render_section(
    lines: list[str],
    heading: str,
    changes: list[AnalyzedChange],
    include_impact: bool = True,
) -> bool:
    """Append a markdown section for *changes* if the list is non-empty.

    Returns ``True`` when at least one entry was rendered.
    """
    if not changes:
        return False
    lines.append(heading)
    for change in changes:
        lines.append(_format_entry(change, include_impact=include_impact))
    lines.append("")
    return True


def _format_entry(change: AnalyzedChange, *, include_impact: bool = True) -> str:
    """Format a single release notes entry.

    Args:
        change: The analyzed change to format.
        include_impact: When ``True``, prepend a **[User-Facing]** highlight for
            user-facing changes and append the ``impact_description``.  Set to
            ``False`` for internal-change entries where those details are omitted.
    """
    parts: list[str] = []

    if include_impact and change.impact_type == ImpactType.USER_FACING:
        parts.append("**[User-Facing]**")

    parts.append(change.what_was_done)

    if change.why_it_matters:
        parts.append(f"— {change.why_it_matters}")

    if include_impact and change.impact_description:
        parts.append(f"— {change.impact_description}")

    entry = " ".join(parts)

    # PR reference link
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
