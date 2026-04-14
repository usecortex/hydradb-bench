"""Data models for the release notes agent."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ChangeCategory(str, Enum):
    """Category of a change for release notes grouping."""

    NEW_FEATURE = "new_feature"
    IMPROVEMENT = "improvement"
    BUG_FIX = "bug_fix"
    INTERNAL = "internal"


class ImpactType(str, Enum):
    """Whether a change is user-facing or organizational/system-level."""

    USER_FACING = "user_facing"
    ORGANIZATIONAL = "organizational"


class MergedPR(BaseModel):
    """A pull request merged into main, with parsed sections."""

    number: int
    title: str
    url: str
    author: str
    body: str = ""
    release_notes_section: str = ""
    summary_section: str = ""
    why_it_matters_section: str = ""
    commits: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    merged_at: datetime
    repo_name: str


class SlackMessage(BaseModel):
    """A single Slack message with optional thread context."""

    text: str
    timestamp: str
    user: str = ""
    thread_replies: list[str] = Field(default_factory=list)
    channel: str = ""


class SlackContext(BaseModel):
    """Slack context matched to a PR."""

    messages: list[SlackMessage] = Field(default_factory=list)
    matched_keywords: list[str] = Field(default_factory=list)


class EnrichedChange(BaseModel):
    """A merged PR enriched with optional Slack context."""

    pr: MergedPR
    slack_context: SlackContext = Field(default_factory=SlackContext)


class AnalyzedChange(BaseModel):
    """A semantically analyzed change ready for release notes."""

    what_was_done: str
    why_it_matters: str
    impact_description: str
    impact_type: ImpactType
    category: ChangeCategory
    pr_url: str
    pr_number: int
    repo_name: str
    is_significant: bool = True  # False means filtered out as noise


class ReleaseNotes(BaseModel):
    """Complete release notes output."""

    date_range_start: datetime
    date_range_end: datetime
    features: list[AnalyzedChange] = Field(default_factory=list)
    improvements: list[AnalyzedChange] = Field(default_factory=list)
    fixes: list[AnalyzedChange] = Field(default_factory=list)
    internal_changes: list[AnalyzedChange] = Field(default_factory=list)
    raw_prs: list[MergedPR] = Field(default_factory=list)
