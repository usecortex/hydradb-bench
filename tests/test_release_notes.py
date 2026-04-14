"""Unit tests for the release notes agent."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from release_notes.analyzer import _deduplicate, analyze_changes
from release_notes.formatter import format_release_notes, save_release_notes
from release_notes.github_collector import _parse_section
from release_notes.models import (
    AnalyzedChange,
    ChangeCategory,
    EnrichedChange,
    ImpactType,
    MergedPR,
    ReleaseNotes,
    SlackMessage,
    extract_keywords,
)
from release_notes.slack_collector import match_slack_context

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestModels:
    """Test Pydantic model validation and defaults."""

    def test_merged_pr_required_fields(self):
        pr = MergedPR(
            number=1,
            title="Test PR",
            url="https://github.com/org/repo/pull/1",
            author="testuser",
            merged_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
            repo_name="org/repo",
        )
        assert pr.number == 1
        assert pr.commits == []
        assert pr.labels == []
        assert pr.body == ""
        assert pr.release_notes_section == ""

    def test_merged_pr_with_all_fields(self):
        pr = MergedPR(
            number=42,
            title="feat: add vector search",
            url="https://github.com/org/repo/pull/42",
            author="dev",
            body="## Summary\nAdded vector search",
            release_notes_section="- Added vector search capability",
            summary_section="Added vector search",
            why_it_matters_section="Enables semantic queries",
            commits=["add vector search", "fix tests"],
            labels=["feature", "search"],
            merged_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
            repo_name="org/repo",
        )
        assert len(pr.commits) == 2
        assert "feature" in pr.labels

    def test_analyzed_change_defaults(self):
        change = AnalyzedChange(
            what_was_done="Added feature",
            why_it_matters="Improves UX",
            impact_description="Better search",
            impact_type=ImpactType.USER_FACING,
            category=ChangeCategory.NEW_FEATURE,
            pr_url="https://github.com/org/repo/pull/1",
            pr_number=1,
            repo_name="org/repo",
        )
        assert change.is_significant is True

    def test_release_notes_empty_lists(self):
        notes = ReleaseNotes(
            date_range_start=datetime(2024, 1, 8, tzinfo=timezone.utc),
            date_range_end=datetime(2024, 1, 15, tzinfo=timezone.utc),
        )
        assert notes.features == []
        assert notes.improvements == []
        assert notes.fixes == []
        assert notes.internal_changes == []
        assert notes.raw_prs == []


# ---------------------------------------------------------------------------
# GitHub Collector — section parsing
# ---------------------------------------------------------------------------


class TestParseSection:
    """Test markdown section extraction from PR bodies."""

    def test_parse_release_notes_section(self):
        body = (
            "## Summary\nSome summary\n\n"
            "## Release Notes\n- Added feature X\n- Fixed bug Y\n\n"
            "## Checklist\n- [x] Tests"
        )
        result = _parse_section(body, "Release Notes")
        assert "Added feature X" in result
        assert "Fixed bug Y" in result
        assert "Checklist" not in result

    def test_parse_summary_section(self):
        body = "## Summary\nThis is the summary\n\n## Other\nStuff"
        result = _parse_section(body, "Summary")
        assert result == "This is the summary"

    def test_parse_why_it_matters(self):
        body = "## Why it matters\nImproves latency by 50%\n\n## Details\nMore info"
        result = _parse_section(body, "Why it matters")
        assert result == "Improves latency by 50%"

    def test_parse_missing_section(self):
        body = "## Summary\nSome text\n\n## Checklist\n- Done"
        result = _parse_section(body, "Release Notes")
        assert result == ""

    def test_parse_empty_body(self):
        assert _parse_section("", "Summary") == ""
        assert _parse_section(None, "Summary") == ""

    def test_parse_section_at_end_of_body(self):
        body = "## Summary\nFirst section\n\n## Release Notes\n- Last section content"
        result = _parse_section(body, "Release Notes")
        assert result == "- Last section content"

    def test_parse_case_insensitive(self):
        body = "## RELEASE NOTES\n- Feature A\n\n## Other\nStuff"
        result = _parse_section(body, "Release Notes")
        assert "Feature A" in result

    def test_parse_section_with_extra_whitespace(self):
        body = "##  Summary \nContent here\n\n## Next\nMore"
        result = _parse_section(body, "Summary")
        assert result == "Content here"


# ---------------------------------------------------------------------------
# Slack Collector — keyword extraction and matching
# ---------------------------------------------------------------------------


class TestSlackKeywords:
    """Test keyword extraction logic."""

    def test_extract_keywords_basic(self):
        keywords = extract_keywords("Added vector search capability for large datasets")
        assert "vector" in keywords
        assert "search" in keywords
        assert "capability" in keywords
        assert "large" in keywords
        assert "datasets" in keywords

    def test_extract_keywords_filters_short_words(self):
        keywords = extract_keywords("a to the and or is it")
        assert len(keywords) == 0

    def test_extract_keywords_filters_stop_words(self):
        keywords = extract_keywords("this that with from have been")
        assert len(keywords) == 0

    def test_extract_keywords_lowercases(self):
        keywords = extract_keywords("Vector SEARCH Capability")
        assert "vector" in keywords
        assert "search" in keywords


class TestSlackMatching:
    """Test PR-to-Slack message matching."""

    def _make_pr(self, **kwargs) -> MergedPR:
        defaults = dict(
            number=42,
            title="feat: add vector search",
            url="https://github.com/org/repo/pull/42",
            author="dev",
            merged_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
            repo_name="org/repo",
        )
        defaults.update(kwargs)
        return MergedPR(**defaults)

    def _make_msg(self, text: str, **kwargs) -> SlackMessage:
        defaults = dict(timestamp="1705300000.000000", channel="engineering")
        defaults.update(kwargs)
        return SlackMessage(text=text, **defaults)

    def test_match_by_url(self):
        pr = self._make_pr()
        msg = self._make_msg(f"Check out {pr.url} for the new search feature")
        ctx = match_slack_context(pr, [msg])
        assert len(ctx.messages) == 1
        assert "url_match" in ctx.matched_keywords

    def test_match_by_pr_number(self):
        pr = self._make_pr(number=42)
        msg = self._make_msg("PR #42 in repo looks good, approved")
        ctx = match_slack_context(pr, [msg])
        assert len(ctx.messages) == 1
        assert "pr_ref_#42" in ctx.matched_keywords

    def test_match_by_keywords(self):
        pr = self._make_pr(
            title="feat: add vector search capability",
            summary_section="vector search for large datasets",
        )
        msg = self._make_msg("The vector search capability is looking great in staging")
        ctx = match_slack_context(pr, [msg])
        assert len(ctx.messages) == 1

    def test_no_match(self):
        pr = self._make_pr(title="feat: add vector search")
        msg = self._make_msg("Unrelated discussion about deployment")
        ctx = match_slack_context(pr, [msg])
        assert len(ctx.messages) == 0

    def test_empty_messages(self):
        pr = self._make_pr()
        ctx = match_slack_context(pr, [])
        assert len(ctx.messages) == 0
        assert len(ctx.matched_keywords) == 0

    def test_match_in_thread_replies(self):
        pr = self._make_pr(number=99)
        msg = self._make_msg(
            "Discussing the new feature in repo",
            thread_replies=["PR #99 is related to this"],
        )
        ctx = match_slack_context(pr, [msg])
        assert len(ctx.messages) == 1


# ---------------------------------------------------------------------------
# Analyzer — deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Test the deduplication heuristic."""

    def _make_change(self, what: str, category: ChangeCategory, repo: str = "org/repo") -> AnalyzedChange:
        return AnalyzedChange(
            what_was_done=what,
            why_it_matters="matters",
            impact_description="impact",
            impact_type=ImpactType.USER_FACING,
            category=category,
            pr_url="https://github.com/org/repo/pull/1",
            pr_number=1,
            repo_name=repo,
        )

    def test_no_duplicates(self):
        changes = [
            self._make_change("Added vector search capability", ChangeCategory.NEW_FEATURE),
            self._make_change("Fixed authentication timeout bug", ChangeCategory.BUG_FIX),
        ]
        result = _deduplicate(changes)
        assert len(result) == 2

    def test_dedup_similar_changes(self):
        changes = [
            self._make_change(
                "Added vector search capability for large datasets",
                ChangeCategory.NEW_FEATURE,
            ),
            self._make_change(
                "Fixed vector search capability edge case for large datasets",
                ChangeCategory.BUG_FIX,
            ),
        ]
        result = _deduplicate(changes)
        assert len(result) == 1
        # Should keep the higher-priority one (NEW_FEATURE)
        assert result[0].category == ChangeCategory.NEW_FEATURE

    def test_no_dedup_across_repos(self):
        changes = [
            self._make_change("Added vector search capability", ChangeCategory.NEW_FEATURE, "org/repo-a"),
            self._make_change("Added vector search capability", ChangeCategory.NEW_FEATURE, "org/repo-b"),
        ]
        result = _deduplicate(changes)
        assert len(result) == 2

    def test_single_change(self):
        changes = [self._make_change("Something", ChangeCategory.IMPROVEMENT)]
        result = _deduplicate(changes)
        assert len(result) == 1

    def test_empty_list(self):
        assert _deduplicate([]) == []


# ---------------------------------------------------------------------------
# Analyzer — analyze_changes with mocked OpenAI
# ---------------------------------------------------------------------------


class TestAnalyzeChanges:
    """Test the analyzer with mocked OpenAI responses."""

    def _make_enriched(self, title: str = "feat: add search", number: int = 1) -> EnrichedChange:
        pr = MergedPR(
            number=number,
            title=title,
            url=f"https://github.com/org/repo/pull/{number}",
            author="dev",
            body="## Summary\nAdded search\n\n## Release Notes\n- New search feature",
            release_notes_section="- New search feature",
            summary_section="Added search",
            merged_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
            repo_name="org/repo",
        )
        return EnrichedChange(pr=pr)

    @patch("release_notes.analyzer._get_openai_client")
    def test_analyze_single_change(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "what_was_done": "Added full-text search",
                            "why_it_matters": "Enables keyword queries",
                            "impact_description": "Users can now search by keywords",
                            "impact_type": "user_facing",
                            "category": "new_feature",
                            "is_significant": True,
                        }
                    )
                )
            )
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        changes = [self._make_enriched()]
        notes = analyze_changes(changes, days=7)

        assert len(notes.features) == 1
        assert notes.features[0].what_was_done == "Added full-text search"
        assert notes.features[0].impact_type == ImpactType.USER_FACING

    @patch("release_notes.analyzer._get_openai_client")
    def test_analyze_filters_insignificant(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "what_was_done": "Updated CI config",
                            "why_it_matters": "Minor cleanup",
                            "impact_description": "No user impact",
                            "impact_type": "organizational",
                            "category": "internal",
                            "is_significant": False,
                        }
                    )
                )
            )
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        changes = [self._make_enriched(title="chore: update CI")]
        notes = analyze_changes(changes, days=7)

        assert len(notes.features) == 0
        assert len(notes.internal_changes) == 0  # filtered as insignificant

    def test_analyze_empty_changes(self):
        notes = analyze_changes([], days=7)
        assert len(notes.features) == 0
        assert len(notes.raw_prs) == 0

    @patch("release_notes.analyzer._get_openai_client")
    def test_analyze_handles_invalid_json(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="not valid json"))]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        changes = [self._make_enriched()]
        notes = analyze_changes(changes, days=7)

        # Should gracefully skip the bad response
        assert len(notes.features) == 0

    @patch("release_notes.analyzer._get_openai_client")
    def test_analyze_handles_markdown_fenced_json(self, mock_get_client):
        json_content = json.dumps(
            {
                "what_was_done": "Added feature",
                "why_it_matters": "Matters",
                "impact_description": "Impact",
                "impact_type": "user_facing",
                "category": "new_feature",
                "is_significant": True,
            }
        )
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=f"```json\n{json_content}\n```"))]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        changes = [self._make_enriched()]
        notes = analyze_changes(changes, days=7)

        assert len(notes.features) == 1


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


class TestFormatter:
    """Test markdown output formatting."""

    def _make_notes(self, **kwargs) -> ReleaseNotes:
        defaults = dict(
            date_range_start=datetime(2024, 1, 8, tzinfo=timezone.utc),
            date_range_end=datetime(2024, 1, 15, tzinfo=timezone.utc),
        )
        defaults.update(kwargs)
        return ReleaseNotes(**defaults)

    def _make_change(
        self,
        what: str = "Added feature",
        category: ChangeCategory = ChangeCategory.NEW_FEATURE,
        impact_type: ImpactType = ImpactType.USER_FACING,
    ) -> AnalyzedChange:
        return AnalyzedChange(
            what_was_done=what,
            why_it_matters="Improves UX",
            impact_description="Better experience",
            impact_type=impact_type,
            category=category,
            pr_url="https://github.com/org/repo/pull/1",
            pr_number=1,
            repo_name="org/repo",
        )

    def test_format_with_all_sections(self):
        notes = self._make_notes(
            features=[self._make_change("New search", ChangeCategory.NEW_FEATURE)],
            improvements=[self._make_change("Faster queries", ChangeCategory.IMPROVEMENT)],
            fixes=[self._make_change("Fixed crash", ChangeCategory.BUG_FIX)],
            internal_changes=[
                self._make_change("Refactored DB layer", ChangeCategory.INTERNAL, ImpactType.ORGANIZATIONAL)
            ],
        )
        output = format_release_notes(notes)

        assert "## New Features" in output
        assert "## Improvements" in output
        assert "## Fixes" in output
        assert "## Internal Changes" in output
        assert "New search" in output
        assert "Faster queries" in output
        assert "Fixed crash" in output
        assert "Refactored DB layer" in output

    def test_format_omits_empty_sections(self):
        notes = self._make_notes(
            features=[self._make_change("New feature")],
        )
        output = format_release_notes(notes)

        assert "## New Features" in output
        assert "## Improvements" not in output
        assert "## Fixes" not in output
        assert "## Internal Changes" not in output

    def test_format_user_facing_highlight(self):
        notes = self._make_notes(
            features=[self._make_change("Search", impact_type=ImpactType.USER_FACING)],
        )
        output = format_release_notes(notes)
        assert "**[User-Facing]**" in output

    def test_format_no_user_facing_for_org_impact(self):
        notes = self._make_notes(
            improvements=[self._make_change("Infra update", ChangeCategory.IMPROVEMENT, ImpactType.ORGANIZATIONAL)],
        )
        output = format_release_notes(notes)
        assert "**[User-Facing]**" not in output

    def test_format_includes_pr_link(self):
        notes = self._make_notes(
            features=[self._make_change()],
        )
        output = format_release_notes(notes)
        assert "[PR #1]" in output
        assert "github.com/org/repo/pull/1" in output

    def test_format_empty_notes(self):
        notes = self._make_notes()
        output = format_release_notes(notes)
        assert "No significant changes this week" in output

    def test_format_includes_date_range(self):
        notes = self._make_notes()
        output = format_release_notes(notes)
        assert "January 08, 2024" in output
        assert "January 15, 2024" in output

    def test_format_includes_footer(self):
        notes = self._make_notes(
            features=[self._make_change()],
            raw_prs=[
                MergedPR(
                    number=1,
                    title="t",
                    url="u",
                    author="a",
                    merged_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
                    repo_name="org/repo",
                )
            ],
        )
        output = format_release_notes(notes)
        assert "Generated from 1 merged PRs" in output
        assert "1 significant changes highlighted" in output

    def test_save_release_notes(self, tmp_path):
        notes = self._make_notes(
            features=[self._make_change()],
        )
        filepath = save_release_notes(notes, output_dir=str(tmp_path))
        assert filepath.exists()
        assert filepath.name == "release-notes-2024-01-15.md"
        content = filepath.read_text()
        assert "## New Features" in content


# ---------------------------------------------------------------------------
# Integration — full pipeline with mocks
# ---------------------------------------------------------------------------


class TestIntegration:
    """Test the full pipeline with mocked external services."""

    @patch("release_notes.analyzer._get_openai_client")
    def test_full_pipeline(self, mock_get_client, tmp_path):
        """End-to-end: PR data -> enrichment -> analysis -> formatting -> save."""
        # Mock OpenAI
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "what_was_done": "Added RRF pipeline for hybrid search",
                            "why_it_matters": "Combines BM25 and vector results for better relevance",
                            "impact_description": "Search quality improves by ~20% on mixed queries",
                            "impact_type": "user_facing",
                            "category": "new_feature",
                            "is_significant": True,
                        }
                    )
                )
            )
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        # Create test PR
        pr = MergedPR(
            number=295,
            title="FEAT: Add RRF pipeline for hybrid search",
            url="https://github.com/usecortex/cortex-application/pull/295",
            author="aadi",
            body="## Summary\nAdded RRF\n\n## Release Notes\n- RRF pipeline for hybrid search",
            release_notes_section="- RRF pipeline for hybrid search",
            summary_section="Added RRF",
            commits=["add RRF pipeline", "fix tests"],
            labels=["feature"],
            merged_at=datetime(2024, 1, 14, tzinfo=timezone.utc),
            repo_name="usecortex/cortex-application",
        )

        # Enrich (no Slack)
        enriched = [EnrichedChange(pr=pr)]

        # Analyze
        notes = analyze_changes(enriched, days=7)
        assert len(notes.features) == 1

        # Format
        output = format_release_notes(notes)
        assert "RRF pipeline" in output
        assert "**[User-Facing]**" in output

        # Save
        filepath = save_release_notes(notes, output_dir=str(tmp_path))
        assert filepath.exists()
        saved = filepath.read_text()
        assert "RRF pipeline" in saved
