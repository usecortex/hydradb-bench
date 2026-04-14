#!/usr/bin/env python3
"""Unit tests for the release notes agent."""

from __future__ import annotations

import os
import sys
import unittest

# Ensure imports work regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch_prs import (
    is_noise,
    is_pure_refactor,
    classify_pr,
    extract_release_notes_section,
    extract_summary_section,
    extract_key_changes_section,
    extract_why_it_matters_section,
    build_pr_summary,
)
from post_to_slack import markdown_to_slack_mrkdwn


class TestNoiseFilter(unittest.TestCase):
    """Tests for is_noise() — PR noise detection."""

    def _pr(self, title: str) -> dict:
        return {"title": title, "body": ""}

    def test_test_prefix_filtered(self):
        self.assertTrue(is_noise(self._pr("TEST: Add executor-level coverage")))

    def test_test_prefix_lowercase_filtered(self):
        self.assertTrue(is_noise(self._pr("test: add unit tests for auth")))

    def test_ci_prefix_filtered(self):
        self.assertTrue(is_noise(self._pr("ci: add CI workflow with lint")))

    def test_ci_prefix_uppercase_filtered(self):
        self.assertTrue(is_noise(self._pr("CI: fix pipeline")))

    def test_bump_prefix_filtered(self):
        self.assertTrue(is_noise(self._pr("BUMP: Milvus limits")))

    def test_bump_lowercase_filtered(self):
        self.assertTrue(is_noise(self._pr("bump: version to 2.0")))

    def test_chore_bump_filtered(self):
        self.assertTrue(is_noise(self._pr("chore: bump dependencies")))

    def test_backport_filtered(self):
        self.assertTrue(is_noise(self._pr("main to staging backport")))

    def test_staging_backport_filtered(self):
        self.assertTrue(is_noise(self._pr("staging to main backport")))

    def test_single_word_staging_filtered(self):
        self.assertTrue(is_noise(self._pr("Staging")))

    def test_single_word_random_filtered(self):
        self.assertTrue(is_noise(self._pr("Hotfix")))

    def test_wip_filtered(self):
        self.assertTrue(is_noise(self._pr("WIP: new feature")))

    def test_draft_filtered(self):
        self.assertTrue(is_noise(self._pr("draft: experimental change")))

    def test_release_notes_workflow_filtered(self):
        self.assertTrue(is_noise(self._pr("Fix Slack notification step in weekly release notes workflow")))

    def test_pr_template_filtered(self):
        self.assertTrue(is_noise(self._pr("Add PR template and weekly release notes workflow")))

    def test_feat_not_filtered(self):
        self.assertFalse(is_noise(self._pr("feat: add query understanding agent")))

    def test_fix_not_filtered(self):
        self.assertFalse(is_noise(self._pr("fix: preserve Indic text in entity normalization")))

    def test_perf_not_filtered(self):
        self.assertFalse(is_noise(self._pr("perf: add workflow timing logs")))

    def test_chore_meaningful_not_filtered(self):
        self.assertFalse(is_noise(self._pr("CHORE: RRF Integration QU v2")))

    def test_staging_in_context_not_filtered(self):
        self.assertFalse(is_noise(self._pr("feat: add staging environment support")))

    def test_fix_with_staging_not_filtered(self):
        self.assertFalse(is_noise(self._pr("fix: staging deployment timeout")))


class TestRefactorFilter(unittest.TestCase):
    """Tests for is_pure_refactor() — refactor detection."""

    def _pr(self, title: str, body: str = "") -> dict:
        return {"title": title, "body": body}

    def test_pure_refactor_filtered(self):
        self.assertTrue(is_pure_refactor(self._pr("refactor: rename variables")))

    def test_chore_cleanup_filtered(self):
        self.assertTrue(is_pure_refactor(self._pr("chore: cleanup old code")))

    def test_chore_lint_filtered(self):
        self.assertTrue(is_pure_refactor(self._pr("chore: lint fixes")))

    def test_license_change_filtered(self):
        self.assertTrue(is_pure_refactor(self._pr("chore: switch license from MIT to Apache 2.0")))

    def test_community_files_filtered(self):
        self.assertTrue(is_pure_refactor(self._pr("chore: add community and governance files")))

    def test_refactor_with_impact_kept(self):
        self.assertFalse(is_pure_refactor(self._pr(
            "refactor: rewrite query engine",
            body="This improves performance by 2x and fixes a latency bug"
        )))

    def test_refactor_with_user_impact_kept(self):
        self.assertFalse(is_pure_refactor(self._pr(
            "refactor: restructure auth module",
            body="Fixes a security issue where user tokens could leak"
        )))

    def test_feat_not_refactor(self):
        self.assertFalse(is_pure_refactor(self._pr("feat: add new feature")))

    def test_fix_not_refactor(self):
        self.assertFalse(is_pure_refactor(self._pr("fix: resolve crash")))


class TestClassifyPR(unittest.TestCase):
    """Tests for classify_pr() — PR category classification."""

    def test_feat_prefix(self):
        self.assertEqual(classify_pr("feat: add new feature", ""), "feature")

    def test_feat_uppercase(self):
        self.assertEqual(classify_pr("FEAT: Add RRF pipeline", ""), "feature")

    def test_feat_slash(self):
        self.assertEqual(classify_pr("Feat/conditional parse", ""), "feature")

    def test_fix_prefix(self):
        self.assertEqual(classify_pr("fix: resolve crash", ""), "fix")

    def test_fix_uppercase(self):
        self.assertEqual(classify_pr("FIX: Document Metadata search", ""), "fix")

    def test_fix_slash(self):
        self.assertEqual(classify_pr("Fix/on prem", ""), "fix")

    def test_perf_prefix(self):
        self.assertEqual(classify_pr("perf: optimize query", ""), "improvement")

    def test_chore_prefix(self):
        self.assertEqual(classify_pr("chore: update deps", ""), "internal")

    def test_refactor_prefix(self):
        self.assertEqual(classify_pr("refactor: clean up code", ""), "internal")

    def test_docs_prefix(self):
        self.assertEqual(classify_pr("docs: update README", ""), "docs")

    def test_improve_keyword(self):
        self.assertEqual(classify_pr("Enhance logging behavior", ""), "improvement")

    def test_add_prefix_feature(self):
        self.assertEqual(classify_pr("Add fallback retry when pre-filter returns 0", ""), "feature")

    def test_add_in_middle_not_feature(self):
        # "add" as substring in middle of word should NOT trigger feature
        self.assertNotEqual(classify_pr("Fix address validation", ""), "feature")

    def test_body_signals_feature(self):
        self.assertEqual(classify_pr("Some vague title", "This introduces a new capability"), "feature")

    def test_body_signals_fix(self):
        self.assertEqual(classify_pr("Some vague title", "This fixes a regression in auth"), "fix")


class TestExtractSections(unittest.TestCase):
    """Tests for PR body section extraction."""

    def test_extract_release_notes(self):
        body = "## Summary\nSome summary\n\n## Release Notes\n- Added feature X\n- Fixed bug Y\n\n## Testing\nAll tests pass"
        result = extract_release_notes_section(body)
        self.assertIn("Added feature X", result)
        self.assertIn("Fixed bug Y", result)

    def test_extract_release_notes_case_insensitive(self):
        body = "## release notes\n- Item 1\n- Item 2"
        result = extract_release_notes_section(body)
        self.assertIn("Item 1", result)

    def test_extract_release_notes_missing(self):
        body = "## Summary\nJust a summary"
        self.assertIsNone(extract_release_notes_section(body))

    def test_extract_release_notes_empty(self):
        body = "## Release Notes\n-\n## Testing"
        self.assertIsNone(extract_release_notes_section(body))

    def test_extract_summary(self):
        body = "## Summary\nThis is the summary of changes\n\n## Key Changes\nDetails"
        result = extract_summary_section(body)
        self.assertIn("summary of changes", result)

    def test_extract_key_changes(self):
        body = "## Key Changes\n### Queue Routing\n- Added new routing logic\n\n## Testing"
        result = extract_key_changes_section(body)
        self.assertIn("Queue Routing", result)

    def test_extract_why_it_matters(self):
        body = "## Why it matters\nReduces latency for enterprise customers\n\n## Testing"
        result = extract_why_it_matters_section(body)
        self.assertIn("Reduces latency", result)

    def test_extract_none_on_empty_body(self):
        self.assertIsNone(extract_release_notes_section(""))
        self.assertIsNone(extract_summary_section(""))
        self.assertIsNone(extract_key_changes_section(""))
        self.assertIsNone(extract_why_it_matters_section(""))

    def test_extract_none_on_none_body(self):
        self.assertIsNone(extract_release_notes_section(None))
        self.assertIsNone(extract_summary_section(None))


class TestBuildPRSummary(unittest.TestCase):
    """Tests for build_pr_summary() — structured PR output."""

    def test_basic_summary(self):
        pr = {
            "title": "feat: add new search mode",
            "body": "## Summary\nAdded vector search\n\n## Release Notes\n- New search mode",
            "number": 42,
            "url": "https://github.com/usecortex/cortex-application/pull/42",
            "repo": "cortex-application",
            "merged_at_parsed": "2026-04-14T10:00:00+00:00",
            "headRefName": "feat/search-mode",
            "author": {"login": "dev1"},
            "labels": [{"name": "feature"}],
            "commits": [
                {"messageHeadline": "feat: add vector search", "messageBody": ""},
                {"messageHeadline": "Merge branch main", "messageBody": ""},
            ],
        }
        result = build_pr_summary(pr)
        self.assertEqual(result["category"], "feature")
        self.assertEqual(result["number"], 42)
        self.assertEqual(result["repo"], "cortex-application")
        self.assertIn("New search mode", result["release_notes"])
        self.assertIn("vector search", result["summary"])
        # Merge commits should be excluded
        self.assertEqual(len(result["commit_messages"]), 1)
        self.assertNotIn("Merge branch main", result["commit_messages"])

    def test_summary_with_no_body(self):
        pr = {
            "title": "fix: crash on startup",
            "body": None,
            "number": 10,
            "url": "",
            "repo": "cortex-ingestion",
            "merged_at_parsed": "2026-04-14T10:00:00+00:00",
            "headRefName": "fix/crash",
            "author": None,
            "labels": [],
            "commits": [],
        }
        result = build_pr_summary(pr)
        self.assertEqual(result["category"], "fix")
        self.assertIsNone(result["release_notes"])
        self.assertIsNone(result["summary"])
        self.assertEqual(result["author"], "unknown")


class TestMarkdownToSlackMrkdwn(unittest.TestCase):
    """Tests for markdown_to_slack_mrkdwn() — Slack formatting conversion."""

    def test_h2_header_conversion(self):
        result = markdown_to_slack_mrkdwn("## :rocket: New Features")
        self.assertIn("*:rocket: New Features*", result)
        self.assertNotIn("##", result)

    def test_h3_header_conversion(self):
        result = markdown_to_slack_mrkdwn("### Sub-header")
        self.assertIn("*Sub-header*", result)
        self.assertNotIn("###", result)

    def test_bold_conversion(self):
        result = markdown_to_slack_mrkdwn("This is **bold text** here")
        self.assertIn("*bold text*", result)
        self.assertNotIn("**", result)

    def test_bullets_preserved(self):
        result = markdown_to_slack_mrkdwn("- Item one\n- Item two")
        self.assertIn("- Item one", result)
        self.assertIn("- Item two", result)

    def test_link_conversion(self):
        result = markdown_to_slack_mrkdwn("[Click here](https://example.com)")
        self.assertIn("<https://example.com|Click here>", result)

    def test_code_preserved(self):
        result = markdown_to_slack_mrkdwn("Use `pip install hydradb`")
        self.assertIn("`pip install hydradb`", result)

    def test_full_release_notes_format(self):
        md = """:newspaper: *HydraDB Weekly Release Notes*

## :rocket: New Features
- **Dedicated Ingestion** — Enterprise customers get dedicated queues

## :bug: Fixes
- **Search crash** — Fixed null pointer in search handler
"""
        result = markdown_to_slack_mrkdwn(md)
        # Headers should be bold, not ## prefixed
        self.assertNotIn("## ", result)
        # Bold items should use single asterisks
        self.assertIn("*Dedicated Ingestion*", result)
        self.assertIn("*Search crash*", result)
        # Bullets should be preserved
        self.assertIn("- *Dedicated Ingestion*", result)

    def test_nested_bold_italic_no_collision(self):
        # Edge case: bold text followed by italic in same line
        md = "- **Title** — *significantly* better"
        result = markdown_to_slack_mrkdwn(md)
        # Should not have triple asterisks or broken formatting
        self.assertNotIn("***", result)

    def test_empty_input(self):
        self.assertEqual(markdown_to_slack_mrkdwn(""), "")

    def test_plain_text_unchanged(self):
        text = "Just plain text with no markdown"
        self.assertEqual(markdown_to_slack_mrkdwn(text), text)


if __name__ == "__main__":
    unittest.main()
