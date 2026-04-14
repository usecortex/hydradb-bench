"""
Configuration for the HydraDB Release Notes Agent.

Repos are split into tiers:
  - CORE: cortex-application, cortex-ingestion (always included)
  - PRODUCT: user-facing plugins, SDKs, dashboard, docs (included if they have activity)
  - INFRA: internal tooling, infra (excluded by default unless impactful)
"""

from __future__ import annotations

# ── GitHub org ────────────────────────────────────────────────────────
GITHUB_ORG = "usecortex"

# ── Repository tiers ──────────────────────────────────────────────────
CORE_REPOS: list[str] = [
    "cortex-application",
    "cortex-ingestion",
]

PRODUCT_REPOS: list[str] = [
    "cortex-dashboard",
    "hydradb-cli",
    "hydradb-mcp",
    "hydradb-claude-code",
    "python-sdk",
    "ts-sdk",
    "docs",
    "hydradb-on-prem-infra",
]

# Infra repos - only included if explicitly requested
INFRA_REPOS: list[str] = [
    "cortex-infra-scripts",
    "argocd-infra",
    "cortex-testing",
]

# Default: core + product repos
DEFAULT_REPOS: list[str] = CORE_REPOS + PRODUCT_REPOS

# ── Time window ───────────────────────────────────────────────────────
DEFAULT_LOOKBACK_DAYS = 7

# ── PR filtering ──────────────────────────────────────────────────────
# PR title prefixes that indicate noise (case-insensitive match)
NOISE_PREFIXES: list[str] = [
    "chore: bump",
    "chore: update lock",
    "chore: merge",
    "revert \"revert",
    "bump:",
    "test:",
    "ci:",
]

# PR title substrings that indicate noise (case-insensitive)
NOISE_SUBSTRINGS: list[str] = [
    "to staging backport",
    "main to staging",
    "staging to main",
    "release notes workflow",
    "pr template",
    "merge conflict",
    "revert revert",
]

# PR title patterns that indicate pure refactors (excluded unless body mentions impact)
REFACTOR_KEYWORDS: list[str] = [
    "refactor:",
    "chore: rename",
    "chore: cleanup",
    "chore: lint",
    "chore: format",
    "chore: add community",
    "chore: add dependabot",
    "chore: switch license",
]

# ── Slack ─────────────────────────────────────────────────────────────
# The webhook URL is read from SLACK_WEBHOOK_URL env var at runtime.
# To use a dedicated release-notes channel, create a webhook for that channel.
