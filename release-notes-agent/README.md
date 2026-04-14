# HydraDB Weekly Release Notes Agent

Autonomous agent that generates high-quality, impact-driven weekly release notes for HydraDB and posts them to Slack.

## How It Works

1. **Data Collection** (`fetch_prs.py`) — Queries GitHub API for all PRs merged in the last 7 days across HydraDB product repos
2. **Filtering** — Automatically excludes noise (lock file bumps, WIP, backports, CI-only changes, pure refactors)
3. **Deduplication** — Cross-repo PRs for the same logical change are merged into a single entry
4. **Analysis** — The Vorflux scheduled session (Claude) analyzes structured PR data and generates semantic, impact-first release notes
5. **Delivery** — Posts formatted release notes to the configured Slack channel

## Repos Scanned

### Core (always included)
- `cortex-application` — Core backend (search, retrieval, auth, multi-tenant)
- `cortex-ingestion` — Ingestion pipeline (chunking, embedding, indexing)

### Product (included if active)
- `cortex-dashboard` — Customer-facing dashboard
- `hydradb-cli` — HydraDB CLI
- `hydradb-mcp` — MCP server for AI coding tools
- `hydradb-claude-code` — Claude Code plugin
- `python-sdk` / `ts-sdk` — Official SDKs
- `docs` — Public documentation
- `hydradb-on-prem-infra` — On-premises infrastructure

## Setup (Required)

### 1. Slack Channel

The agent posts release notes via a Slack Incoming Webhook. To use a dedicated channel:

1. Create a `#release-notes` channel in your Slack workspace
2. Go to [Slack App Management](https://api.slack.com/apps) > Your App > Incoming Webhooks
3. Click "Add New Webhook to Workspace" and select the `#release-notes` channel
4. Copy the new webhook URL

### 2. Vorflux Secrets

These secrets must be configured in your Vorflux account (Settings > Secrets):

| Secret | Description |
|--------|-------------|
| `GITHUB_TOKEN` | GitHub PAT with `repo` scope for the `usecortex` org |
| `SLACK_WEBHOOK_URL` | Slack Incoming Webhook URL for the target channel |

### 3. Schedule

The Vorflux scheduled session runs **every Friday at 5:00 PM IST** (11:30 AM UTC). This is set up via the `create_schedule` tool — confirm the schedule card in your Vorflux session.

## Manual Run

To trigger release notes on any day, start a new Vorflux session and paste the prompt from `AGENT_PROMPT.md`.

Or run the data collection script directly:

```bash
cd /path/to/hydradb-bench

# Fetch PR data (outputs JSON to stdout)
python release-notes-agent/generate.py

# Custom lookback window
python release-notes-agent/generate.py --days 14

# Include infra repos
python release-notes-agent/generate.py --include-infra

# Post to Slack
python release-notes-agent/post_to_slack.py --file release_notes.md
```

## Output Format

```
:newspaper: *HydraDB Weekly Release Notes — Week of [DATE]*

## :rocket: New Features
- <What was built> — <Why it matters> — <Impact>

## :zap: Improvements
- <What changed> — <Why it matters> — <Impact>

## :bug: Fixes
- <Issue fixed> — <Root cause or context> — <Impact>

## :brain: Internal Changes (optional)
- <Infra / refactor / tooling> — <Why it matters>
```

## Configuration

Edit `config.py` to:
- Add/remove repos from scan
- Adjust noise filters and refactor keywords
- Change the default lookback window

## Architecture

```
Vorflux Scheduled Session (Friday 5PM IST)
  |
  v
generate.py  -->  fetch_prs.py  -->  GitHub API (gh CLI)
  |                                    |
  |  <-- structured JSON (PRs) --------+
  |
  v
Agent (Claude) analyzes PRs, generates release notes
  |
  v
post_to_slack.py  -->  Slack Webhook  -->  #release-notes channel
```
