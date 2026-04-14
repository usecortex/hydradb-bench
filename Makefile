.DEFAULT_GOAL := help
SHELL := /bin/bash

VENV      := .venv
PYTHON    := $(VENV)/bin/python
PIP       := $(VENV)/bin/pip
ACTIVATE  := source $(VENV)/bin/activate

.PHONY: help bootstrap install run generate-data report convert-report lint format clean release-notes release-notes-dry

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

bootstrap: ## Full setup: venv, install, .env (zero-to-running)
	@bash scripts/bootstrap.sh

install: ## Install dependencies into venv
	@if [ ! -d "$(VENV)" ]; then python3 -m venv $(VENV); fi
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

run: ## Run the benchmark (default: --provider hydradb)
	$(PYTHON) run_benchmark.py --provider hydradb

run-both: ## Run benchmark for both providers with comparison report
	$(PYTHON) run_benchmark.py --provider both

generate-data: ## Generate synthetic test data from source documents
	$(PYTHON) generate_test_data.py

report: ## Convert latest JSON report to CSV
	@latest=$$(ls -t reports/*.json 2>/dev/null | head -1); \
	if [ -z "$$latest" ]; then \
		echo "No JSON reports found in reports/. Run a benchmark first."; \
		exit 1; \
	fi; \
	echo "Converting $$latest to CSV..."; \
	$(PYTHON) json_to_csv.py "$$latest"

convert-report: ## Convert a specific report: make convert-report FILE=reports/foo.json
	$(PYTHON) json_to_csv.py "$(FILE)"

lint: ## Run linting checks (ruff)
	$(VENV)/bin/ruff check .
	$(VENV)/bin/ruff format --check .

format: ## Auto-format code (ruff)
	$(VENV)/bin/ruff check --fix .
	$(VENV)/bin/ruff format .

release-notes: ## Generate weekly release notes for HydraDB
	$(PYTHON) generate_release_notes.py

release-notes-dry: ## Dry run: show merged PRs without generating notes
	$(PYTHON) generate_release_notes.py --dry-run

clean: ## Remove generated/build artifacts
	rm -rf $(VENV) __pycache__ src/*.egg-info *.egg-info .eggs
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
