#!/usr/bin/env python3
"""Generate experiment configuration files for the HydraDB retrieval quality study.

Experiments:
  1. Baseline: full_recall, alpha=0.8, thinking, graph=true, K=10
  2. Alpha sweep: 0.0, 0.3, 0.5, 0.7, 0.8, 1.0
  3. Graph context: on vs off
  4. Mode: fast vs thinking
  5. K sweep: 3, 5, 10, 15, 20
  6. Endpoint: full_recall vs boolean_recall (or/and operators)
  7. Minimal latency: fast mode + no graph
  8. Endpoint: recall_preferences (user memory search)
"""

import copy
from pathlib import Path

import yaml

EXPERIMENTS_DIR = Path("config/experiments")

# Base config template (no supermemory section)
BASE = {
    "hydradb": {
        "base_url": "https://api.hydradb.com",
        "tenant_id": "my-knowledge",
        "sub_tenant_id": "legal-privacyqa",
        "timeout_seconds": 30,
        "polling_interval_seconds": 10,
        "max_polling_attempts": 60,
        "create_tenant_on_start": False,
    },
    "ingestion": {
        "documents_dir": "./data/privacy_qa",
        "file_extensions": [".txt", ".pdf", ".md", ".docx"],
        "verify_before_querying": True,
        "upload_delay_seconds": 5,
        "upload_concurrency": 1,
    },
    "evaluation": {
        "test_dataset_path": "./data/privacy_qa.json",
        "search_endpoint": "full_recall",
        "max_results": 10,
        "concurrent_requests": 3,
        "mode": "thinking",
        "alpha": 0.8,
        "graph_context": True,
        "recency_bias": 0.0,
        "boolean_search_mode": "sources",
    },
    "deepeval": {
        "model": "gpt-4o",
        "threshold": 0.5,
        "include_reason": True,
        "eval_concurrency": 3,
        "metric_timeout_seconds": 120,
        "generator_model": "gpt-4o",
        "generator_temperature": 0.0,
        "generator_max_tokens": 1024,
        "metrics": [
            "answer_accuracy",
            "contextual_precision",
            "contextual_recall",
            "contextual_relevancy",
            "faithfulness",
            "answer_relevancy",
        ],
    },
    "reporting": {
        "output_dir": "./reports/experiments",
        "formats": ["json", "html", "csv"],
        "include_per_sample": True,
    },
}


def make_config(name: str, overrides: dict) -> dict:
    """Deep-copy base and apply overrides."""
    cfg = copy.deepcopy(BASE)
    cfg["benchmark"] = {"name": name}
    for section, values in overrides.items():
        if section in cfg:
            cfg[section].update(values)
        else:
            cfg[section] = values
    return cfg


def main():
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    experiments = []

    # ── Experiment 1: Baseline ──────────────────────────────────────────────────
    experiments.append(
        (
            "01_baseline",
            "Baseline: full_recall alpha=0.8 thinking graph=true K=10",
            {},
        )
    )

    # ── Experiment 2: Alpha sweep ───────────────────────────────────────────────
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        experiments.append(
            (
                f"02_alpha_{alpha:.1f}".replace(".", "_"),
                f"Alpha Sweep: alpha={alpha:.1f}",
                {"evaluation": {"alpha": alpha}},
            )
        )

    # ── Experiment 3: Graph context off ─────────────────────────────────────────
    experiments.append(
        (
            "03_no_graph",
            "No Graph Context: full_recall alpha=0.8 thinking graph=false K=10",
            {"evaluation": {"graph_context": False}},
        )
    )

    # ── Experiment 4: Fast mode ─────────────────────────────────────────────────
    experiments.append(
        (
            "04_fast_mode",
            "Fast Mode: full_recall alpha=0.8 fast graph=true K=10",
            {"evaluation": {"mode": "fast"}},
        )
    )

    # ── Experiment 5: K sweep ───────────────────────────────────────────────────
    for k in [3, 5, 15, 20]:
        experiments.append(
            (
                f"05_k_{k}",
                f"K Sweep: max_results={k}",
                {"evaluation": {"max_results": k}},
            )
        )

    # ── Experiment 6: Boolean recall ────────────────────────────────────────────
    experiments.append(
        (
            "06_boolean_or",
            "Boolean Recall: operator=or",
            {"evaluation": {"search_endpoint": "boolean_recall", "boolean_operator": "or"}},
        )
    )
    experiments.append(
        (
            "06_boolean_and",
            "Boolean Recall: operator=and",
            {"evaluation": {"search_endpoint": "boolean_recall", "boolean_operator": "and"}},
        )
    )

    # ── Experiment 7: Fast mode + no graph (minimal latency) ────────────────────
    experiments.append(
        (
            "07_fast_no_graph",
            "Minimal Latency: fast mode, no graph, alpha=0.8, K=10",
            {"evaluation": {"mode": "fast", "graph_context": False}},
        )
    )

    # ── Experiment 8: Recall preferences endpoint ─────────────────────────────
    experiments.append(
        (
            "08_recall_preferences",
            "Recall Preferences: recall_preferences alpha=0.8 thinking graph=true K=10",
            {"evaluation": {"search_endpoint": "recall_preferences"}},
        )
    )

    # ── Generate files ──────────────────────────────────────────────────────────
    manifest = []
    for filename, name, overrides in experiments:
        cfg = make_config(name, overrides)
        path = EXPERIMENTS_DIR / f"{filename}.yaml"
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        manifest.append({"id": filename, "name": name, "config": str(path)})
        print(f"  Generated: {path}")

    # Write manifest for the runner
    manifest_path = EXPERIMENTS_DIR / "manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump({"experiments": manifest}, f, default_flow_style=False, sort_keys=False)
    print(f"\nManifest: {manifest_path}")
    print(f"Total experiments: {len(experiments)}")


if __name__ == "__main__":
    main()
