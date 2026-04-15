#!/usr/bin/env python3
"""
Cross-experiment analysis for HydraDB retrieval quality R&D study.

Reads experiment results from reports/experiments/ and produces:
1. Comparison tables (per-metric, per-experiment)
2. Hypothesis validation summaries
3. Decision framework recommendations
4. Visualization-ready CSV output
5. Final markdown research report

Usage:
    python analyze_experiments.py [--results-dir reports/experiments] [--output-dir reports/analysis]
"""

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

METRICS = [
    "answer_accuracy",
    "contextual_precision",
    "contextual_recall",
    "contextual_relevancy",
    "faithfulness",
    "answer_relevancy",
]

# Experiment metadata for grouping and analysis
EXPERIMENT_GROUPS = {
    "alpha_sweep": {
        "ids": [
            "02_alpha_0_0",
            "02_alpha_0_3",
            "02_alpha_0_5",
            "02_alpha_0_7",
            "01_baseline",  # alpha=0.8
            "02_alpha_1_0",
        ],
        "variable": "alpha",
        "values": [0.0, 0.3, 0.5, 0.7, 0.8, 1.0],
        "hypothesis": "H1",
    },
    "graph_context": {
        "ids": ["01_baseline", "03_no_graph"],
        "variable": "graph_context",
        "values": [True, False],
        "hypothesis": "H2",
    },
    "mode": {
        "ids": ["01_baseline", "04_fast_mode"],
        "variable": "mode",
        "values": ["thinking", "fast"],
        "hypothesis": "H3",
    },
    "k_sweep": {
        "ids": ["05_k_3", "05_k_5", "01_baseline", "05_k_15", "05_k_20"],
        "variable": "max_results",
        "values": [3, 5, 10, 15, 20],
        "hypothesis": "H4",
    },
    "endpoint": {
        "ids": ["01_baseline", "08_recall_preferences", "06_boolean_or", "06_boolean_and"],
        "variable": "search_endpoint",
        "values": ["full_recall", "recall_preferences", "boolean_recall (or)", "boolean_recall (and)"],
        "hypothesis": "H5",
    },
    "minimal_latency": {
        "ids": ["01_baseline", "07_fast_no_graph"],
        "variable": "combined",
        "values": ["baseline", "fast+no_graph"],
        "hypothesis": "H6",
    },
}


def _extract_experiment_id(data: dict, filename_stem: str) -> str:
    """Extract experiment ID from result data or filename.

    run_experiments.py passes provider_name=exp_id to the reporter, which produces:
    - name: "Experiment: ... [01_baseline]"  (exp_id in brackets)
    - run_id: "{uuid}_{exp_id}"  (exp_id as suffix)
    - filename: "{run_id}.json" -> stem = "{uuid}_{exp_id}"

    We check these sources in order of reliability.
    """
    # 0. Check for explicit experiment_id field (set by run_experiments.py summary)
    if "experiment_id" in data:
        return data["experiment_id"]

    name = data.get("name", "")

    # 1. Check for experiment ID in brackets at end of name: "... [01_baseline]"
    bracket_match = re.search(r"\[([^\]]+)\]$", name.strip())
    if bracket_match:
        candidate = bracket_match.group(1)
        # Validate it looks like an experiment ID (starts with digit or known prefix)
        if re.match(r"^\d{2}_", candidate):
            return candidate

    # 2. Check run_id suffix: "{uuid[:8]}_{exp_id}"
    run_id = data.get("run_id", "")
    # run_id format: "abcd1234_01_baseline" -> extract everything after 6-8 hex char prefix
    run_match = re.match(r"^[a-f0-9]{6,8}_(.+)$", run_id)
    if run_match:
        candidate = run_match.group(1)
        if re.match(r"^\d{2}_", candidate):
            return candidate

    # 3. Check filename stem with same pattern
    stem_match = re.match(r"^[a-f0-9]{6,8}_(.+)$", filename_stem)
    if stem_match:
        candidate = stem_match.group(1)
        if re.match(r"^\d{2}_", candidate):
            return candidate

    # 4. Check if filename stem itself is an experiment ID
    if re.match(r"^\d{2}_", filename_stem):
        return filename_stem

    # Last resort: use filename stem as-is
    return filename_stem


def _get_result_timestamp(data: dict, json_file: Path) -> float:
    """Extract a comparable timestamp from a result, falling back to file mtime."""
    # Prefer embedded timestamp field
    ts = data.get("timestamp")
    if isinstance(ts, str) and ts:
        try:
            from datetime import datetime

            return datetime.fromisoformat(ts).timestamp()
        except (ValueError, TypeError):
            pass
    # Fall back to file modification time
    return json_file.stat().st_mtime


def load_experiment_results(results_dir: Path) -> dict:
    """Load all experiment result JSON files from the results directory.

    When multiple files map to the same experiment_id, the newest result
    (by embedded ``timestamp`` or file mtime) is kept.
    """
    results: dict[str, dict] = {}
    result_files: dict[str, Path] = {}
    for json_file in sorted(results_dir.glob("*.json")):
        if json_file.name in ("all_experiments.json", "experiment_log.jsonl"):
            continue
        try:
            data = json.loads(json_file.read_text())
            exp_id = _extract_experiment_id(data, json_file.stem)
            if exp_id in results:
                existing_ts = _get_result_timestamp(results[exp_id], result_files[exp_id])
                new_ts = _get_result_timestamp(data, json_file)
                if new_ts > existing_ts:
                    print(
                        f"  Warning: Duplicate experiment ID '{exp_id}' from {json_file.name}, "
                        f"replacing older result from {result_files[exp_id].name}"
                    )
                    results[exp_id] = data
                    result_files[exp_id] = json_file
                else:
                    print(
                        f"  Warning: Duplicate experiment ID '{exp_id}' from {json_file.name}, "
                        f"keeping newer result from {result_files[exp_id].name}"
                    )
                continue
            results[exp_id] = data
            result_files[exp_id] = json_file
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: Could not load {json_file.name}: {e}")
    return results


def extract_metric_scores(result: dict) -> dict[str, list[float]]:
    """Extract per-sample scores for each metric from a result.

    Handles the actual BenchmarkResult JSON format from reporter.py:
    - per_sample: list of dicts with "scores" sub-dict (primary format)
    - aggregate_scores: dict of metric -> float (fallback if per_sample missing)
    """
    scores = {m: [] for m in METRICS}

    # Primary format: BenchmarkResult.per_sample (from reporter.py / models.py)
    # Each sample has: {sample_id, question, answer, ..., scores: {metric: float}, ...}
    if "per_sample" in result:
        for sample in result["per_sample"]:
            sample_scores = sample.get("scores", {})
            for metric in METRICS:
                if metric in sample_scores:
                    val = sample_scores[metric]
                    if val is not None:
                        scores[metric].append(float(val))
    elif "aggregate_scores" in result:
        # Fallback: only aggregate scores available (no per-sample breakdown)
        for metric in METRICS:
            if metric in result["aggregate_scores"]:
                val = result["aggregate_scores"][metric]
                if val is not None:
                    scores[metric] = [float(val)]

    return scores


def compute_aggregate(scores: list[float]) -> dict:
    """Compute mean, std, min, max for a list of scores."""
    if not scores:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": statistics.mean(scores),
        "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "min": min(scores),
        "max": max(scores),
        "n": len(scores),
    }


def compute_average_scores(table: dict) -> dict[str, float]:
    """Compute per-experiment average score across all metrics.

    Returns a dict mapping experiment ID to its mean score across all
    metrics that have a non-None mean.
    """
    avg_scores: dict[str, float] = {}
    for exp_id, metrics in table.items():
        means = [metrics[m]["mean"] for m in METRICS if metrics[m]["mean"] is not None]
        if means:
            avg_scores[exp_id] = statistics.mean(means)
    return avg_scores


def build_comparison_table(results: dict) -> dict:
    """Build a comparison table: experiment -> metric -> aggregate stats."""
    table = {}
    for exp_id, result in results.items():
        scores = extract_metric_scores(result)
        table[exp_id] = {}
        for metric in METRICS:
            table[exp_id][metric] = compute_aggregate(scores[metric])
    return table


def format_comparison_markdown(table: dict, title: str = "All Experiments") -> str:
    """Format comparison table as markdown."""
    lines = [f"## {title}\n"]

    # Header
    header = "| Experiment |"
    separator = "|---|"
    for m in METRICS:
        short = m.replace("contextual_", "ctx_").replace("answer_", "ans_")
        header += f" {short} |"
        separator += "---|"
    header += " Avg |"
    separator += "---|"
    lines.append(header)
    lines.append(separator)

    # Rows
    for exp_id in sorted(table.keys()):
        row = f"| {exp_id} |"
        means = []
        for metric in METRICS:
            stats = table[exp_id][metric]
            if stats["mean"] is not None:
                row += f" {stats['mean']:.3f} |"
                means.append(stats["mean"])
            else:
                row += " - |"
        avg = statistics.mean(means) if means else None
        row += f" {avg:.3f} |" if avg is not None else " - |"
        lines.append(row)

    return "\n".join(lines)


def analyze_hypothesis(group_name: str, group_info: dict, table: dict) -> dict:
    """Analyze a specific hypothesis group."""
    analysis = {
        "hypothesis": group_info["hypothesis"],
        "variable": group_info["variable"],
        "experiments_found": [],
        "experiments_missing": [],
        "metric_comparisons": {},
        "conclusion": "",
    }

    # Check which experiments have results
    for exp_id in group_info["ids"]:
        if exp_id in table:
            analysis["experiments_found"].append(exp_id)
        else:
            analysis["experiments_missing"].append(exp_id)

    if len(analysis["experiments_found"]) < 2:
        analysis["conclusion"] = "INSUFFICIENT DATA: Need at least 2 experiments to compare."
        return analysis

    # Compare metrics across experiments in this group
    for metric in METRICS:
        metric_data = {}
        for exp_id in analysis["experiments_found"]:
            stats = table[exp_id][metric]
            if stats["mean"] is not None:
                metric_data[exp_id] = stats

        if len(metric_data) >= 2:
            best_exp = max(metric_data, key=lambda x: metric_data[x]["mean"])
            worst_exp = min(metric_data, key=lambda x: metric_data[x]["mean"])
            spread = metric_data[best_exp]["mean"] - metric_data[worst_exp]["mean"]

            analysis["metric_comparisons"][metric] = {
                "best": best_exp,
                "best_score": metric_data[best_exp]["mean"],
                "worst": worst_exp,
                "worst_score": metric_data[worst_exp]["mean"],
                "spread": spread,
                "significant": spread > 0.1,  # >0.1 is meaningful given variance
            }

    # Generate conclusion
    significant_metrics = [m for m, data in analysis["metric_comparisons"].items() if data["significant"]]

    if significant_metrics:
        best_configs = {}
        for m in significant_metrics:
            best = analysis["metric_comparisons"][m]["best"]
            best_configs[best] = best_configs.get(best, 0) + 1
        winner = max(best_configs, key=best_configs.get)
        analysis["conclusion"] = (
            f"SUPPORTED: {winner} is best across {best_configs[winner]} "
            f"significant metrics ({', '.join(significant_metrics)})"
        )
    else:
        analysis["conclusion"] = (
            "NOT SUPPORTED: No metric showed >0.1 spread across configurations. Differences are within noise margin."
        )

    return analysis


def format_hypothesis_markdown(group_name: str, analysis: dict) -> str:
    """Format hypothesis analysis as markdown."""
    lines = [
        f"### {analysis['hypothesis']}: {group_name.replace('_', ' ').title()}",
        f"**Variable**: `{analysis['variable']}`",
        f"**Experiments found**: {len(analysis['experiments_found'])} / "
        f"{len(analysis['experiments_found']) + len(analysis['experiments_missing'])}",
        "",
    ]

    if analysis["experiments_missing"]:
        lines.append(f"**Missing**: {', '.join(analysis['experiments_missing'])}")
        lines.append("")

    lines.append(f"**Conclusion**: {analysis['conclusion']}")
    lines.append("")

    if analysis["metric_comparisons"]:
        lines.append("| Metric | Best Config | Score | Worst Config | Score | Spread | Significant? |")
        lines.append("|---|---|---|---|---|---|---|")
        for metric, data in analysis["metric_comparisons"].items():
            sig = "YES" if data["significant"] else "no"
            lines.append(
                f"| {metric} | {data['best']} | {data['best_score']:.3f} | "
                f"{data['worst']} | {data['worst_score']:.3f} | "
                f"{data['spread']:.3f} | {sig} |"
            )
        lines.append("")

    return "\n".join(lines)


def generate_decision_framework(table: dict) -> str:
    """Generate a decision framework based on experiment results."""
    lines = [
        "## Decision Framework\n",
        "Based on the experimental results, here are recommended configurations for different use cases:\n",
    ]

    # Find overall best config
    if table:
        avg_scores = compute_average_scores(table)

        if avg_scores:
            best_overall = max(avg_scores, key=avg_scores.get)
            lines.append(f"### Best Overall Quality: `{best_overall}` (avg: {avg_scores[best_overall]:.3f})\n")

            # Recommendations by use case
            lines.append("### Use Case Recommendations\n")
            lines.append("| Use Case | Recommended Config | Key Metric | Score |")
            lines.append("|---|---|---|---|")

            use_cases = {
                "Maximum accuracy": "answer_accuracy",
                "Trustworthy answers": "faithfulness",
                "Complete retrieval": "contextual_recall",
                "Relevant context": "contextual_relevancy",
            }

            for use_case, metric in use_cases.items():
                best = max(
                    table.keys(),
                    key=lambda x, m=metric: table[x][m]["mean"] if table[x][m]["mean"] is not None else -1,
                )
                score = table[best][metric]["mean"]
                if score is not None:
                    lines.append(f"| {use_case} | `{best}` | {metric} | {score:.3f} |")

    return "\n".join(lines)


def export_csv(table: dict, output_path: Path):
    """Export comparison table as CSV for further analysis."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["experiment"] + METRICS + ["average"]
        writer.writerow(header)

        for exp_id in sorted(table.keys()):
            row = [exp_id]
            means = []
            for metric in METRICS:
                val = table[exp_id][metric]["mean"]
                row.append(f"{val:.4f}" if val is not None else "")
                if val is not None:
                    means.append(val)
            avg = statistics.mean(means) if means else None
            row.append(f"{avg:.4f}" if avg is not None else "")
            writer.writerow(row)


def generate_full_report(table: dict, analyses: dict, output_path: Path):
    """Generate the full markdown research report."""
    sections = []

    # Title
    sections.append("# HydraDB Retrieval Quality R&D Study: Results Report\n")
    sections.append(
        f"**Experiments analyzed**: {len(table)}\n"
        f"**Metrics evaluated**: {', '.join(METRICS)}\n"
        f"**Judge model**: gpt-4o\n"
    )

    # Executive Summary
    sections.append("## Executive Summary\n")
    if table:
        avg_scores = compute_average_scores(table)

        if avg_scores:
            best = max(avg_scores, key=avg_scores.get)
            worst = min(avg_scores, key=avg_scores.get)
            sections.append(
                f"- **Best configuration**: `{best}` (avg score: {avg_scores[best]:.3f})\n"
                f"- **Worst configuration**: `{worst}` (avg score: {avg_scores[worst]:.3f})\n"
                f"- **Score range**: {avg_scores[worst]:.3f} - {avg_scores[best]:.3f} "
                f"(spread: {avg_scores[best] - avg_scores[worst]:.3f})\n"
            )

    # Comparison Table
    sections.append(format_comparison_markdown(table))
    sections.append("")

    # Hypothesis Results
    sections.append("## Hypothesis Validation\n")
    for group_name, analysis in analyses.items():
        sections.append(format_hypothesis_markdown(group_name, analysis))

    # Decision Framework
    sections.append(generate_decision_framework(table))

    # Methodology
    sections.append("\n## Methodology\n")
    sections.append(
        "- **Dataset**: privacy_qa (30 samples, 7 source documents)\n"
        "- **Evaluation**: DeepEval framework with GPT-4o as judge\n"
        "- **Metrics**: 6 RAG quality metrics (answer accuracy, contextual precision/recall/relevancy, "
        "faithfulness, answer relevancy)\n"
        "- **Significance threshold**: >0.1 spread (accounts for +/-0.1-0.15 LLM judge variance)\n"
        "- **Configurations**: 16 experiment variants testing alpha, graph context, mode, K, "
        "search endpoint, and combined settings\n"
    )

    report = "\n".join(sections)
    output_path.write_text(report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze HydraDB experiment results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("reports/experiments"),
        help="Directory containing experiment result JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/analysis"),
        help="Directory for analysis output",
    )
    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.results_dir}...")
    results = load_experiment_results(args.results_dir)

    if not results:
        print("No experiment results found. Run experiments first.")
        print(f"  Expected JSON files in: {args.results_dir}/")
        sys.exit(1)

    print(f"  Loaded {len(results)} experiment results")

    # Build comparison table
    table = build_comparison_table(results)

    # Analyze hypotheses
    analyses = {}
    for group_name, group_info in EXPERIMENT_GROUPS.items():
        analyses[group_name] = analyze_hypothesis(group_name, group_info, table)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate outputs
    print("\nGenerating analysis outputs...")

    # 1. Comparison table markdown
    comparison_md = format_comparison_markdown(table)
    (args.output_dir / "comparison_table.md").write_text(comparison_md)
    print(f"  Written: {args.output_dir}/comparison_table.md")

    # 2. Hypothesis analysis
    hypothesis_lines = ["# Hypothesis Validation Results\n"]
    for group_name, analysis in analyses.items():
        hypothesis_lines.append(format_hypothesis_markdown(group_name, analysis))
    (args.output_dir / "hypothesis_results.md").write_text("\n".join(hypothesis_lines))
    print(f"  Written: {args.output_dir}/hypothesis_results.md")

    # 3. Decision framework
    framework = generate_decision_framework(table)
    (args.output_dir / "decision_framework.md").write_text(framework)
    print(f"  Written: {args.output_dir}/decision_framework.md")

    # 4. CSV export
    export_csv(table, args.output_dir / "results.csv")
    print(f"  Written: {args.output_dir}/results.csv")

    # 5. Full report
    generate_full_report(table, analyses, args.output_dir / "full_report.md")
    print(f"  Written: {args.output_dir}/full_report.md")

    # 6. Raw analysis data (JSON)
    raw_data = {
        "comparison_table": table,
        "hypothesis_analyses": analyses,
    }
    (args.output_dir / "raw_analysis.json").write_text(json.dumps(raw_data, indent=2, default=str))
    print(f"  Written: {args.output_dir}/raw_analysis.json")

    # Print summary to stdout
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(comparison_md)
    print()
    for _group_name, analysis in analyses.items():
        print(f"{analysis['hypothesis']}: {analysis['conclusion']}")
    print()


if __name__ == "__main__":
    main()
