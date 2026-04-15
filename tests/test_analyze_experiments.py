"""Unit tests for analyze_experiments.py."""

import json

from analyze_experiments import (
    METRICS,
    _extract_experiment_id,
    analyze_hypothesis,
    build_comparison_table,
    compute_aggregate,
    export_csv,
    extract_metric_scores,
    format_comparison_markdown,
    generate_decision_framework,
    load_experiment_results,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sample(scores: dict) -> dict:
    """Create a per_sample entry matching BenchmarkResult format."""
    return {"sample_id": "s1", "question": "q", "answer": "a", "scores": scores}


def _make_result(
    name: str = "Test",
    run_id: str = "abcd1234_01_baseline",
    samples: list[dict] | None = None,
    experiment_id: str | None = None,
) -> dict:
    """Create a result dict matching BenchmarkResult JSON format."""
    result = {
        "name": name,
        "run_id": run_id,
        "per_sample": samples or [],
        "aggregate_scores": {},
    }
    if experiment_id:
        result["experiment_id"] = experiment_id
    return result


SAMPLE_SCORES_A = {
    "answer_accuracy": 0.8,
    "contextual_precision": 0.7,
    "contextual_recall": 0.9,
    "contextual_relevancy": 0.6,
    "faithfulness": 0.85,
    "answer_relevancy": 0.75,
}

SAMPLE_SCORES_B = {
    "answer_accuracy": 0.9,
    "contextual_precision": 0.8,
    "contextual_recall": 0.85,
    "contextual_relevancy": 0.7,
    "faithfulness": 0.9,
    "answer_relevancy": 0.8,
}


# ---------------------------------------------------------------------------
# _extract_experiment_id
# ---------------------------------------------------------------------------


class TestExtractExperimentId:
    """Tests for experiment ID extraction from result data."""

    def test_explicit_experiment_id_field(self):
        data = {"experiment_id": "01_baseline", "name": "whatever"}
        assert _extract_experiment_id(data, "anything") == "01_baseline"

    def test_brackets_in_name(self):
        data = {"name": "Experiment: Baseline [01_baseline]", "run_id": ""}
        assert _extract_experiment_id(data, "") == "01_baseline"

    def test_brackets_alpha_sweep(self):
        data = {"name": "Alpha Sweep [02_alpha_0_5]", "run_id": ""}
        assert _extract_experiment_id(data, "") == "02_alpha_0_5"

    def test_run_id_with_hex_prefix(self):
        data = {"name": "No brackets", "run_id": "deadbeef_02_alpha_0_5"}
        assert _extract_experiment_id(data, "") == "02_alpha_0_5"

    def test_filename_stem_with_hex_prefix(self):
        data = {"name": "", "run_id": ""}
        assert _extract_experiment_id(data, "cafe1234_05_k_15") == "05_k_15"

    def test_filename_stem_is_experiment_id(self):
        data = {"name": "", "run_id": ""}
        assert _extract_experiment_id(data, "03_no_graph") == "03_no_graph"

    def test_non_hex_prefix_falls_through(self):
        """Filenames like 'fast_mode_test' should NOT match hex prefix pattern."""
        data = {"name": "", "run_id": ""}
        assert _extract_experiment_id(data, "fast_mode_test") == "fast_mode_test"

    def test_legacy_report_falls_through(self):
        """Legacy reports like 'c4fa3bb6_HydraDB' should not match experiment pattern."""
        data = {
            "name": "HydraDB DeepEval Benchmark [HydraDB]",
            "run_id": "c4fa3bb6_HydraDB",
        }
        result = _extract_experiment_id(data, "c4fa3bb6_HydraDB")
        assert result == "c4fa3bb6_HydraDB"

    def test_short_hex_prefix_rejected(self):
        """Hex prefixes shorter than 6 chars should not match."""
        data = {"name": "", "run_id": "abc_01_baseline"}
        # 'abc' is only 3 hex chars, should not match the 6-8 char pattern
        # Falls through to filename stem
        result = _extract_experiment_id(data, "abc_01_baseline")
        assert result == "abc_01_baseline"

    def test_priority_order(self):
        """experiment_id field takes priority over brackets and run_id."""
        data = {
            "experiment_id": "from_field",
            "name": "Test [from_brackets]",
            "run_id": "abcd1234_from_runid",
        }
        assert _extract_experiment_id(data, "from_stem") == "from_field"


# ---------------------------------------------------------------------------
# extract_metric_scores
# ---------------------------------------------------------------------------


class TestExtractMetricScores:
    """Tests for metric score extraction from result data."""

    def test_per_sample_format(self):
        result = _make_result(
            samples=[
                _make_sample(SAMPLE_SCORES_A),
                _make_sample(SAMPLE_SCORES_B),
            ]
        )
        scores = extract_metric_scores(result)
        assert scores["answer_accuracy"] == [0.8, 0.9]
        assert scores["faithfulness"] == [0.85, 0.9]
        assert len(scores["contextual_recall"]) == 2

    def test_none_scores_skipped(self):
        result = _make_result(
            samples=[
                _make_sample({"answer_accuracy": None, "contextual_precision": 0.5}),
            ]
        )
        scores = extract_metric_scores(result)
        assert scores["answer_accuracy"] == []
        assert scores["contextual_precision"] == [0.5]

    def test_missing_metrics_empty(self):
        result = _make_result(samples=[_make_sample({"answer_accuracy": 0.8})])
        scores = extract_metric_scores(result)
        assert scores["answer_accuracy"] == [0.8]
        assert scores["faithfulness"] == []

    def test_aggregate_scores_fallback(self):
        result = {
            "aggregate_scores": {"answer_accuracy": 0.85, "faithfulness": 0.9},
        }
        scores = extract_metric_scores(result)
        assert scores["answer_accuracy"] == [0.85]
        assert scores["faithfulness"] == [0.9]
        assert scores["contextual_precision"] == []

    def test_empty_result(self):
        scores = extract_metric_scores({})
        for metric in METRICS:
            assert scores[metric] == []

    def test_empty_per_sample(self):
        result = _make_result(samples=[])
        scores = extract_metric_scores(result)
        for metric in METRICS:
            assert scores[metric] == []


# ---------------------------------------------------------------------------
# compute_aggregate
# ---------------------------------------------------------------------------


class TestComputeAggregate:
    """Tests for aggregate statistics computation."""

    def test_normal_scores(self):
        agg = compute_aggregate([0.8, 0.9, 0.7])
        assert abs(agg["mean"] - 0.8) < 0.001
        assert agg["n"] == 3
        assert agg["min"] == 0.7
        assert agg["max"] == 0.9
        assert agg["std"] > 0

    def test_single_score(self):
        agg = compute_aggregate([0.5])
        assert agg["mean"] == 0.5
        assert agg["std"] == 0.0
        assert agg["n"] == 1

    def test_empty_scores(self):
        agg = compute_aggregate([])
        assert agg["mean"] is None
        assert agg["std"] is None
        assert agg["n"] == 0

    def test_identical_scores(self):
        agg = compute_aggregate([0.8, 0.8, 0.8])
        assert agg["mean"] == 0.8
        assert agg["std"] == 0.0


# ---------------------------------------------------------------------------
# build_comparison_table
# ---------------------------------------------------------------------------


class TestBuildComparisonTable:
    """Tests for comparison table construction."""

    def test_single_experiment(self):
        results = {
            "01_baseline": _make_result(
                samples=[
                    _make_sample(SAMPLE_SCORES_A),
                    _make_sample(SAMPLE_SCORES_B),
                ]
            ),
        }
        table = build_comparison_table(results)
        assert "01_baseline" in table
        assert abs(table["01_baseline"]["answer_accuracy"]["mean"] - 0.85) < 0.001

    def test_multiple_experiments(self):
        results = {
            "01_baseline": _make_result(samples=[_make_sample(SAMPLE_SCORES_A)]),
            "03_no_graph": _make_result(samples=[_make_sample(SAMPLE_SCORES_B)]),
        }
        table = build_comparison_table(results)
        assert len(table) == 2
        assert table["01_baseline"]["answer_accuracy"]["mean"] == 0.8
        assert table["03_no_graph"]["answer_accuracy"]["mean"] == 0.9


# ---------------------------------------------------------------------------
# analyze_hypothesis
# ---------------------------------------------------------------------------


class TestAnalyzeHypothesis:
    """Tests for hypothesis analysis."""

    def test_significant_difference(self):
        table = {
            "01_baseline": {m: {"mean": 0.8, "std": 0.1, "n": 10} for m in METRICS},
            "03_no_graph": {m: {"mean": 0.5, "std": 0.1, "n": 10} for m in METRICS},
        }
        group = {
            "ids": ["01_baseline", "03_no_graph"],
            "variable": "graph_context",
            "values": [True, False],
            "hypothesis": "H2",
        }
        analysis = analyze_hypothesis("graph_context", group, table)
        assert "SUPPORTED" in analysis["conclusion"]
        assert len(analysis["experiments_found"]) == 2

    def test_no_significant_difference(self):
        table = {
            "01_baseline": {m: {"mean": 0.8, "std": 0.1, "n": 10} for m in METRICS},
            "04_fast_mode": {m: {"mean": 0.78, "std": 0.1, "n": 10} for m in METRICS},
        }
        group = {
            "ids": ["01_baseline", "04_fast_mode"],
            "variable": "mode",
            "values": ["thinking", "fast"],
            "hypothesis": "H3",
        }
        analysis = analyze_hypothesis("mode", group, table)
        assert "NOT SUPPORTED" in analysis["conclusion"]

    def test_insufficient_data(self):
        table = {"01_baseline": {m: {"mean": 0.8} for m in METRICS}}
        group = {
            "ids": ["01_baseline", "missing_exp"],
            "variable": "test",
            "values": [1, 2],
            "hypothesis": "H0",
        }
        analysis = analyze_hypothesis("test", group, table)
        assert "INSUFFICIENT DATA" in analysis["conclusion"]
        assert "missing_exp" in analysis["experiments_missing"]


# ---------------------------------------------------------------------------
# generate_decision_framework
# ---------------------------------------------------------------------------


class TestGenerateDecisionFramework:
    """Tests for decision framework generation."""

    def test_different_metrics_pick_different_configs(self):
        """Verify lambda late-binding fix: each metric picks the correct best config."""
        table = {
            "01_baseline": {
                "answer_accuracy": {"mean": 0.9},
                "contextual_precision": {"mean": 0.8},
                "contextual_recall": {"mean": 0.7},
                "contextual_relevancy": {"mean": 0.6},
                "faithfulness": {"mean": 0.5},
                "answer_relevancy": {"mean": 0.7},
            },
            "03_no_graph": {
                "answer_accuracy": {"mean": 0.6},
                "contextual_precision": {"mean": 0.5},
                "contextual_recall": {"mean": 0.9},
                "contextual_relevancy": {"mean": 0.8},
                "faithfulness": {"mean": 0.95},
                "answer_relevancy": {"mean": 0.6},
            },
        }
        framework = generate_decision_framework(table)
        # Maximum accuracy -> 01_baseline (0.9 > 0.6)
        assert "01_baseline" in framework
        # Trustworthy answers -> 03_no_graph (0.95 > 0.5)
        assert "03_no_graph" in framework
        # Verify specific use case recommendations
        lines = framework.split("\n")
        accuracy_line = [line for line in lines if "Maximum accuracy" in line]
        assert accuracy_line and "01_baseline" in accuracy_line[0]
        faith_line = [line for line in lines if "Trustworthy answers" in line]
        assert faith_line and "03_no_graph" in faith_line[0]

    def test_empty_table(self):
        framework = generate_decision_framework({})
        assert "Decision Framework" in framework


# ---------------------------------------------------------------------------
# format_comparison_markdown
# ---------------------------------------------------------------------------


class TestFormatComparisonMarkdown:
    """Tests for markdown table formatting."""

    def test_basic_formatting(self):
        table = {
            "01_baseline": {m: {"mean": 0.8} for m in METRICS},
        }
        md = format_comparison_markdown(table)
        assert "01_baseline" in md
        assert "0.800" in md
        assert "| Experiment |" in md

    def test_none_values_show_dash(self):
        table = {
            "01_baseline": {m: {"mean": None} for m in METRICS},
        }
        md = format_comparison_markdown(table)
        assert "- |" in md


# ---------------------------------------------------------------------------
# export_csv
# ---------------------------------------------------------------------------


class TestExportCsv:
    """Tests for CSV export."""

    def test_csv_output(self, tmp_path):
        table = {
            "01_baseline": {m: {"mean": 0.8} for m in METRICS},
            "03_no_graph": {m: {"mean": 0.6} for m in METRICS},
        }
        csv_path = tmp_path / "results.csv"
        export_csv(table, csv_path)
        content = csv_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 3  # header + 2 rows
        assert "experiment" in lines[0]
        assert "01_baseline" in lines[1]


# ---------------------------------------------------------------------------
# load_experiment_results
# ---------------------------------------------------------------------------


class TestLoadExperimentResults:
    """Tests for loading results from JSON files."""

    def test_loads_json_files(self, tmp_path):
        result = _make_result(
            name="Test [01_baseline]",
            run_id="abcd1234_01_baseline",
            samples=[_make_sample(SAMPLE_SCORES_A)],
        )
        (tmp_path / "abcd1234_01_baseline.json").write_text(json.dumps(result))
        results = load_experiment_results(tmp_path)
        assert "01_baseline" in results

    def test_skips_non_json(self, tmp_path):
        (tmp_path / "notes.txt").write_text("not json")
        results = load_experiment_results(tmp_path)
        assert len(results) == 0

    def test_skips_special_files(self, tmp_path):
        (tmp_path / "all_experiments.json").write_text("{}")
        (tmp_path / "experiment_log.jsonl").write_text("{}")
        results = load_experiment_results(tmp_path)
        assert len(results) == 0

    def test_handles_invalid_json(self, tmp_path):
        (tmp_path / "bad.json").write_text("not valid json{{{")
        results = load_experiment_results(tmp_path)
        assert len(results) == 0

    def test_duplicate_experiment_id_keeps_newest(self, tmp_path):
        result1 = _make_result(name="Test [01_baseline]")
        result2 = _make_result(name="Test [01_baseline]")
        result2["timestamp"] = "2099-01-01T00:00:00+00:00"
        file1 = tmp_path / "run1_01_baseline.json"
        file2 = tmp_path / "run2_01_baseline.json"
        file1.write_text(json.dumps(result1))
        file2.write_text(json.dumps(result2))
        results = load_experiment_results(tmp_path)
        assert len(results) == 1
        # The result with the newer timestamp should be kept
        assert results["01_baseline"].get("timestamp") == "2099-01-01T00:00:00+00:00"
