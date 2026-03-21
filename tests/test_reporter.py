"""Unit tests for report generation."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from hydradb_bench.models import (
    BenchmarkResult,
    BenchmarkSample,
    HydraSearchResult,
    ReportingConfig,
    TestSample as BenchTestSample,
    TokenUsageResult,
)
from hydradb_bench.reporter import BenchmarkReporter, _compute_latency_stats


def _make_result(n_samples: int = 3, with_token_usage: bool = False) -> BenchmarkResult:
    samples = []
    per_sample = []
    for i in range(n_samples):
        ts = BenchTestSample(
            id=f"q{i+1:03d}",
            question=f"Question {i+1}?",
            reference_answer=f"Reference answer {i+1}.",
        )
        hs = HydraSearchResult(
            answer=f"Generated answer {i+1}.",
            retrieved_contexts=[f"Context {i+1} text."],
        )
        samples.append(BenchmarkSample(test_sample=ts, hydra_result=hs, latency_ms=100 + i * 50))
        per_sample.append({
            "sample_id": ts.id,
            "question": ts.question,
            "reference_answer": ts.reference_answer,
            "hydra_answer": hs.answer,
            "latency_ms": 100 + i * 50,
            "contexts_retrieved": 1,
            "faithfulness": round(0.9 - i * 0.1, 2),
            "response_relevancy": 0.85,
            "answer_completeness": float(i % 2),  # aspect critic (binary 0/1)
        })

    token_usage = (
        TokenUsageResult(input_tokens=1200, output_tokens=400, total_tokens=1600,
                         estimated_cost_usd=0.0004, model="gpt-4o-mini")
        if with_token_usage else TokenUsageResult()
    )

    return BenchmarkResult(
        run_id="test123",
        timestamp="2026-03-21T10:00:00Z",
        benchmark_name="Test Benchmark",
        config_snapshot={
            "hydradb": {"tenant_id": "test-tenant", "sub_tenant_id": "bench"},
            "evaluation": {"search_endpoint": "qna"},
        },
        samples=samples,
        ragas_scores={"faithfulness": 0.87, "response_relevancy": 0.85, "answer_completeness": 0.67},
        multi_turn_scores={"topic_adherence": 0.91},
        per_sample_scores=per_sample,
        latency_stats={"p50": 100.0, "p95": 200.0, "mean": 150.0, "min": 100.0, "max": 200.0},
        token_usage=token_usage,
        error_count=0,
        evaluated_count=n_samples,
    )


# ---------------------------------------------------------------------------
# Latency stats
# ---------------------------------------------------------------------------

class TestLatencyStats:
    def test_computes_stats(self):
        samples = []
        for ms in [100.0, 200.0, 300.0, 400.0, 500.0]:
            ts = BenchTestSample(id="x", question="Q", reference_answer="A")
            hs = HydraSearchResult(answer="a", retrieved_contexts=[])
            samples.append(BenchmarkSample(test_sample=ts, hydra_result=hs, latency_ms=ms))
        stats = _compute_latency_stats(samples)
        assert "p50" in stats
        assert "mean" in stats
        assert stats["min"] == 100.0
        assert stats["max"] == 500.0

    def test_empty_returns_empty(self):
        assert _compute_latency_stats([]) == {}

    def test_skips_errored_samples(self):
        ts = BenchTestSample(id="x", question="Q", reference_answer="A")
        errored = BenchmarkSample(test_sample=ts, error="timeout")
        valid_ts = BenchTestSample(id="y", question="Q2", reference_answer="A2")
        valid_hs = HydraSearchResult(answer="a", retrieved_contexts=[])
        valid = BenchmarkSample(test_sample=valid_ts, hydra_result=valid_hs, latency_ms=500.0)
        stats = _compute_latency_stats([errored, valid])
        assert stats["min"] == 500.0


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

class TestJSONReport:
    def test_writes_valid_json(self):
        result = _make_result(3)
        config = ReportingConfig(formats=["json"])
        reporter = BenchmarkReporter(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = reporter.generate(result, output_dir=tmpdir)
            assert len(paths) == 1
            assert paths[0].suffix == ".json"
            data = json.loads(paths[0].read_text())
            assert data["run_id"] == "test123"
            assert "ragas_scores" in data
            assert data["ragas_scores"]["faithfulness"] == pytest.approx(0.87)

    def test_includes_multi_turn_scores(self):
        result = _make_result(2)
        config = ReportingConfig(formats=["json"])
        reporter = BenchmarkReporter(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = reporter.generate(result, output_dir=tmpdir)
            data = json.loads(paths[0].read_text())
            assert "multi_turn_scores" in data
            assert data["multi_turn_scores"]["topic_adherence"] == pytest.approx(0.91)

    def test_includes_token_usage(self):
        result = _make_result(2, with_token_usage=True)
        config = ReportingConfig(formats=["json"])
        reporter = BenchmarkReporter(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = reporter.generate(result, output_dir=tmpdir)
            data = json.loads(paths[0].read_text())
            assert "token_usage" in data
            assert data["token_usage"]["total_tokens"] == 1600
            assert data["token_usage"]["model"] == "gpt-4o-mini"

    def test_includes_per_sample_scores(self):
        result = _make_result(3)
        config = ReportingConfig(formats=["json"], include_per_sample_scores=True)
        reporter = BenchmarkReporter(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = reporter.generate(result, output_dir=tmpdir)
            data = json.loads(paths[0].read_text())
            assert len(data["per_sample_scores"]) == 3

    def test_excludes_per_sample_when_disabled(self):
        result = _make_result(3)
        config = ReportingConfig(formats=["json"], include_per_sample_scores=False)
        reporter = BenchmarkReporter(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = reporter.generate(result, output_dir=tmpdir)
            data = json.loads(paths[0].read_text())
            assert "per_sample_scores" not in data

    def test_aspect_critic_scores_in_output(self):
        """AspectCritic scores (binary) should appear in per-sample output."""
        result = _make_result(3)
        config = ReportingConfig(formats=["json"], include_per_sample_scores=True)
        reporter = BenchmarkReporter(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = reporter.generate(result, output_dir=tmpdir)
            data = json.loads(paths[0].read_text())
            # answer_completeness is our simulated aspect critic
            assert "answer_completeness" in data["ragas_scores"]


# ---------------------------------------------------------------------------
# CSV report
# ---------------------------------------------------------------------------

class TestCSVReport:
    def test_writes_csv_with_all_metric_columns(self):
        result = _make_result(3)
        config = ReportingConfig(formats=["csv"])
        reporter = BenchmarkReporter(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = reporter.generate(result, output_dir=tmpdir)
            csv_path = next(p for p in paths if p.suffix == ".csv")
            content = csv_path.read_text()
            assert "faithfulness" in content
            assert "response_relevancy" in content
            assert "answer_completeness" in content


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

class TestHTMLReport:
    def test_writes_valid_html(self):
        result = _make_result(3)
        config = ReportingConfig(formats=["html"])
        reporter = BenchmarkReporter(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = reporter.generate(result, output_dir=tmpdir)
            html_path = next(p for p in paths if p.suffix == ".html")
            content = html_path.read_text()
            assert "<!DOCTYPE html>" in content
            assert "test123" in content
            assert "faithfulness" in content.lower()

    def test_html_shows_multi_turn_section(self):
        result = _make_result(2)
        config = ReportingConfig(formats=["html"])
        reporter = BenchmarkReporter(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = reporter.generate(result, output_dir=tmpdir)
            html_path = next(p for p in paths if p.suffix == ".html")
            content = html_path.read_text()
            assert "topic_adherence" in content.lower() or "multi" in content.lower()

    def test_html_shows_token_usage(self):
        result = _make_result(2, with_token_usage=True)
        config = ReportingConfig(formats=["html"])
        reporter = BenchmarkReporter(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = reporter.generate(result, output_dir=tmpdir)
            html_path = next(p for p in paths if p.suffix == ".html")
            content = html_path.read_text()
            assert "1,600" in content or "1600" in content  # total tokens

    def test_all_formats_generated(self):
        result = _make_result(2)
        config = ReportingConfig(formats=["json", "csv", "html"])
        reporter = BenchmarkReporter(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = reporter.generate(result, output_dir=tmpdir)
            suffixes = {p.suffix for p in paths}
            assert ".json" in suffixes
            assert ".csv" in suffixes
            assert ".html" in suffixes
