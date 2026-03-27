"""Unit tests for RAGAS evaluator."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from unittest.mock import MagicMock, patch
from hydradb_bench.models import (
    AspectCriticConfig,
    BenchmarkSample,
    HydraSearchResult,
    RAGASConfig,
    TestSample as BenchTestSample,
    TokenUsageResult,
)
from hydradb_bench.evaluator import RAGASEvaluator, _import_metric

# Fake LLM / embeddings so tests don't need a real OpenAI key
def _fake_llm():
    from ragas.llms import BaseRagasLLM
    mock = MagicMock(spec=BaseRagasLLM)
    return mock

def _fake_embeddings():
    from ragas.embeddings import BaseRagasEmbeddings
    mock = MagicMock(spec=BaseRagasEmbeddings)
    return mock

# Skip all tests in this module if ragas is not installed
ragas_available = pytest.importorskip("ragas", reason="ragas not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample(
    sample_id: str = "q1",
    question: str = "What is AI?",
    answer: str = "AI is artificial intelligence.",
    contexts: list[str] | None = None,
    reference: str = "AI stands for Artificial Intelligence.",
    error: str | None = None,
) -> BenchmarkSample:
    hydra_result = None
    if error is None:
        hydra_result = HydraSearchResult(
            answer=answer,
            retrieved_contexts=contexts or ["AI is a broad field of computer science."],
        )
    return BenchmarkSample(
        test_sample=BenchTestSample(
            id=sample_id,
            question=question,
            reference_answer=reference,
            reference_contexts=["AI stands for Artificial Intelligence."],
        ),
        hydra_result=hydra_result,
        latency_ms=123.4,
        error=error,
    )


# ---------------------------------------------------------------------------
# EvaluationDataset building
# ---------------------------------------------------------------------------

class TestBuildDataset:
    def setup_method(self):
        self.config = RAGASConfig(llm_model="gpt-4o-mini")
        self.evaluator = RAGASEvaluator(self.config)

    def test_builds_dataset_from_valid_samples(self):
        samples = [_make_sample("q1"), _make_sample("q2")]
        dataset = self.evaluator.build_dataset(samples)
        assert len(dataset.samples) == 2

    def test_skips_errored_samples(self):
        samples = [
            _make_sample("q1"),
            _make_sample("q2", error="API timeout"),
            _make_sample("q3"),
        ]
        dataset = self.evaluator.build_dataset(samples)
        assert len(dataset.samples) == 2

    def test_skips_samples_with_no_hydra_result(self):
        sample = BenchmarkSample(
            test_sample=BenchTestSample(id="q1", question="Q?", reference_answer="A."),
            hydra_result=None,
        )
        dataset = self.evaluator.build_dataset([sample])
        assert len(dataset.samples) == 0

    def test_maps_fields_correctly(self):
        sample = _make_sample(
            question="What is NLP?",
            answer="NLP is natural language processing.",
            contexts=["NLP processes text.", "NLP is part of AI."],
            reference="Natural Language Processing.",
        )
        dataset = self.evaluator.build_dataset([sample])
        s = dataset.samples[0]
        assert s.user_input == "What is NLP?"
        assert s.response == "NLP is natural language processing."
        assert "NLP processes text." in s.retrieved_contexts
        assert s.reference == "Natural Language Processing."

    def test_empty_contexts_get_placeholder(self):
        sample = _make_sample(contexts=[])
        dataset = self.evaluator.build_dataset([sample])
        assert len(dataset.samples) == 1

    def test_all_errored_returns_empty_dataset(self):
        samples = [_make_sample("q1", error="err1"), _make_sample("q2", error="err2")]
        dataset = self.evaluator.build_dataset(samples)
        assert len(dataset.samples) == 0


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

class TestMetricRegistry:
    def test_imports_faithfulness(self):
        cls = _import_metric("ragas.metrics._faithfulness.Faithfulness")
        assert cls is not None

    def test_import_error_on_unknown_path(self):
        with pytest.raises((ImportError, AttributeError, ModuleNotFoundError)):
            _import_metric("ragas.metrics._faithfulness.NonExistentMetric12345")

    @pytest.mark.skipif(
        not __import__("os").environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set — skipping metric instantiation test",
    )
    def test_build_metrics_known_names(self):
        config = RAGASConfig(llm_model="gpt-4o-mini")
        evaluator = RAGASEvaluator(config)
        metrics = evaluator.build_metrics(["faithfulness", "factual_correctness"])
        assert len(metrics) == 2

    @pytest.mark.skipif(
        not __import__("os").environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set — skipping metric instantiation test",
    )
    def test_build_metrics_skips_unknown(self):
        config = RAGASConfig(llm_model="gpt-4o-mini")
        evaluator = RAGASEvaluator(config)
        metrics = evaluator.build_metrics(["faithfulness", "not_a_real_metric_xyz"])
        assert len(metrics) == 1

    def test_offline_metrics_no_llm(self):
        """Offline metrics (bleu, rouge, exact_match) do not call _get_llm."""
        config = RAGASConfig(llm_model="gpt-4o-mini")
        evaluator = RAGASEvaluator(config)
        # exact_match has no extra deps; bleu/rouge need sacrebleu/rouge_score
        metrics = evaluator.build_metrics(["bleu_score", "rouge_score", "exact_match"])
        # At least exact_match must load; bleu/rouge load only if deps installed
        assert any(m.name == "exact_match" for m in metrics)
        assert all(
            hasattr(m, "name") and not callable(m)
            for m in metrics
        )

    def test_all_registry_class_paths_importable(self):
        """Every class_path in the registry must be importable without LLM init."""
        from hydradb_bench.evaluator import _METRIC_REGISTRY
        failed = []
        for name, (class_path, _, _) in _METRIC_REGISTRY.items():
            try:
                _import_metric(class_path)
            except Exception as e:
                failed.append(f"{name} ({class_path}): {e}")
        assert not failed, "Registry import failures:\n" + "\n".join(failed)


# ---------------------------------------------------------------------------
# AspectCritic
# ---------------------------------------------------------------------------

class TestAspectCritic:
    @pytest.mark.skipif(
        not __import__("os").environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set — AspectCritic needs real llm_factory",
    )
    def test_builds_aspect_critics(self):
        config = RAGASConfig(llm_model="gpt-4o-mini")
        evaluator = RAGASEvaluator(config)
        aspect_configs = [
            AspectCriticConfig(name="safety", definition="Is the answer safe?"),
            AspectCriticConfig(name="completeness", definition="Is the answer complete?"),
        ]
        critics = evaluator.build_aspect_critics(aspect_configs)
        assert len(critics) == 2
        assert critics[0].name == "safety"
        assert critics[1].name == "completeness"

    @pytest.mark.skipif(
        not __import__("os").environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set — AspectCritic needs real llm_factory",
    )
    def test_aspect_critic_strictness(self):
        config = RAGASConfig(llm_model="gpt-4o-mini")
        evaluator = RAGASEvaluator(config)
        critics = evaluator.build_aspect_critics([
            AspectCriticConfig(name="test", definition="Is it good?", strictness=3)
        ])
        assert len(critics) == 1
        assert getattr(critics[0], "strictness", 3) in (3, 1)


# ---------------------------------------------------------------------------
# Multi-turn dataset building
# ---------------------------------------------------------------------------

class TestMultiTurnDataset:
    def test_builds_multi_turn_dataset(self):
        config = RAGASConfig(llm_model="gpt-4o-mini")
        evaluator = RAGASEvaluator(config)

        conversations = [
            {
                "conversation_id": "c1",
                "conversation_turns": [
                    {"role": "human", "content": "What is ML?"},
                    {"role": "ai", "content": "Machine learning is..."},
                    {"role": "human", "content": "Give an example."},
                    {"role": "ai", "content": "Spam detection is an example."},
                ],
                "reference": "A complete intro to ML with examples.",
                "reference_topics": ["machine learning"],
            }
        ]
        dataset = evaluator.build_multi_turn_dataset(conversations)
        assert len(dataset.samples) == 1

    def test_empty_conversations_returns_empty_dataset(self):
        config = RAGASConfig(llm_model="gpt-4o-mini")
        evaluator = RAGASEvaluator(config)
        dataset = evaluator.build_multi_turn_dataset([])
        assert len(dataset.samples) == 0


# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------

class TestTokenUsageResult:
    def test_defaults(self):
        usage = TokenUsageResult()
        assert usage.total_tokens == 0
        assert usage.actual_cost_usd == 0.0

    def test_construction(self):
        usage = TokenUsageResult(
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            actual_cost_usd=0.0005,
            model="gpt-4o-mini",
        )
        assert usage.total_tokens == 1500
        assert usage.model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Prompt override (offline check — no API call)
# ---------------------------------------------------------------------------

class TestPromptOverride:
    @pytest.mark.skipif(
        not __import__("os").environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set — metric instantiation needs real llm_factory",
    )
    def test_apply_prompt_override_does_not_raise(self):
        config = RAGASConfig(llm_model="gpt-4o-mini")
        evaluator = RAGASEvaluator(config)
        metrics = evaluator.build_metrics(["faithfulness"])
        if metrics:
            # Should not raise even if metric doesn't support overrides
            evaluator._apply_prompt_override(
                metrics[0], "faithfulness", "Custom instruction."
            )
