"""Unit tests for HydraDB client response normalization."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hydradb_bench.client import _extract_answer, _extract_contexts
from hydradb_bench.models import HydraConfig, HydraSearchResult


# ---------------------------------------------------------------------------
# Response normalization tests (no API calls needed)
# ---------------------------------------------------------------------------

class TestExtractAnswer:
    def test_extracts_answer_key(self):
        assert _extract_answer({"answer": "Paris"}) == "Paris"

    def test_extracts_response_key(self):
        assert _extract_answer({"response": "Paris"}) == "Paris"

    def test_answer_takes_priority(self):
        assert _extract_answer({"answer": "correct", "response": "wrong"}) == "correct"

    def test_strips_whitespace(self):
        assert _extract_answer({"answer": "  hello  "}) == "hello"

    def test_fallback_to_string(self):
        result = _extract_answer({"unknown_key": "data"})
        assert isinstance(result, str)
        assert len(result) > 0


class TestExtractContexts:
    def test_extracts_contexts_list_of_strings(self):
        data = {"contexts": ["chunk A", "chunk B"]}
        assert _extract_contexts(data) == ["chunk A", "chunk B"]

    def test_extracts_chunks_with_chunk_content(self):
        data = {
            "chunks": [
                {"chunk_content": "text one", "relevancy_score": 0.9},
                {"chunk_content": "text two", "relevancy_score": 0.8},
            ]
        }
        result = _extract_contexts(data)
        assert result == ["text one", "text two"]

    def test_extracts_content_key_from_dicts(self):
        data = {"sources": [{"content": "source text"}]}
        assert _extract_contexts(data) == ["source text"]

    def test_returns_empty_on_no_match(self):
        assert _extract_contexts({"no_context": "here"}) == []

    def test_filters_empty_strings(self):
        data = {"contexts": ["valid", "", "  ", "also valid"]}
        result = _extract_contexts(data)
        assert "valid" in result
        assert "also valid" in result
        # empty strings should be filtered
        assert all(c.strip() for c in result)

    def test_hydradb_full_recall_format(self):
        """Test the exact format shown in HydraDB quickstart docs."""
        data = {
            "chunks": [
                {
                    "chunk_uuid": "a1b2c3d4",
                    "id": "doc_12345",
                    "chunk_content": "The team discussed a tiered pricing model.",
                    "source_type": "file",
                    "relevancy_score": 0.92,
                }
            ],
            "graph_context": {}
        }
        result = _extract_contexts(data)
        assert result == ["The team discussed a tiered pricing model."]


class TestHydraSearchResult:
    def test_model_construction(self):
        result = HydraSearchResult(
            answer="Test answer",
            retrieved_contexts=["ctx 1", "ctx 2"],
        )
        assert result.answer == "Test answer"
        assert len(result.retrieved_contexts) == 2

    def test_defaults(self):
        result = HydraSearchResult(answer="a", retrieved_contexts=[])
        assert result.raw_response == {}


class TestHydraConfig:
    def test_required_fields(self):
        config = HydraConfig(
            api_key="test-key",
            tenant_id="my-tenant",
        )
        assert config.base_url == "https://api.hydradb.com"
        assert config.timeout_seconds == 30

    def test_sub_tenant_optional(self):
        config = HydraConfig(api_key="k", tenant_id="t")
        assert config.sub_tenant_id == ""
