"""Unit tests for hf_loader — column detection, normalisation, and path building."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from hydradb_bench.hf_loader import _detect_column, _slugify, _to_string_list


# ---------------------------------------------------------------------------
# _detect_column
# ---------------------------------------------------------------------------

class TestDetectColumn:
    def test_exact_match(self):
        assert _detect_column(["question", "answer"], ["question"]) == "question"

    def test_case_insensitive(self):
        assert _detect_column(["Question", "Answer"], ["question"]) == "Question"

    def test_returns_first_candidate_match(self):
        # "query" is second candidate for questions; "question" not in cols
        assert _detect_column(["query", "answer"], ["question", "query"]) == "query"

    def test_returns_none_when_no_match(self):
        assert _detect_column(["foo", "bar"], ["question", "query"]) is None

    def test_empty_columns(self):
        assert _detect_column([], ["question"]) is None

    def test_preserves_original_case_in_return(self):
        # Returns the original column name, not the lowercased candidate
        result = _detect_column(["Ground_Truth"], ["ground_truth"])
        assert result == "Ground_Truth"


# ---------------------------------------------------------------------------
# _to_string_list
# ---------------------------------------------------------------------------

class TestToStringList:
    def test_none_returns_empty(self):
        assert _to_string_list(None) == []

    def test_empty_string_returns_empty(self):
        assert _to_string_list("   ") == []

    def test_non_empty_string_wrapped(self):
        assert _to_string_list("hello") == ["hello"]

    def test_list_of_strings_passthrough(self):
        assert _to_string_list(["a", "b", "c"]) == ["a", "b", "c"]

    def test_filters_blank_strings_from_list(self):
        assert _to_string_list(["a", "  ", "b"]) == ["a", "b"]

    def test_dict_items_extract_text_key(self):
        items = [{"text": "passage one", "score": 0.9}]
        assert _to_string_list(items) == ["passage one"]

    def test_dict_items_extract_passage_key(self):
        items = [{"passage": "passage text"}]
        assert _to_string_list(items) == ["passage text"]

    def test_dict_items_extract_content_key(self):
        items = [{"content": "the content"}]
        assert _to_string_list(items) == ["the content"]

    def test_dict_items_extract_chunk_content_key(self):
        items = [{"chunk_content": "chunk text"}]
        assert _to_string_list(items) == ["chunk text"]

    def test_dict_fallback_to_str(self):
        items = [{"unknown_key": "value"}]
        result = _to_string_list(items)
        assert len(result) == 1
        assert "unknown_key" in result[0]

    def test_mixed_list(self):
        items = ["plain", {"text": "from dict"}, 42]
        result = _to_string_list(items)
        assert "plain" in result
        assert "from dict" in result
        assert "42" in result

    def test_scalar_wrapped(self):
        assert _to_string_list(123) == ["123"]


# ---------------------------------------------------------------------------
# _slugify
# ---------------------------------------------------------------------------

class TestSlugify:
    def test_basic(self):
        assert _slugify("Hello World") == "hello_world"

    def test_special_chars_removed(self):
        assert _slugify("doc/path:file") == "docpathfile"

    def test_multiple_spaces_collapsed(self):
        assert _slugify("a   b") == "a_b"

    def test_leading_trailing_underscores_stripped(self):
        result = _slugify("  hello  ")
        assert not result.startswith("_")
        assert not result.endswith("_")

    def test_empty_string_returns_doc(self):
        assert _slugify("") == "doc"

    def test_only_special_chars_returns_doc(self):
        assert _slugify("!!!") == "doc"


# ---------------------------------------------------------------------------
# load_qa_dataset (integration — skipped without `datasets` library)
# ---------------------------------------------------------------------------

datasets_available = pytest.importorskip("datasets", reason="datasets not installed")


class TestLoadQaDataset:
    """
    Integration tests that call HuggingFace APIs.
    Skipped if the `datasets` library is not installed.
    These use a tiny public dataset to avoid slow downloads.
    """

    @pytest.mark.skipif(
        not __import__("os").environ.get("HF_INTEGRATION_TESTS"),
        reason="Set HF_INTEGRATION_TESTS=1 to run HuggingFace integration tests",
    )
    def test_loads_amnesty_qa(self):
        from hydradb_bench.hf_loader import load_qa_dataset
        samples = load_qa_dataset(
            repo_id="explodinggradients/amnesty_qa",
            split="eval",
            max_samples=3,
        )
        assert len(samples) == 3
        assert all(s.question for s in samples)
        assert all(s.id.startswith("hf_") for s in samples)

    @pytest.mark.skipif(
        not __import__("os").environ.get("HF_INTEGRATION_TESTS"),
        reason="Set HF_INTEGRATION_TESTS=1 to run HuggingFace integration tests",
    )
    def test_explicit_column_map(self):
        from hydradb_bench.hf_loader import load_qa_dataset
        # amnesty_qa has "question" and "answer" columns
        samples = load_qa_dataset(
            repo_id="explodinggradients/amnesty_qa",
            split="eval",
            column_map={"question": "question", "reference_answer": "answer"},
            max_samples=2,
        )
        assert len(samples) == 2

    @pytest.mark.skipif(
        not __import__("os").environ.get("HF_INTEGRATION_TESTS"),
        reason="Set HF_INTEGRATION_TESTS=1 to run HuggingFace integration tests",
    )
    def test_saves_json(self, tmp_path):
        from hydradb_bench.hf_loader import load_qa_dataset
        save_path = str(tmp_path / "qa_out.json")
        load_qa_dataset(
            repo_id="explodinggradients/amnesty_qa",
            split="eval",
            max_samples=2,
            save_path=save_path,
        )
        import json
        data = json.loads(Path(save_path).read_text())
        assert len(data) == 2
        assert "question" in data[0]


class TestLoadCorpus:
    @pytest.mark.skipif(
        not __import__("os").environ.get("HF_INTEGRATION_TESTS"),
        reason="Set HF_INTEGRATION_TESTS=1 to run HuggingFace integration tests",
    )
    def test_loads_legal_rag_bench_corpus(self, tmp_path):
        from hydradb_bench.hf_loader import load_corpus_as_documents
        paths = load_corpus_as_documents(
            repo_id="isaacus/legal-rag-bench",
            split="train",
            max_docs=3,
            output_dir=str(tmp_path),
        )
        assert len(paths) == 3
        for p in paths:
            assert p.exists()
            assert p.suffix == ".txt"
            assert p.read_text(encoding="utf-8").strip()


class TestExtractQaCorpus:
    @pytest.mark.skipif(
        not __import__("os").environ.get("HF_INTEGRATION_TESTS"),
        reason="Set HF_INTEGRATION_TESTS=1 to run HuggingFace integration tests",
    )
    def test_one_file_per_row(self, tmp_path):
        """Without group_by, each row becomes its own .txt file."""
        from hydradb_bench.hf_loader import extract_qa_corpus
        paths = extract_qa_corpus(
            repo_id="explodinggradients/amnesty_qa",
            split="eval",
            max_docs=3,
            output_dir=str(tmp_path),
        )
        assert len(paths) == 3
        for p in paths:
            assert p.suffix == ".txt"
            assert p.read_text(encoding="utf-8").strip()

    @pytest.mark.skipif(
        not __import__("os").environ.get("HF_INTEGRATION_TESTS"),
        reason="Set HF_INTEGRATION_TESTS=1 to run HuggingFace integration tests",
    )
    def test_group_by_merges_passages(self, tmp_path):
        """With group_by=doc_name, passages for the same doc end up in one file."""
        from hydradb_bench.hf_loader import extract_qa_corpus
        paths = extract_qa_corpus(
            repo_id="PatronusAI/financebench",
            split="train",
            context_column="evidence",
            group_by_column="doc_name",
            max_docs=5,
            output_dir=str(tmp_path),
        )
        assert 1 <= len(paths) <= 5
        for p in paths:
            assert p.suffix == ".txt"
            content = p.read_text(encoding="utf-8")
            assert content.strip()

    @pytest.mark.skipif(
        not __import__("os").environ.get("HF_INTEGRATION_TESTS"),
        reason="Set HF_INTEGRATION_TESTS=1 to run HuggingFace integration tests",
    )
    def test_explicit_context_column(self, tmp_path):
        """Explicit context_column bypasses auto-detection."""
        from hydradb_bench.hf_loader import extract_qa_corpus
        paths = extract_qa_corpus(
            repo_id="explodinggradients/amnesty_qa",
            split="eval",
            context_column="context",
            max_docs=2,
            output_dir=str(tmp_path),
        )
        assert len(paths) == 2
