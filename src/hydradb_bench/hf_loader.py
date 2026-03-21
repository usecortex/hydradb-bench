"""
HuggingFace dataset loader for HydraDB Benchmark.

Supports two modes:

  CORPUS MODE  — dataset is a document collection (e.g. isaacus/legal-rag-bench).
                 Rows are treated as documents; text field is uploaded to HydraDB
                 and optionally used with TestsetGenerator to auto-create Q&A.

  QA MODE      — dataset has pre-made Q&A pairs (e.g. explodinggradients/amnesty_qa).
                 Rows are mapped to TestSample objects for RAGAS evaluation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rich.console import Console

from .models import TestSample

console = Console()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Auto-detection: column name candidates for each TestSample field
# ---------------------------------------------------------------------------

_QUESTION_CANDIDATES  = ["question", "query", "input", "prompt", "user_input", "q"]
_ANSWER_CANDIDATES    = ["answer", "reference", "ground_truth", "reference_answer",
                         "ideal_answer", "expected_answer", "label", "output", "a",
                         "correct_answer", "target"]
_CONTEXTS_CANDIDATES  = ["contexts", "context", "retrieved_contexts", "documents",
                         "passages", "source_documents", "chunks", "evidence"]
_TEXT_CANDIDATES      = ["text", "content", "body", "passage", "document", "chunk"]
_ID_CANDIDATES        = ["id", "idx", "sample_id", "question_id", "qid", "_id"]


def _detect_column(columns: list[str], candidates: list[str]) -> str | None:
    """Return the first column name that matches a candidate (case-insensitive)."""
    lower_cols = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    return None


def _to_string_list(value: Any) -> list[str]:
    """Normalise a context value to a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                # Some datasets store context as {"text": "...", "score": ...}
                for key in ("evidence_text", "text", "passage", "content", "chunk_content"):
                    if isinstance(item.get(key), str):
                        result.append(item[key])
                        break
                else:
                    result.append(str(item))
            else:
                result.append(str(item))
        return [s for s in result if s.strip()]
    return [str(value)]


# ---------------------------------------------------------------------------
# QA loader — for evaluation datasets with question/answer pairs
# ---------------------------------------------------------------------------

def load_qa_dataset(
    repo_id: str,
    split: str = "test",
    config_name: str | None = None,
    column_map: dict[str, str] | None = None,
    max_samples: int | None = None,
    id_prefix: str = "hf",
    save_path: str | None = None,
) -> list[TestSample]:
    """
    Load a HuggingFace Q&A dataset and convert to TestSample list.

    Args:
        repo_id:      HuggingFace dataset ID, e.g. "explodinggradients/amnesty_qa"
        split:        Dataset split to load ("test", "train", "validation", etc.)
        column_map:   Optional explicit mapping: {"question": "col_name", ...}
                      Keys: question, reference_answer, reference_contexts, id
        max_samples:  Limit number of samples loaded
        id_prefix:    Prefix for auto-generated IDs
        save_path:    If set, save resulting TestSamples as JSON

    Returns:
        List of TestSample objects ready for RAGAS evaluation
    """
    from datasets import load_dataset

    console.print(f"[bold]Loading QA dataset:[/bold] {repo_id} (split={split})")
    ds = load_dataset(repo_id, config_name, split=split) if config_name else load_dataset(repo_id, split=split)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    cols = ds.column_names
    console.print(f"  Columns: {cols}")
    console.print(f"  Rows: {len(ds):,}")

    # Resolve column names — explicit map takes priority, then auto-detect
    cm = column_map or {}
    q_col   = cm.get("question")        or _detect_column(cols, _QUESTION_CANDIDATES)
    a_col   = cm.get("reference_answer") or _detect_column(cols, _ANSWER_CANDIDATES)
    ctx_col = cm.get("reference_contexts") or _detect_column(cols, _CONTEXTS_CANDIDATES)
    id_col  = cm.get("id")              or _detect_column(cols, _ID_CANDIDATES)

    if not q_col:
        raise ValueError(
            f"Cannot find a question column in {cols}. "
            f"Set column_map.question explicitly in benchmark.yaml."
        )
    if not a_col:
        logger.warning(
            "No answer column found — reference_answer will be empty. "
            "Metrics requiring reference (context_recall, factual_correctness) will not score."
        )

    console.print(f"  Mapped: question={q_col!r}, answer={a_col!r}, "
                  f"contexts={ctx_col!r}, id={id_col!r}")

    samples: list[TestSample] = []
    for i, row in enumerate(ds):
        sample_id = str(row[id_col]) if id_col else f"{id_prefix}_{i+1:04d}"
        question  = str(row.get(q_col, "")).strip()
        reference = str(row.get(a_col, "")).strip() if a_col else ""
        contexts  = _to_string_list(row.get(ctx_col)) if ctx_col else []

        if not question:
            continue  # skip empty questions

        samples.append(TestSample(
            id=sample_id,
            question=question,
            reference_answer=reference,
            reference_contexts=contexts,
        ))

    console.print(f"  [green]Loaded {len(samples)} Q&A samples.[/green]")

    if save_path:
        _save_samples(samples, save_path)

    return samples


# ---------------------------------------------------------------------------
# Corpus loader — for document collections to ingest into HydraDB
# ---------------------------------------------------------------------------

def load_corpus_as_documents(
    repo_id: str,
    split: str = "test",
    text_column: str | None = None,
    title_column: str | None = None,
    max_docs: int | None = None,
    output_dir: str = "./data/hf_documents",
) -> list[Path]:
    """
    Download a HuggingFace corpus dataset and save each document as a .txt file
    so HydraDB's ingestion pipeline can upload them.

    Args:
        repo_id:      e.g. "isaacus/legal-rag-bench"
        split:        Dataset split
        text_column:  Column containing document text (auto-detected if None)
        title_column: Column containing document title (optional, used for filenames)
        max_docs:     Limit number of documents saved
        output_dir:   Directory to save .txt files

    Returns:
        List of saved file paths
    """
    from datasets import load_dataset

    console.print(f"[bold]Loading corpus dataset:[/bold] {repo_id} (split={split})")
    ds = load_dataset(repo_id, split=split)

    if max_docs:
        ds = ds.select(range(min(max_docs, len(ds))))

    cols = ds.column_names
    console.print(f"  Columns: {cols}")
    console.print(f"  Documents: {len(ds):,}")

    # Resolve columns
    t_col = text_column or _detect_column(cols, _TEXT_CANDIDATES)
    title_col = title_column or _detect_column(cols, ["title", "name", "heading", "subject"])
    id_col = _detect_column(cols, _ID_CANDIDATES)

    if not t_col:
        raise ValueError(
            f"Cannot find a text column in {cols}. "
            f"Set hf_corpus.text_column in benchmark.yaml."
        )

    console.print(f"  Using text column: {t_col!r}, title: {title_col!r}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for i, row in enumerate(ds):
        text = str(row.get(t_col, "")).strip()
        if not text:
            continue

        # Build filename from title or id or index
        if title_col and row.get(title_col):
            slug = _slugify(str(row[title_col]))[:60]
        elif id_col and row.get(id_col):
            slug = _slugify(str(row[id_col]))[:60]
        else:
            slug = f"doc_{i+1:04d}"

        file_path = out / f"{slug}.txt"

        # Add title as header if available
        content = ""
        if title_col and row.get(title_col):
            content = f"# {row[title_col]}\n\n"
        content += text

        file_path.write_text(content, encoding="utf-8")
        saved_paths.append(file_path)

    console.print(f"  [green]Saved {len(saved_paths)} document(s) to {out}/[/green]")
    return saved_paths


# ---------------------------------------------------------------------------
# Generic corpus extractor — works with any Q&A HF dataset that has context
# ---------------------------------------------------------------------------

def extract_qa_corpus(
    repo_id: str,
    split: str = "train",
    context_column: str | None = None,
    group_by_column: str | None = None,
    max_docs: int | None = None,
    output_dir: str = "./data/hf_corpus",
) -> list[Path]:
    """
    Extract context/evidence passages from any HuggingFace Q&A dataset and
    save them as .txt files so HydraDB can ingest them.

    Works with datasets like:
      - PatronusAI/financebench  (context_column="evidence", group_by_column="doc_name")
      - explodinggradients/amnesty_qa  (context_column="context")
      - Any dataset with a contexts / evidence / passages column

    Args:
        repo_id:         HuggingFace dataset ID
        split:           Dataset split to load
        context_column:  Column containing passages (auto-detected if None)
        group_by_column: Group passages by this column before saving
                         (e.g. "doc_name" merges all passages for each source doc
                         into one file). If None, one file is saved per row.
        max_docs:        Cap on number of output files (None = all)
        output_dir:      Directory to write .txt files

    Returns:
        List of saved file paths
    """
    from datasets import load_dataset

    console.print(f"[bold]Extracting corpus from:[/bold] {repo_id} (split={split})")
    ds = load_dataset(repo_id, split=split)
    cols = ds.column_names
    console.print(f"  Columns: {cols}")

    # Resolve context column
    ctx_col = context_column or _detect_column(cols, _CONTEXTS_CANDIDATES)
    if not ctx_col:
        raise ValueError(
            f"Cannot find a context column in {cols}. "
            f"Set --context-column explicitly."
        )
    console.print(f"  Context column: {ctx_col!r}  |  Group by: {group_by_column!r}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if group_by_column and group_by_column in cols:
        # Group all passages for the same source document into one file
        doc_texts: dict[str, list[str]] = {}
        for row in ds:
            group_key = str(row.get(group_by_column, "unknown"))
            passages = _to_string_list(row.get(ctx_col))
            doc_texts.setdefault(group_key, []).extend(passages)

        saved: list[Path] = []
        for group_key, passages in list(doc_texts.items())[:max_docs]:
            seen: set[str] = set()
            unique = [p for p in passages if not (p in seen or seen.add(p))]  # type: ignore[func-returns-value]
            slug = _slugify(group_key)[:60] or "doc"
            file_path = out / f"{slug}.txt"
            file_path.write_text(
                f"# {group_key}\n\n" + "\n\n---\n\n".join(unique),
                encoding="utf-8",
            )
            saved.append(file_path)
    else:
        # One file per row
        rows = list(ds)
        if max_docs:
            rows = rows[:max_docs]
        id_col = _detect_column(cols, _ID_CANDIDATES)
        saved = []
        for i, row in enumerate(rows):
            passages = _to_string_list(row.get(ctx_col))
            if not passages:
                continue
            name = str(row[id_col]) if id_col else f"doc_{i+1:04d}"
            slug = _slugify(name)[:60] or f"doc_{i+1:04d}"
            file_path = out / f"{slug}.txt"
            file_path.write_text("\n\n---\n\n".join(passages), encoding="utf-8")
            saved.append(file_path)

    console.print(f"  Saved [bold]{len(saved)}[/bold] document file(s) to {out}/")
    return saved


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    """Convert text to a safe filename."""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text.strip("_") or "doc"


def _save_samples(samples: list[TestSample], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump([s.model_dump() for s in samples], f, indent=2, ensure_ascii=False)
    console.print(f"  [blue]Saved to:[/blue] {p}")


def print_dataset_info(repo_id: str, split: str = "test") -> None:
    """Preview a HuggingFace dataset's schema and first row. Useful for debugging."""
    from datasets import load_dataset, get_dataset_split_names
    # Auto-fall-back if the requested split doesn't exist
    try:
        available = get_dataset_split_names(repo_id)
    except Exception:
        available = []
    if available and split not in available:
        console.print(
            f"  [yellow]Split '{split}' not found. "
            f"Available: {available}. Using '{available[0]}'.[/yellow]"
        )
        split = available[0]
    ds = load_dataset(repo_id, split=split)
    console.print(f"\n[bold]{repo_id}[/bold] ({split} split)")
    console.print(f"  Rows: {len(ds):,}")
    console.print(f"  Columns: {ds.column_names}")
    console.print("\n  [bold]First row preview:[/bold]")
    row = ds[0]
    for k, v in row.items():
        preview = str(v)[:150].replace("\n", " ")
        console.print(f"    [cyan]{k}[/cyan] ({type(v).__name__}): {preview}")
