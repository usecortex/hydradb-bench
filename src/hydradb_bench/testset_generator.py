"""Auto-generate Q&A test pairs from documents using RAGAS TestsetGenerator."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rich.console import Console

from .models import RAGASConfig, TestSample, TestsetGenerationConfig

console = Console()
logger = logging.getLogger(__name__)


def _load_documents_langchain(documents_dir: str, extensions: list[str]) -> list:
    """Load documents from disk as LangChain Document objects."""
    from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader

    docs_path = Path(documents_dir)
    documents = []

    loader_map = {
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".pdf": PyPDFLoader,
    }

    for file_path in sorted(docs_path.iterdir()):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext not in extensions:
            continue

        loader_cls = loader_map.get(ext)
        if loader_cls is None:
            logger.warning("No loader for extension %s, skipping %s", ext, file_path.name)
            continue

        try:
            loader = loader_cls(str(file_path))
            file_docs = loader.load()
            for doc in file_docs:
                doc.metadata.setdefault("source", file_path.name)
            documents.extend(file_docs)
            logger.info("Loaded %d chunks from %s", len(file_docs), file_path.name)
        except Exception as e:
            logger.warning("Could not load %s: %s", file_path.name, e)

    return documents


def _testset_sample_to_bench_sample(sample: Any, idx: int) -> TestSample:
    """Convert a RAGAS TestsetSample to our TestSample format."""
    # RAGAS TestsetSample fields (0.4.x): user_input, reference, reference_contexts
    question = getattr(sample, "user_input", "") or ""
    reference = getattr(sample, "reference", "") or ""
    ref_contexts = getattr(sample, "reference_contexts", None) or []

    # Normalise: reference_contexts can be strings or objects
    context_strings: list[str] = []
    for ctx in ref_contexts:
        if isinstance(ctx, str):
            context_strings.append(ctx)
        elif hasattr(ctx, "page_content"):
            context_strings.append(ctx.page_content)
        else:
            context_strings.append(str(ctx))

    return TestSample(
        id=f"gen_{idx + 1:03d}",
        question=question,
        reference_answer=reference,
        reference_contexts=context_strings,
    )


class TestsetGeneratorWrapper:
    """
    Wraps RAGAS TestsetGenerator to auto-generate Q&A pairs from documents.

    Workflow:
        1. Load documents from the documents directory.
        2. Use TestsetGenerator.generate_with_langchain_docs() to create testset.
        3. Convert to our TestSample format and save to JSON.
    """

    def __init__(self, ragas_config: RAGASConfig, gen_config: TestsetGenerationConfig) -> None:
        self.ragas_config = ragas_config
        self.gen_config = gen_config
        self._llm = None
        self._embeddings = None

    def _get_llm(self):
        if self._llm is None:
            import os
            from langchain_openai import ChatOpenAI
            from ragas.llms import LangchainLLMWrapper
            llm = ChatOpenAI(
                model=self.ragas_config.llm_model,
                api_key=os.environ.get("OPENAI_API_KEY", ""),
            )
            self._llm = LangchainLLMWrapper(llm, bypass_n=True)
        return self._llm

    def _get_embeddings(self):
        if self._embeddings is None:
            import os
            from langchain_openai import OpenAIEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper
            self._embeddings = LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(
                    model=self.ragas_config.embeddings_model,
                    api_key=os.environ.get("OPENAI_API_KEY", ""),
                )
            )
        return self._embeddings

    def _build_query_distribution(self) -> list:
        """Build RAGAS QueryDistribution from our config dict."""
        from ragas.testset.synthesizers import (
            AbstractQuerySynthesizer,
            ComparativeAbstractQuerySynthesizer,
            SpecificQuerySynthesizer,
        )

        dist_cfg = self.gen_config.query_distribution
        # Map our config keys to RAGAS synthesizer classes
        synthesizer_map = {
            "simple":        (SpecificQuerySynthesizer, {"llm": self._get_llm()}),
            "multi_context": (AbstractQuerySynthesizer, {"llm": self._get_llm()}),
            "reasoning":     (ComparativeAbstractQuerySynthesizer, {"llm": self._get_llm()}),
        }

        distribution = []
        for key, weight in dist_cfg.items():
            entry = synthesizer_map.get(key)
            if entry is None:
                logger.warning("Unknown query type '%s', skipping.", key)
                continue
            cls, kwargs = entry
            try:
                distribution.append((cls(**kwargs), weight))
            except Exception as e:
                logger.warning("Could not create synthesizer '%s': %s", key, e)

        # Normalise weights to sum to 1.0
        total = sum(w for _, w in distribution)
        if total > 0:
            distribution = [(s, w / total) for s, w in distribution]

        return distribution

    def generate(
        self,
        documents_dir: str,
        extensions: list[str] | None = None,
        save: bool = True,
    ) -> list[TestSample]:
        """
        Generate test samples from documents.

        Args:
            documents_dir: Path to directory containing knowledge documents.
            extensions:    File extensions to include (default from config).
            save:          If True, write output to gen_config.output_path.

        Returns:
            List of TestSample objects.
        """
        import nest_asyncio
        nest_asyncio.apply()

        exts = extensions or [".txt", ".pdf", ".md"]
        console.print(f"[bold]Loading documents from {documents_dir}...[/bold]")
        documents = _load_documents_langchain(documents_dir, exts)

        if not documents:
            raise ValueError(f"No documents found in {documents_dir} with extensions {exts}")

        console.print(f"  Loaded [bold]{len(documents)}[/bold] document chunk(s).")
        console.print(
            f"[bold]Generating {self.gen_config.testset_size} test samples via RAGAS...[/bold]"
        )
        console.print("  [dim]This calls OpenAI — expect 1–3 minutes for 15 samples.[/dim]")

        from ragas.testset import TestsetGenerator

        generator = TestsetGenerator(
            llm=self._get_llm(),
            embedding_model=self._get_embeddings(),
        )

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            try:
                query_distribution = self._build_query_distribution()
                testset = generator.generate_with_langchain_docs(
                    documents=documents,
                    testset_size=self.gen_config.testset_size,
                    query_distribution=query_distribution,
                )
            except TypeError:
                # Fallback: some versions don't accept query_distribution here
                testset = generator.generate_with_langchain_docs(
                    documents=documents,
                    testset_size=self.gen_config.testset_size,
                )

        samples = [
            _testset_sample_to_bench_sample(s, i)
            for i, s in enumerate(testset.samples)
        ]

        # Filter out empty questions
        samples = [s for s in samples if s.question.strip()]
        console.print(f"  Generated [bold]{len(samples)}[/bold] test sample(s).")

        if save:
            self._save(samples)

        return samples

    def _save(self, samples: list[TestSample]) -> None:
        output_path = Path(self.gen_config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([s.model_dump() for s in samples], f, indent=2, ensure_ascii=False)
        console.print(f"  [blue]Testset saved to:[/blue] {output_path}")
        logger.info("Testset saved: %d samples → %s", len(samples), output_path)
