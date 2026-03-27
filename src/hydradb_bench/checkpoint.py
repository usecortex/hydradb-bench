"""Checkpoint manager — saves query results per sample so a crash can be resumed."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .models import BenchmarkSample, HydraSearchResult, TestSample

logger = logging.getLogger(__name__)

_CHECKPOINT_DIR = Path("./checkpoints")


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


class CheckpointManager:
    """
    Appends one JSON line per completed sample to a .jsonl file.
    On resume, already-completed samples are skipped and restored from disk.
    """

    def __init__(self, run_id: str, dataset_name: str) -> None:
        _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.path = _CHECKPOINT_DIR / f"{run_id}_{_slug(dataset_name)}.jsonl"
        self._done: dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    self._done[entry["sample_id"]] = entry
                except Exception:
                    pass
        if self._done:
            logger.info(
                "Checkpoint: %d completed sample(s) loaded from %s",
                len(self._done), self.path,
            )

    # ------------------------------------------------------------------
    def is_done(self, sample_id: str) -> bool:
        return sample_id in self._done

    @property
    def completed_count(self) -> int:
        return len(self._done)

    # ------------------------------------------------------------------
    def save(self, sample: BenchmarkSample) -> None:
        """Append one completed sample to the checkpoint file."""
        entry: dict = {
            "sample_id": sample.test_sample.id,
            "question": sample.test_sample.question,
            "reference_answer": sample.test_sample.reference_answer,
            "reference_contexts": sample.test_sample.reference_contexts or [],
            "hydra_answer": sample.hydra_result.answer if sample.hydra_result else "",
            "retrieved_contexts": (
                sample.hydra_result.retrieved_contexts if sample.hydra_result else []
            ),
            "latency_ms": sample.latency_ms,
            "error": sample.error,
        }
        self._done[sample.test_sample.id] = entry
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    def restore(self, test_samples: list[TestSample]) -> list[BenchmarkSample]:
        """Reconstruct BenchmarkSample objects for already-completed samples."""
        restored: list[BenchmarkSample] = []
        for ts in test_samples:
            if ts.id not in self._done:
                continue
            entry = self._done[ts.id]
            if entry.get("error"):
                restored.append(BenchmarkSample(
                    test_sample=ts,
                    error=entry["error"],
                    latency_ms=entry.get("latency_ms", 0.0),
                ))
            else:
                restored.append(BenchmarkSample(
                    test_sample=ts,
                    hydra_result=HydraSearchResult(
                        answer=entry["hydra_answer"],
                        retrieved_contexts=entry["retrieved_contexts"],
                        network_latency_ms=entry.get("latency_ms", 0.0),
                    ),
                    latency_ms=entry.get("latency_ms", 0.0),
                ))
        return restored

    # ------------------------------------------------------------------
    def delete(self) -> None:
        """Remove checkpoint after successful run completion."""
        if self.path.exists():
            self.path.unlink()
            logger.info("Checkpoint deleted: %s", self.path)
