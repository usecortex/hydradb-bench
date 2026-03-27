"""Document upload and indexing orchestrator."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .client import HydraDBClient
from .models import HydraConfig, IngestionConfig, IngestionFileStatus, IngestionReport

console = Console()
logger = logging.getLogger(__name__)


def _is_processing_complete(verify_response: dict) -> bool:
    """Interpret verify_processing response to determine if indexing is done."""
    _DONE = ("complete", "completed", "done", "indexed", "success", "graph_creation")

    # HydraDB BatchProcessingStatus: {"statuses": [{indexing_status, ...}]}
    statuses = verify_response.get("statuses")
    if isinstance(statuses, list) and statuses:
        return all(
            s.get("indexing_status", "").lower() in _DONE
            for s in statuses
            if isinstance(s, dict)
        )

    # Fallback: top-level status string
    status = verify_response.get("status", "")
    if isinstance(status, str) and status:
        return status.lower() in _DONE

    all_processed = verify_response.get("all_processed")
    if isinstance(all_processed, bool):
        return all_processed

    # Fallback: other list shapes
    files = verify_response.get("files", verify_response.get("results", []))
    if isinstance(files, list) and files:
        return all(
            f.get("status", "").lower() in _DONE
            for f in files
            if isinstance(f, dict)
        )

    # If we get a 200 with no clear status field, assume done
    return True


class IngestionOrchestrator:
    """Uploads documents to HydraDB and waits for indexing to complete."""

    def __init__(
        self,
        client: HydraDBClient,
        config: IngestionConfig,
        hydra_config: HydraConfig,
    ) -> None:
        self.client = client
        self.config = config
        self.hydra_config = hydra_config

    async def run(self) -> IngestionReport:
        start = time.monotonic()
        report = IngestionReport()

        # Optionally create tenant first
        if self.hydra_config.create_tenant_on_start:
            try:
                await self.client.create_tenant()
            except Exception as e:
                logger.warning("Tenant creation failed (may already exist): %s", e)

        # Discover documents
        docs_dir = Path(self.config.documents_dir)
        if not docs_dir.exists():
            logger.warning("Documents directory does not exist: %s", docs_dir)
            report.elapsed_seconds = time.monotonic() - start
            return report

        files = [
            f for f in docs_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.config.file_extensions
        ]

        if not files:
            logger.warning("No documents found in %s with extensions %s",
                           docs_dir, self.config.file_extensions)
            report.elapsed_seconds = time.monotonic() - start
            return report

        console.print(f"[bold]Ingesting {len(files)} document(s)...[/bold]")

        # Upload each file
        uploaded_ids: list[str] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for file_path in files:
                status = IngestionFileStatus(file_path=str(file_path))
                task = progress.add_task(f"Uploading {file_path.name}", total=None)
                try:
                    result = await self.client.upload_knowledge(file_path)
                    results_list = result.get("results", [])
                    first = results_list[0] if results_list else {}
                    file_id = (
                        first.get("source_id")
                        or first.get("file_id")
                        or first.get("id")
                        or result.get("file_id")
                        or result.get("id")
                        or result.get("document_id")
                        or ""
                    )
                    status.file_id = str(file_id)
                    status.status = "uploaded"
                    if file_id:
                        uploaded_ids.append(str(file_id))
                    logger.info("Uploaded %s → file_id=%s", file_path.name, file_id)
                except Exception as e:
                    status.status = "failed"
                    status.error = str(e)
                    logger.error("Failed to upload %s: %s", file_path.name, e)
                finally:
                    progress.remove_task(task)
                report.files.append(status)
                if self.config.upload_delay_seconds > 0:
                    await asyncio.sleep(self.config.upload_delay_seconds)

        # Poll for indexing completion
        uploaded_count = sum(1 for s in report.files if s.status == "uploaded")
        if self.config.verify_before_querying and uploaded_ids:
            console.print("[bold]Waiting for indexing to complete...[/bold]")
            all_indexed = await self._poll_until_indexed(uploaded_ids)
            for status in report.files:
                if status.status == "uploaded" and status.file_id in uploaded_ids:
                    status.status = "indexed" if all_indexed else "indexing_timeout"
            report.all_indexed = all_indexed
        elif self.config.verify_before_querying and not uploaded_ids and uploaded_count:
            # API didn't return file IDs — fall back to a timed wait so indexing
            # has time to complete before queries run.
            wait_secs = self.hydra_config.polling_interval_seconds * 6
            console.print(
                f"[bold]No file IDs returned by API — waiting {wait_secs}s "
                f"for async indexing...[/bold]"
            )
            await asyncio.sleep(wait_secs)
            for status in report.files:
                if status.status == "uploaded":
                    status.status = "indexed"
            report.all_indexed = True
        elif not self.config.verify_before_querying:
            for status in report.files:
                if status.status == "uploaded":
                    status.status = "indexed"
            report.all_indexed = True

        report.elapsed_seconds = time.monotonic() - start
        return report

    async def _poll_until_indexed(self, file_ids: list[str]) -> bool:
        """Poll verify_processing until all files are indexed or timeout."""
        interval = self.hydra_config.polling_interval_seconds
        max_attempts = self.hydra_config.max_polling_attempts

        for attempt in range(1, max_attempts + 1):
            try:
                result = await self.client.verify_processing(file_ids)
                if _is_processing_complete(result):
                    logger.info("Indexing complete after %d poll(s).", attempt)
                    return True
                logger.debug("Attempt %d/%d: still indexing...", attempt, max_attempts)
            except Exception as e:
                logger.warning("verify_processing error (attempt %d): %s", attempt, e)

            if attempt < max_attempts:
                await asyncio.sleep(interval)

        logger.warning("Indexing did not complete after %d attempts.", max_attempts)
        return False
