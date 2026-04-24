from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .client import HydraDBClient
from .models import HydraConfig, IngestionConfig

console = Console()

_DONE_STATUSES = {"completed", "success", "graph_creation", "done", "indexed"}
_ERROR_STATUSES = {"errored", "error", "failed"}


class DocumentIngester:
    def __init__(
        self,
        client: HydraDBClient,
        config: IngestionConfig,
        hydra_config: HydraConfig,
        sub_tenant_filter: str | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._hydra_config = hydra_config
        self._sub_tenant_filter = sub_tenant_filter

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    SOURCE_ID_MAP_PATH = Path("source_id_map.json")

    async def run(self) -> tuple[int, int, float]:
        """Upload and optionally verify all documents.

        Returns:
            (indexed_count, failed_count, elapsed_seconds)
        """
        start = time.monotonic()
        documents_dir = Path(self._config.documents_dir)

        if not documents_dir.exists():
            console.print(f"[yellow]Documents directory not found: {documents_dir}[/yellow]")
            return 0, 0, 0.0

        grouped = self._collect_files(documents_dir)
        total_files = sum(len(files) for files in grouped.values())
        if not total_files:
            console.print(f"[yellow]No matching files found in {documents_dir}[/yellow]")
            return 0, 0, 0.0

        console.print(f"[cyan]Found {total_files} file(s) across {len(grouped)} group(s) to upload[/cyan]")

        # Load existing map so re-runs merge rather than overwrite
        source_id_map: dict[str, dict[str, str]] = {}
        if self.SOURCE_ID_MAP_PATH.exists():
            source_id_map = json.loads(self.SOURCE_ID_MAP_PATH.read_text())

        total_indexed = 0
        total_failed = 0
        for sub_tenant, files in grouped.items():
            console.print(f"\n[bold cyan]sub_tenant_id={sub_tenant}[/bold cyan] — {len(files)} file(s)")
            source_ids, failed_count, doc_map = await self._upload_files(files, sub_tenant_id=sub_tenant)
            total_indexed += len(source_ids)
            total_failed += failed_count

            # Merge doc_id → source_id mapping for this sub_tenant
            source_id_map.setdefault(sub_tenant, {}).update(doc_map)

            if source_ids and self._config.verify_before_querying:
                await self._poll_until_ready(source_ids, sub_tenant_id=sub_tenant)

        self.SOURCE_ID_MAP_PATH.write_text(json.dumps(source_id_map, indent=2))
        console.print(f"[cyan]Source ID map saved → {self.SOURCE_ID_MAP_PATH}[/cyan]")

        elapsed = time.monotonic() - start
        return total_indexed, total_failed, elapsed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect_files(self, directory: Path) -> dict[str, list[Path]]:
        """Return files grouped by sub_tenant_id.

        Files directly in `directory` use the config sub_tenant_id.
        Files inside a subdirectory use that subdirectory's name as sub_tenant_id.
        """
        extensions = {ext.lower() for ext in self._config.file_extensions}
        grouped: dict[str, list[Path]] = {}
        for f in directory.rglob("*"):
            if not (f.is_file() and f.suffix.lower() in extensions):
                continue
            rel = f.relative_to(directory)
            sub_tenant = rel.parts[0] if len(rel.parts) > 1 else (self._hydra_config.sub_tenant_id or "")
            if self._sub_tenant_filter and sub_tenant != self._sub_tenant_filter:
                continue
            grouped.setdefault(sub_tenant, []).append(f)
        return grouped

    async def _upload_files(self, files: list[Path], sub_tenant_id: str | None = None) -> tuple[list[str], int, dict[str, str]]:
        source_ids: list[str] = []
        failed = 0
        doc_map: dict[str, str] = {}  # doc_id (filename stem) → source_id (UUID)
        concurrency = max(1, self._config.upload_concurrency)
        batch_delay = self._config.upload_delay_seconds

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Uploading documents… (concurrency={concurrency})", total=len(files))

            # Process files in batches of `concurrency`
            for batch_start in range(0, len(files), concurrency):
                batch = files[batch_start : batch_start + concurrency]
                batch_num = batch_start // concurrency + 1
                total_batches = (len(files) + concurrency - 1) // concurrency
                progress.update(
                    task,
                    description=f"Uploading batch {batch_num}/{total_batches} ({len(batch)} files)…",
                )

                async def _upload_one(file_path: Path) -> tuple[str | None, str | None]:
                    """Returns (source_id_or_None, error_or_None)."""
                    if file_path.stat().st_size == 0:
                        return None, "skipped (empty file)"
                    try:
                        result = await self._client.upload_knowledge(
                            file_path,
                            sub_tenant_id=sub_tenant_id,
                        )
                        sid = self._extract_source_id(result, file_path.name)
                        return sid, None
                    except Exception as exc:
                        return None, str(exc)

                results = await asyncio.gather(*[_upload_one(f) for f in batch])

                for file_path, (sid, err) in zip(batch, results, strict=True):
                    if err and err.startswith("skipped"):
                        console.print(f"  [yellow]–[/yellow] {file_path.name} — {err}")
                    elif err:
                        console.print(f"  [red]✗[/red] {file_path.name} — {err}")
                        failed += 1
                    elif sid:
                        source_ids.append(sid)
                        doc_map[file_path.stem] = sid
                        console.print(f"  [green]✓[/green] {file_path.name} → source_id={sid}")
                    else:
                        console.print(
                            f"  [yellow]?[/yellow] {file_path.name} — could not extract source_id; skipping verification"
                        )
                    progress.advance(task)

                # Small delay between batches to avoid flooding HydraDB's queue
                if batch_delay > 0 and batch_start + concurrency < len(files):
                    await asyncio.sleep(batch_delay)

        return source_ids, failed, doc_map

    @staticmethod
    def _extract_source_id(result: dict, filename: str) -> str | None:
        """Extract a source_id / file_id from an upload response."""
        results_list = result.get("results", [])
        if results_list and isinstance(results_list, list):
            first = results_list[0]
            for key in ("source_id", "file_id", "id"):
                val = first.get(key)
                if val:
                    return str(val)
        # Top-level fallbacks
        for key in ("file_id", "source_id", "id"):
            val = result.get(key)
            if val:
                return str(val)
        return None

    async def _poll_until_ready(self, source_ids: list[str], sub_tenant_id: str | None = None) -> None:
        interval = self._hydra_config.polling_interval_seconds
        max_attempts = self._hydra_config.max_polling_attempts

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Verifying document processing…", total=None)

            for attempt in range(1, max_attempts + 1):
                progress.update(
                    task,
                    description=f"Polling processing status… (attempt {attempt}/{max_attempts})",
                )
                try:
                    response = await self._client.verify_processing(source_ids, sub_tenant_id=sub_tenant_id)
                    statuses = response.get("statuses", [])

                    if not statuses:
                        # No status info yet — keep polling
                        await asyncio.sleep(interval)
                        continue

                    all_done = True
                    any_error = False
                    for entry in statuses:
                        status = entry.get("indexing_status", "").lower()
                        if status in _ERROR_STATUSES:
                            fid = entry.get("file_id", "?")
                            console.print(f"  [red]✗[/red] file_id={fid} errored during indexing")
                            any_error = True
                        elif status not in _DONE_STATUSES:
                            all_done = False

                    if all_done or any_error:
                        if all_done and not any_error:
                            console.print("[green]All documents indexed successfully.[/green]")
                        else:
                            console.print("[yellow]Some documents completed with errors; proceeding.[/yellow]")
                        return

                except Exception as exc:
                    console.print(f"  [yellow]Warning: verify_processing failed: {exc}[/yellow]")

                await asyncio.sleep(interval)

        console.print(
            f"[yellow]Warning: max polling attempts ({max_attempts}) reached. "
            "Proceeding without full verification.[/yellow]"
        )
