"""Async HTTP client for the Supermemory API (api.supermemory.ai).

Endpoints used:
  POST /v3/documents/file        — upload any file (TXT, MD, PDF, DOCX, …)
  GET  /v3/documents/processing  — poll indexing status
  POST /v3/documents/list        — list stored documents
  POST /v3/search                — hybrid semantic + keyword search
  DELETE /v3/container-tags/{t}  — wipe all docs in a container (reset)

Ingestion strategy
------------------
All file types (including .txt and .md) are uploaded via POST /v3/documents/file
(multipart/form-data) rather than reading content into memory and sending it as
a JSON string via the batch endpoint. Reasons:
  - Text files in this benchmark are large (10K+ lines); stuffing them into a
    JSON string field risks hitting undocumented per-document content-size limits.
  - /v3/documents/file streams bytes directly and supports up to 50 MB per file.
  - Supported formats: TXT, MD, PDF, DOC, DOCX, CSV, JPG, PNG, MP4, and more.
  - Supermemory handles chunking and embedding internally for all formats.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .models import IngestionConfig, SupermemoryConfig

console = Console()


class SupermemoryClient:
    """Thin async wrapper around the Supermemory v3 REST API."""

    def __init__(self, config: SupermemoryConfig) -> None:
        self._cfg = config
        self._http: httpx.AsyncClient | None = None

    async def __aenter__(self) -> SupermemoryClient:
        self._http = httpx.AsyncClient(
            base_url=self._cfg.base_url,
            headers={
                "Authorization": f"Bearer {self._cfg.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self._cfg.timeout_seconds,
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._http:
            await self._http.aclose()

    @property
    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            raise RuntimeError("SupermemoryClient must be used as an async context manager.")
        return self._http

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code >= 400:
            raise RuntimeError(f"Supermemory API error {response.status_code}: {response.text}")

    # ── Document management ──────────────────────────────────────────────────

    async def upload_file(
        self,
        file_path: Path,
        container_tag: str,
    ) -> dict[str, Any]:
        """POST /v3/documents/file — upload a binary file (PDF, DOCX, etc.).

        Uses a separate httpx client because multipart/form-data must NOT have
        a pre-set Content-Type header (httpx sets the boundary automatically).

        Response: { "id": str, "status": str }
        """
        async with httpx.AsyncClient(
            base_url=self._cfg.base_url,
            headers={"Authorization": f"Bearer {self._cfg.api_key}"},
            timeout=self._cfg.timeout_seconds,
        ) as mc:
            with file_path.open("rb") as fh:
                r = await mc.post(
                    "/v3/documents/file",
                    files={"file": (file_path.name, fh, "application/octet-stream")},
                    data={"containerTag": container_tag},
                )
            self._raise_for_status(r)
            return r.json()

    async def get_processing_documents(self) -> dict[str, Any]:
        """GET /v3/documents/processing — returns documents still being indexed.

        Response: { "documents": [...], "totalCount": int }
        Poll until totalCount == 0.
        """
        r = await self._client.get("/v3/documents/processing")
        self._raise_for_status(r)
        return r.json()

    async def list_documents(self, container_tags: list[str]) -> dict[str, Any]:
        """POST /v3/documents/list — list stored documents filtered by container tags."""
        r = await self._client.post(
            "/v3/documents/list",
            json={"containerTags": container_tags},
        )
        self._raise_for_status(r)
        return r.json()

    async def delete_container_tag(self, container_tag: str) -> dict[str, Any]:
        """DELETE /v3/container-tags/{containerTag} — cascade-delete all docs in container.

        Response: { "success": bool, "containerTag": str,
                    "deletedDocumentsCount": int, "deletedMemoriesCount": int }
        """
        r = await self._client.delete(f"/v3/container-tags/{container_tag}")
        self._raise_for_status(r)
        return r.json()

    # ── Search ───────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        container_tag: str | None = None,
        limit: int = 10,
        search_mode: str = "hybrid",
        rerank: bool = False,
        threshold: float = 0.0,
    ) -> dict[str, Any]:
        """POST /v3/search — hybrid semantic + keyword search.

        Args:
            query:        Natural-language search string.
            container_tag: Scope results to this container (maps to HydraDB tenant).
            limit:        Max number of results to return.
            search_mode:  "hybrid" (default) or "memories".
            rerank:       Re-score results for higher precision (~100 ms overhead).
            threshold:    Minimum similarity score (0–1); 0 = return everything.

        Actual response (verified via live API call):
        {
          "results": [
            {
              "documentId": str,
              "title": str,
              "score": float,       # top chunk score for this document
              "type": str,          # e.g. "text"
              "metadata": dict,
              "createdAt": str,
              "updatedAt": str,
              "chunks": [           # ← nested array of matched chunks
                {
                  "content": str,   # ← the raw chunk text
                  "position": int,  # chunk position within the document
                  "isRelevant": bool,
                  "score": float
                }
              ]
            }
          ],
          "timing": int,   # query time in ms
          "total": int     # total number of matching chunks
        }
        Extract text with: [c["content"] for r in results for c in r["chunks"]]
        """
        payload: dict[str, Any] = {
            "q": query,
            "limit": limit,
            "searchMode": search_mode,
            "rerank": rerank,
            "threshold": threshold,
        }
        if container_tag:
            payload["containerTag"] = container_tag
        r = await self._client.post("/v3/search", json=payload)
        self._raise_for_status(r)
        return r.json()


# ── Ingester ─────────────────────────────────────────────────────────────────


class SupermemoryIngester:
    """Uploads all documents to Supermemory via POST /v3/documents/file and polls until indexed.

    Every file type (TXT, MD, PDF, DOCX, …) is uploaded as a multipart file —
    no file content is ever read into memory as a string. This is safe for large
    documents (up to 50 MB per file) and avoids any per-document JSON body size
    limits that the batch endpoint might impose.
    """

    def __init__(
        self,
        client: SupermemoryClient,
        ingestion_cfg: IngestionConfig,
        sm_cfg: SupermemoryConfig,
    ) -> None:
        self._client = client
        self._ing = ingestion_cfg
        self._sm = sm_cfg

    async def run(self) -> tuple[int, int, float]:
        """Upload all documents and wait for indexing to complete.

        Returns:
            (indexed_count, failed_count, elapsed_seconds)
        """
        t0 = time.monotonic()
        doc_dir = Path(self._ing.documents_dir)

        if not doc_dir.exists():
            console.print(f"[yellow]Documents directory not found: {doc_dir}[/yellow]")
            return 0, 0, 0.0

        allowed_ext = set(self._ing.file_extensions)
        all_files = [f for f in doc_dir.rglob("*") if f.suffix.lower() in allowed_ext]

        if not all_files:
            console.print(f"[yellow]No files matching {allowed_ext} in {doc_dir}[/yellow]")
            return 0, 0, 0.0

        console.print(f"  Found [bold]{len(all_files)}[/bold] file(s) to ingest into Supermemory")

        indexed = 0
        failed = 0
        container_tag = self._sm.container_tag
        semaphore = asyncio.Semaphore(max(1, self._ing.upload_concurrency))

        async def _upload(f: Path) -> None:
            nonlocal indexed, failed
            async with semaphore:
                try:
                    await self._client.upload_file(f, container_tag)
                    indexed += 1
                except Exception as exc:
                    console.print(f"  [red]Upload failed ({f.name}): {exc}[/red]")
                    failed += 1
                finally:
                    progress.advance(task)
                if self._ing.upload_delay_seconds > 0:
                    await asyncio.sleep(self._ing.upload_delay_seconds)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Uploading to Supermemory (container: {container_tag})…",
                total=len(all_files),
            )
            await asyncio.gather(*[_upload(f) for f in all_files])

        # ── Poll until all documents are indexed ──────────────────────────
        if self._sm.verify_before_querying and indexed > 0:
            console.print("  Waiting for Supermemory to finish indexing…")
            for attempt in range(self._sm.max_polling_attempts):
                try:
                    status = await self._client.get_processing_documents()
                    remaining = status.get("totalCount", 0)
                    if remaining == 0:
                        console.print("  [green]All Supermemory documents indexed.[/green]")
                        break
                    console.print(
                        f"  [{attempt + 1}/{self._sm.max_polling_attempts}] Still processing {remaining} document(s)…"
                    )
                except Exception as exc:
                    console.print(f"  [yellow]Polling error: {exc}[/yellow]")
                await asyncio.sleep(self._sm.polling_interval_seconds)
            else:
                console.print(
                    "[yellow]Warning: max polling attempts reached; some documents may still be indexing.[/yellow]"
                )

        elapsed = time.monotonic() - t0
        return indexed, failed, elapsed
