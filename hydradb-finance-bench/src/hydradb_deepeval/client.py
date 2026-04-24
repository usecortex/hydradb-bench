"""HydraDB async HTTP client — full API coverage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from .models import HydraConfig


class HydraDBClient:
    """Async HTTP client for the HydraDB API."""

    def __init__(self, config: HydraConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> HydraDBClient:
        self._client = httpx.AsyncClient(
            base_url=self._config.base_url,
            headers={"Authorization": f"Bearer {self._config.api_key}"},
            timeout=self._config.timeout_seconds,
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("HydraDBClient must be used as an async context manager")
        return self._client

    def _tenant_params(self) -> dict[str, str]:
        p: dict[str, str] = {"tenant_id": self._config.tenant_id}
        if self._config.sub_tenant_id:
            p["sub_tenant_id"] = self._config.sub_tenant_id
        return p

    def _tenant_body(self) -> dict[str, Any]:
        b: dict[str, Any] = {"tenant_id": self._config.tenant_id}
        if self._config.sub_tenant_id:
            b["sub_tenant_id"] = self._config.sub_tenant_id
        return b

    def _tenant_body_with_sub(self) -> dict[str, Any]:
        """Like _tenant_body but always includes sub_tenant_id (required by raw-embeddings endpoints)."""
        return {
            "tenant_id": self._config.tenant_id,
            "sub_tenant_id": self._config.sub_tenant_id or "",
        }

    # ------------------------------------------------------------------
    # Tenant management
    # ------------------------------------------------------------------

    async def create_tenant(self) -> dict:
        """POST /tenants/create"""
        client = self._ensure_client()
        response = await client.post(
            "/tenants/create",
            json={"tenant_id": self._config.tenant_id},
        )
        if response.status_code == 409:
            return {"status": "already_exists"}
        response.raise_for_status()
        return response.json()

    async def delete_tenant(self) -> dict:
        """DELETE /tenants/delete?tenant_id=..."""
        client = self._ensure_client()
        response = await client.delete(
            "/tenants/delete",
            params={"tenant_id": self._config.tenant_id},
        )
        if response.status_code == 404:
            return {"status": "not_found"}
        response.raise_for_status()
        return response.json()

    async def list_tenant_ids(self) -> dict:
        """GET /tenants/tenant_ids — list all tenant IDs for the org."""
        client = self._ensure_client()
        response = await client.get("/tenants/tenant_ids")
        response.raise_for_status()
        return response.json()

    async def list_sub_tenant_ids(self) -> dict:
        """GET /tenants/sub_tenant_ids — list sub-tenants with indexed data."""
        client = self._ensure_client()
        response = await client.get(
            "/tenants/sub_tenant_ids",
            params={"tenant_id": self._config.tenant_id},
        )
        response.raise_for_status()
        return response.json()

    async def get_infra_status(self) -> dict:
        """GET /tenants/infra/status — scheduler/graph/vectorstore status."""
        client = self._ensure_client()
        response = await client.get(
            "/tenants/infra/status",
            params={"tenant_id": self._config.tenant_id},
        )
        response.raise_for_status()
        return response.json()

    async def get_tenant_stats(self) -> dict:
        """GET /tenants/monitor — object count and vector dimensions."""
        client = self._ensure_client()
        response = await client.get(
            "/tenants/monitor",
            params={"tenant_id": self._config.tenant_id},
        )
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    async def upload_knowledge(self, file_path: Path, upsert: bool = True, sub_tenant_id: str | None = None) -> dict:
        """POST /ingestion/upload_knowledge  multipart form."""
        client = self._ensure_client()
        form_data: dict[str, str] = {
            "tenant_id": self._config.tenant_id,
            "upsert": str(upsert).lower(),
        }
        effective_sub_tenant = sub_tenant_id if sub_tenant_id is not None else self._config.sub_tenant_id
        if effective_sub_tenant:
            form_data["sub_tenant_id"] = effective_sub_tenant

        with file_path.open("rb") as f:
            files = {"files": (file_path.name, f, "application/octet-stream")}
            response = await client.post(
                "/ingestion/upload_knowledge",
                files=files,
                data=form_data,
            )
        response.raise_for_status()
        return response.json()

    async def verify_processing(self, source_ids: list[str], sub_tenant_id: str | None = None) -> dict:
        """POST /ingestion/verify_processing — repeated file_ids query params."""
        client = self._ensure_client()
        params: list[tuple[str, str]] = [("file_ids", sid) for sid in source_ids]
        params.append(("tenant_id", self._config.tenant_id))
        effective_sub_tenant = sub_tenant_id if sub_tenant_id is not None else self._config.sub_tenant_id
        if effective_sub_tenant:
            params.append(("sub_tenant_id", effective_sub_tenant))
        response = await client.post("/ingestion/verify_processing", params=params)
        response.raise_for_status()
        return response.json()

    async def delete_knowledge(self, source_ids: list[str]) -> dict:
        """POST /knowledge/delete_knowledge — delete sources by ID."""
        client = self._ensure_client()
        payload = {**self._tenant_body(), "ids": source_ids}
        response = await client.post("/knowledge/delete_knowledge", json=payload)
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Memories
    # ------------------------------------------------------------------

    async def add_memory(
        self,
        text: str,
        *,
        title: str = "",
        infer: bool = False,
        document_metadata: dict | None = None,
        expiry_time: int | None = None,
    ) -> dict:
        """POST /memories/add_memory — body is array of MemoryItem."""
        client = self._ensure_client()
        item: dict[str, Any] = {"text": text, "infer": infer}
        if title:
            item["title"] = title
        if document_metadata:
            # API expects document_metadata as a JSON string
            item["document_metadata"] = json.dumps(document_metadata)
        if expiry_time is not None:
            item["expiry_time"] = expiry_time
        payload = {**self._tenant_body(), "memories": [item]}
        response = await client.post("/memories/add_memory", json=payload)
        response.raise_for_status()
        return response.json()

    async def delete_memory(self, memory_id: str) -> dict:
        """DELETE /memories/delete_memory"""
        client = self._ensure_client()
        params = {**self._tenant_params(), "memory_id": memory_id}
        response = await client.delete("/memories/delete_memory", params=params)
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Recall / search
    # ------------------------------------------------------------------

    async def full_recall(
        self,
        query: str,
        max_results: int = 10,
        mode: str = "fast",
        alpha: float = 0.8,
        graph_context: bool = True,
        recency_bias: float = 0.0,
        metadata_filters: dict | None = None,
        additional_context: str | None = None,
        sub_tenant_id: str | None = None,
        source_ids: list[str] | None = None,
    ) -> dict:
        """POST /recall/full_recall"""
        client = self._ensure_client()
        body = {**self._tenant_body()}
        if sub_tenant_id is not None:
            body["sub_tenant_id"] = sub_tenant_id
        payload: dict[str, Any] = {
            **body,
            "query": query,
            "max_results": max_results,
            "mode": mode,
            "alpha": alpha,
            "graph_context": graph_context,
            "recency_bias": recency_bias,
        }
        if metadata_filters:
            payload["metadata_filters"] = metadata_filters
        if additional_context:
            payload["additional_context"] = additional_context
        if source_ids:
            payload["source_ids"] = source_ids
        response = await client.post("/recall/full_recall", json=payload)
        response.raise_for_status()
        return response.json()

    async def recall_preferences(
        self,
        query: str,
        max_results: int = 10,
        mode: str = "fast",
        alpha: float = 0.8,
        graph_context: bool = True,
        sub_tenant_id: str | None = None,
    ) -> dict:
        """POST /recall/recall_preferences — search user memories."""
        client = self._ensure_client()
        body = {**self._tenant_body()}
        if sub_tenant_id is not None:
            body["sub_tenant_id"] = sub_tenant_id
        payload: dict[str, Any] = {
            **body,
            "query": query,
            "max_results": max_results,
            "mode": mode,
            "alpha": alpha,
            "graph_context": graph_context,
        }
        response = await client.post("/recall/recall_preferences", json=payload)
        response.raise_for_status()
        return response.json()

    async def boolean_recall(
        self,
        query: str,
        operator: str = "or",
        max_results: int = 10,
        search_mode: str = "sources",
        sub_tenant_id: str | None = None,
    ) -> dict:
        """POST /recall/boolean_recall — full-text / boolean search."""
        client = self._ensure_client()
        body = {**self._tenant_body()}
        if sub_tenant_id is not None:
            body["sub_tenant_id"] = sub_tenant_id
        payload: dict[str, Any] = {
            **body,
            "query": query,
            "operator": operator,
            "max_results": max_results,
            "search_mode": search_mode,
        }
        response = await client.post("/recall/boolean_recall", json=payload)
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # List & fetch
    # ------------------------------------------------------------------

    async def list_knowledge(
        self,
        page: int = 1,
        page_size: int = 50,
        source_ids: list[str] | None = None,
        filters: dict | None = None,
        include_fields: list[str] | None = None,
    ) -> dict:
        """POST /list/data  kind=knowledge"""
        client = self._ensure_client()
        payload: dict[str, Any] = {
            **self._tenant_body(),
            "kind": "knowledge",
            "page": page,
            "page_size": page_size,
        }
        if source_ids:
            payload["source_ids"] = source_ids
        if filters:
            payload["filters"] = filters
        if include_fields:
            payload["include_fields"] = include_fields
        response = await client.post("/list/data", json=payload)
        response.raise_for_status()
        return response.json()

    async def list_memories(
        self,
        page: int = 1,
        page_size: int = 50,
    ) -> dict:
        """POST /list/data  kind=memories"""
        client = self._ensure_client()
        payload: dict[str, Any] = {
            **self._tenant_body(),
            "kind": "memories",
            "page": page,
            "page_size": page_size,
        }
        response = await client.post("/list/data", json=payload)
        response.raise_for_status()
        return response.json()

    async def get_graph_relations(self, source_id: str, limit: int = 250, is_memory: bool = False) -> dict:
        """GET /list/graph_relations_by_id"""
        client = self._ensure_client()
        params = {
            **self._tenant_params(),
            "source_id": source_id,
            "limit": limit,
            "is_memory": str(is_memory).lower(),
        }
        response = await client.get("/list/graph_relations_by_id", params=params)
        response.raise_for_status()
        return response.json()

    async def fetch_content(
        self,
        source_id: str,
        mode: str = "content",
        expiry_seconds: int | None = None,
    ) -> dict:
        """POST /fetch/content — retrieve source content or presigned URL."""
        client = self._ensure_client()
        payload: dict[str, Any] = {
            **self._tenant_body(),
            "source_id": source_id,
            "mode": mode,
        }
        if expiry_seconds is not None:
            payload["expiry_seconds"] = expiry_seconds
        response = await client.post("/fetch/content", json=payload)
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Raw embeddings
    # ------------------------------------------------------------------

    async def insert_raw_embeddings(
        self,
        embeddings: list[dict],
        upsert: bool = False,
    ) -> dict:
        """POST /embeddings/insert_raw_embeddings
        embeddings: list of {source_id, metadata?, embeddings: [{chunk_id, embedding}]}
        """
        client = self._ensure_client()
        payload: dict[str, Any] = {
            **self._tenant_body(),
            "embeddings": embeddings,
            "upsert": upsert,
        }
        response = await client.post("/embeddings/insert_raw_embeddings", json=payload)
        response.raise_for_status()
        return response.json()

    async def search_raw_embeddings(
        self,
        query_embedding: list[float],
        limit: int = 10,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
    ) -> dict:
        """POST /embeddings/search_raw_embeddings — sub_tenant_id required."""
        client = self._ensure_client()
        payload: dict[str, Any] = {
            **self._tenant_body_with_sub(),
            "query_embedding": query_embedding,
            "limit": limit,
        }
        if filter_expr:
            payload["filter_expr"] = filter_expr
        if output_fields:
            payload["output_fields"] = output_fields
        response = await client.post("/embeddings/search_raw_embeddings", json=payload)
        response.raise_for_status()
        return response.json()

    async def filter_raw_embeddings(
        self,
        source_id: str | None = None,
        chunk_ids: list[str] | None = None,
        limit: int = 100,
        output_fields: list[str] | None = None,
    ) -> dict:
        """POST /embeddings/filter_raw_embeddings — sub_tenant_id required."""
        client = self._ensure_client()
        payload: dict[str, Any] = {**self._tenant_body_with_sub(), "limit": limit}
        if source_id:
            payload["source_id"] = source_id
        if chunk_ids:
            payload["chunk_ids"] = chunk_ids
        if output_fields:
            payload["output_fields"] = output_fields
        response = await client.post("/embeddings/filter_raw_embeddings", json=payload)
        response.raise_for_status()
        return response.json()

    async def delete_raw_embeddings(
        self,
        source_id: str | None = None,
        chunk_ids: list[str] | None = None,
    ) -> dict:
        """DELETE /embeddings/delete_raw_embeddings"""
        client = self._ensure_client()
        # Build as list of tuples so repeated chunk_ids keys are sent correctly
        params: list[tuple[str, str]] = [
            ("tenant_id", self._config.tenant_id),
        ]
        if self._config.sub_tenant_id:
            params.append(("sub_tenant_id", self._config.sub_tenant_id))
        if source_id:
            params.append(("source_id", source_id))
        if chunk_ids:
            params.extend(("chunk_ids", cid) for cid in chunk_ids)
        response = await client.delete("/embeddings/delete_raw_embeddings", params=params)
        response.raise_for_status()
        return response.json()
