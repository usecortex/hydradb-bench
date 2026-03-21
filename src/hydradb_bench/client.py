"""HydraDB API client using httpx (async-first)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx

from .models import HydraConfig, HydraSearchResult

logger = logging.getLogger(__name__)


# Keys to probe when normalizing search responses
_ANSWER_KEYS = ["answer", "response", "result", "text", "content", "output"]
_CONTEXT_KEYS = ["contexts", "chunks", "sources", "passages", "results", "documents"]
_CONTEXT_TEXT_KEYS = ["chunk_content", "content", "text", "passage", "document"]


def _extract_answer(data: dict[str, Any]) -> str:
    """Extract answer string from HydraDB response."""
    for key in _ANSWER_KEYS:
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    # Fallback: stringify whole response
    logger.warning("Could not find answer in response keys: %s", list(data.keys()))
    return str(data)


def _extract_contexts(data: dict[str, Any]) -> list[str]:
    """Extract context strings from HydraDB response."""
    for key in _CONTEXT_KEYS:
        val = data.get(key)
        if isinstance(val, list) and val:
            contexts = []
            for item in val:
                if isinstance(item, str):
                    contexts.append(item)
                elif isinstance(item, dict):
                    # Try known text field names
                    for text_key in _CONTEXT_TEXT_KEYS:
                        if isinstance(item.get(text_key), str):
                            contexts.append(item[text_key])
                            break
                    else:
                        contexts.append(str(item))
            return [c for c in contexts if c.strip()]
    logger.warning("Could not find context list in response keys: %s", list(data.keys()))
    return []


class HydraDBClient:
    """Async HTTP client for HydraDB API."""

    def __init__(self, config: HydraConfig) -> None:
        self.config = config
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "HydraDBClient":
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers=self._auth_headers(),
            timeout=self.config.timeout_seconds,
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
        }

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = ""
            try:
                body = response.text[:500]
            except Exception:
                pass
            raise RuntimeError(
                f"HydraDB API error {response.status_code} "
                f"for {response.request.url}: {body}"
            ) from e
        try:
            return response.json()
        except Exception:
            return {"raw": response.text}

    def _tenant_payload(self) -> dict[str, str]:
        payload: dict[str, str] = {"tenant_id": self.config.tenant_id}
        if self.config.sub_tenant_id:
            payload["sub_tenant_id"] = self.config.sub_tenant_id
        return payload

    async def delete_tenant(self) -> dict[str, Any]:
        """DELETE /tenants/delete — wipes all docs, vectors and metadata for the tenant."""
        assert self._client is not None
        logger.info("Deleting tenant: %s", self.config.tenant_id)
        # API only accepts tenant_id — deletes entire tenant and all sub-tenants
        params = {"tenant_id": self.config.tenant_id}
        response = await self._client.delete("/tenants/delete", params=params)
        if response.status_code == 404:
            logger.info("Tenant not found — nothing to delete.")
            return {"status": "not_found"}
        return self._handle_response(response)

    async def create_tenant(self) -> dict[str, Any]:
        """POST /tenants/create"""
        assert self._client is not None
        logger.info("Creating tenant: %s", self.config.tenant_id)
        payload = {"tenant_id": self.config.tenant_id}
        if self.config.sub_tenant_id:
            payload["sub_tenant_id"] = self.config.sub_tenant_id
        response = await self._client.post("/tenants/create", json=payload)
        # 409 Conflict (already exists) is acceptable
        if response.status_code == 409:
            logger.info("Tenant already exists, continuing.")
            return {"status": "already_exists"}
        return self._handle_response(response)

    async def upload_knowledge(self, file_path: Path) -> dict[str, Any]:
        """POST /ingestion/upload_knowledge (multipart)"""
        assert self._client is not None
        logger.info("Uploading: %s", file_path.name)
        with open(file_path, "rb") as f:
            files = {"files": (file_path.name, f, "application/octet-stream")}
            data = self._tenant_payload()
            response = await self._client.post(
                "/ingestion/upload_knowledge", files=files, data=data
            )
        return self._handle_response(response)

    async def add_memory(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """POST /memories/add_memory"""
        assert self._client is not None
        payload = {**self._tenant_payload(), "text": text, "metadata": metadata or {}}
        response = await self._client.post("/memories/add_memory", json=payload)
        return self._handle_response(response)

    async def verify_processing(self, file_ids: list[str]) -> dict[str, Any]:
        """POST /ingestion/verify_processing"""
        assert self._client is not None
        params = {
            "file_ids": ",".join(file_ids),
            "tenant_id": self.config.tenant_id,
        }
        if self.config.sub_tenant_id:
            params["sub_tenant_id"] = self.config.sub_tenant_id
        response = await self._client.post(
            "/ingestion/verify_processing", params=params
        )
        return self._handle_response(response)

    async def search_qna(self, query: str, max_results: int = 5) -> HydraSearchResult:
        """POST /recall/qna"""
        assert self._client is not None
        payload = {
            **self._tenant_payload(),
            "question": query,
            "max_chunks": max_results,
        }
        response = await self._client.post("/recall/qna", json=payload)
        data = self._handle_response(response)
        return HydraSearchResult(
            answer=_extract_answer(data),
            retrieved_contexts=_extract_contexts(data),
            raw_response=data,
        )

    async def full_recall(self, query: str, max_results: int = 10) -> HydraSearchResult:
        """POST /recall/full_recall"""
        assert self._client is not None
        payload = {
            **self._tenant_payload(),
            "query": query,
            "max_results": max_results,
        }
        response = await self._client.post("/recall/full_recall", json=payload)
        data = self._handle_response(response)
        return HydraSearchResult(
            answer=_extract_answer(data),
            retrieved_contexts=_extract_contexts(data),
            raw_response=data,
        )
