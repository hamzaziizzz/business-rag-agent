from __future__ import annotations

"""Tests for deleting ingested sources."""

import os

import httpx
import pytest

os.environ["RAG_VECTORSTORE"] = "memory"
os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["EMBEDDING_DIMENSION"] = "256"
os.environ["RAG_CHUNK_SIZE"] = "1000"
os.environ["RAG_CHUNK_OVERLAP"] = "100"

from src.app.dependencies import reset_pipeline_cache
from src.app.main import app

pytestmark = pytest.mark.anyio


def get_client() -> httpx.AsyncClient:
    """Build an ASGI test client."""
    reset_pipeline_cache()
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_delete_by_source() -> None:
    """Ensure delete endpoint removes documents by source."""
    original = os.environ.get("RAG_API_KEY_MAP")
    os.environ["RAG_API_KEY_MAP"] = '{"admin-key": {"role": "admin", "tenant_id": "default"}}'
    async with get_client() as client:
        ingest_response = await client.post(
            "/ingest",
            json={
                "documents": [
                    {"doc_id": "db-1", "content": "Alpha", "metadata": {"source_type": "db", "source_name": "a"}},
                    {"doc_id": "db-2", "content": "Beta", "metadata": {"source_type": "db", "source_name": "b"}},
                ]
            },
            headers={"X-API-Key": "admin-key"},
        )
        assert ingest_response.status_code == 200

        delete_response = await client.post(
            "/ingest/delete",
            json={"source_names": ["a"], "source_filter_mode": "and"},
            headers={"X-API-Key": "admin-key"},
        )
        assert delete_response.status_code == 200
        assert delete_response.json()["deleted"] >= 1

        query_response = await client.post(
            "/query",
            json={"query": "Alpha", "source_names": ["a"], "source_filter_mode": "and"},
            headers={"X-API-Key": "admin-key"},
        )
        assert query_response.status_code == 200
        assert query_response.json()["sources"] == []
    if original is None:
        os.environ.pop("RAG_API_KEY_MAP", None)
    else:
        os.environ["RAG_API_KEY_MAP"] = original
