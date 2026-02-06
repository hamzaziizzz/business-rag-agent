from __future__ import annotations

"""Tests for tenant isolation and role enforcement."""

import json
import os

import httpx
import pytest

os.environ["RAG_VECTORSTORE"] = "memory"
os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["EMBEDDING_DIMENSION"] = "256"
os.environ["RAG_METADATA_DB_URI"] = ""
os.environ["RAG_AUDIT_DB_URI"] = ""
from src.app.dependencies import reset_pipeline_cache
from src.app.main import app

pytestmark = pytest.mark.anyio


def get_client() -> httpx.AsyncClient:
    """Build an ASGI test client."""
    reset_pipeline_cache()
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_reader_cannot_ingest() -> None:
    """Ensure reader role cannot ingest documents."""
    original = os.environ.get("RAG_API_KEY_MAP")
    os.environ["RAG_API_KEY_MAP"] = json.dumps(
        {
            "reader-key": {"role": "reader", "tenant_id": "tenant-a"},
        }
    )
    async with get_client() as client:
        response = await client.post(
            "/ingest",
            json={"documents": [{"content": "confidential data"}]},
            headers={"X-API-Key": "reader-key"},
        )
        assert response.status_code == 403
    if original is None:
        os.environ.pop("RAG_API_KEY_MAP", None)
    else:
        os.environ["RAG_API_KEY_MAP"] = original


async def test_tenant_isolation_in_query() -> None:
    """Ensure tenant-specific queries only see tenant data."""
    original = os.environ.get("RAG_API_KEY_MAP")
    os.environ["RAG_API_KEY_MAP"] = json.dumps(
        {
            "reader-key": {"role": "reader", "tenant_id": "tenant-a"},
            "writer-key-a": {"role": "writer", "tenant_id": "tenant-a"},
            "writer-key-b": {"role": "writer", "tenant_id": "tenant-b"},
        }
    )
    async with get_client() as client:
        await client.post(
            "/ingest",
            json={"documents": [{"content": "Alpha only data"}]},
            headers={"X-API-Key": "writer-key-a"},
        )
        await client.post(
            "/ingest",
            json={"documents": [{"content": "Beta only data"}]},
            headers={"X-API-Key": "writer-key-b"},
        )

        response_a = await client.post(
            "/query",
            json={"query": "Alpha data", "top_k": 3},
            headers={"X-API-Key": "reader-key"},
        )
        assert response_a.status_code == 200
        payload_a = response_a.json()
        assert all(
            chunk["metadata"].get("tenant_id") == "tenant-a"
            for chunk in payload_a["sources"]
        )

        response_b = await client.post(
            "/query",
            json={"query": "Alpha data", "top_k": 3},
            headers={"X-API-Key": "writer-key-b"},
        )
        assert response_b.status_code == 200
        payload_b = response_b.json()
        assert all(
            chunk["metadata"].get("tenant_id") == "tenant-b"
            for chunk in payload_b["sources"]
        )
    if original is None:
        os.environ.pop("RAG_API_KEY_MAP", None)
    else:
        os.environ["RAG_API_KEY_MAP"] = original
