from __future__ import annotations

import os

import httpx
import pytest

os.environ["RAG_VECTORSTORE"] = "memory"
os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["EMBEDDING_DIMENSION"] = "256"

from src.app.dependencies import reset_pipeline_cache
from src.app.main import app

pytestmark = pytest.mark.anyio


def get_client() -> httpx.AsyncClient:
    reset_pipeline_cache()
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_api_key_required_for_query() -> None:
    original = os.environ.get("RAG_API_KEYS")
    os.environ["RAG_API_KEYS"] = "secret"
    try:
        async with get_client() as client:
            response = await client.post("/query", json={"query": "test"})
            assert response.status_code == 401

            ok_response = await client.post(
                "/query",
                json={"query": "test"},
                headers={"X-API-Key": "secret"},
            )
            assert ok_response.status_code == 200
    finally:
        if original is None:
            os.environ.pop("RAG_API_KEYS", None)
        else:
            os.environ["RAG_API_KEYS"] = original
