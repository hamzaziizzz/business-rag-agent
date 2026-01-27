from __future__ import annotations

import os

import httpx
import pytest

os.environ["RAG_VECTORSTORE"] = "memory"
os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["EMBEDDING_DIMENSION"] = "256"
os.environ["RAG_CHUNK_SIZE"] = "1000"
os.environ["RAG_CHUNK_OVERLAP"] = "100"
os.environ["RAG_DB_MAX_ROWS"] = "100"
os.environ["RAG_API_TIMEOUT"] = "5"
os.environ["RAG_API_MAX_BYTES"] = "10240"

from src.app.dependencies import reset_pipeline_cache
from src.app.main import app

pytestmark = pytest.mark.anyio


def get_client() -> httpx.AsyncClient:
    reset_pipeline_cache()
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_health_endpoint() -> None:
    async with get_client() as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_ingest_and_query() -> None:
    async with get_client() as client:
        ingest_response = await client.post(
            "/ingest",
            json={
                "documents": [
                    {
                        "doc_id": "sales",
                        "content": "Q4 sales were 100 units in North India.",
                        "metadata": {"source": "report"},
                    }
                ]
            },
        )
        assert ingest_response.status_code == 200
        assert ingest_response.json()["ingested"] >= 1

        query_response = await client.post(
            "/query", json={"query": "What were Q4 sales in North India?"}
        )
    assert query_response.status_code == 200
    payload = query_response.json()
    assert "Q4 sales were 100 units" in payload["answer"]
    assert payload["route"] == "rag"
    assert payload["request_id"]
    assert payload["sources"][0]["highlights"]


async def test_query_refuses_without_context() -> None:
    async with get_client() as client:
        response = await client.post("/query", json={"query": "What is the travel policy?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"].startswith("I don't know")
    assert payload["route"] == "rag"
    assert payload["request_id"]


async def test_ingest_files_text() -> None:
    async with get_client() as client:
        files = {
            "files": (
                "policy.txt",
                b"Travel policy: employees must book flights 7 days in advance.",
                "text/plain",
            )
        }
        ingest_response = await client.post("/ingest/files", files=files)
        assert ingest_response.status_code == 200
        assert ingest_response.json()["ingested"] >= 1

        query_response = await client.post("/query", json={"query": "What is the travel policy?"})
    assert query_response.status_code == 200
    payload = query_response.json()
    assert "Travel policy" in payload["answer"]
    assert payload["sources"][0]["highlights"]


async def test_stats_endpoint() -> None:
    async with get_client() as client:
        response = await client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["document_count"] == 0

        ingest_response = await client.post(
            "/ingest",
            json={
                "documents": [
                    {"doc_id": "policy", "content": "Travel policy details.", "metadata": {}}
                ]
            },
        )
        assert ingest_response.status_code == 200

        response = await client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["document_count"] >= 1


async def test_stats_health_endpoint() -> None:
    async with get_client() as client:
        response = await client.get("/stats/health")
        assert response.status_code == 200
        data = response.json()
        assert data["backend"] == "memory"
        assert data["ok"] is True


async def test_embedding_health_endpoint() -> None:
    async with get_client() as client:
        response = await client.get("/stats/embedding")
        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "hash"
        assert data["ok"] is True


async def test_query_routing_summarize() -> None:
    async with get_client() as client:
        ingest_response = await client.post(
            "/ingest",
            json={
                "documents": [
                    {
                        "doc_id": "summary",
                        "content": "The policy covers travel bookings and approvals.",
                        "metadata": {"source": "policy"},
                    }
                ]
            },
        )
        assert ingest_response.status_code == 200
        response = await client.post(
            "/query",
            json={"query": "Summarize the travel policy."},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "summarize"
    assert payload["answer"].startswith("Summary based on the provided context")


async def test_query_routing_sql_refuses() -> None:
    async with get_client() as client:
        response = await client.post(
            "/query",
            json={"query": "sql: select * from users", "route": "sql"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "sql"
    assert payload["refusal_reason"] == "sql_not_configured"
