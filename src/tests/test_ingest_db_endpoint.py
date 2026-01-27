from __future__ import annotations

import os
import sqlite3

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
    reset_pipeline_cache()
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_ingest_db_endpoint(tmp_path) -> None:
    db_path = tmp_path / "users.db"
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        cursor.executemany("INSERT INTO users (name) VALUES (?)", [("Ava",), ("Ben",)])
        conn.commit()
    finally:
        conn.close()

    async with get_client() as client:
        response = await client.post(
            "/ingest/db",
            json={
                "connection_uri": f"sqlite:///{db_path}",
                "query": "SELECT id, name FROM users",
                "params": {},
                "limit": 10,
                "source_name": "users",
            },
        )
        assert response.status_code == 200
        assert response.json()["ingested"] >= 1
        manual_response = await client.post(
            "/ingest",
            json={"documents": [{"doc_id": "note", "content": "Manual note entry."}]},
        )
        assert manual_response.status_code == 200

        query_response = await client.post(
            "/query",
            json={"query": "Ava", "top_k": 3, "source_types": ["db"]},
        )
        manual_query = await client.post(
            "/query",
            json={
                "query": "Manual note entry.",
                "top_k": 3,
                "source_types": ["db"],
                "source_names": ["manual"],
                "source_filter_mode": "or",
            },
        )
    assert query_response.status_code == 200
    payload = query_response.json()
    assert payload["sources"]
    assert all(source["metadata"]["source_type"] == "db" for source in payload["sources"])
    assert payload["structured"]
    assert payload["structured"]["type"] == "db"
    assert manual_query.status_code == 200
    manual_payload = manual_query.json()
    assert manual_payload["sources"]
    assert any(
        source["metadata"]["source_type"] == "manual" for source in manual_payload["sources"]
    )
