from __future__ import annotations

import os
import sqlite3

import httpx
import pytest

os.environ["RAG_VECTORSTORE"] = "memory"
os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["EMBEDDING_DIMENSION"] = "256"
os.environ["RAG_SQL_ALLOWED_TABLES"] = "users"
os.environ["RAG_SQL_ALLOWED_COLUMNS"] = "id,name"

from src.app.dependencies import reset_pipeline_cache
from src.app.main import app

pytestmark = pytest.mark.anyio


def get_client() -> httpx.AsyncClient:
    reset_pipeline_cache()
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_sql_query_route(tmp_path) -> None:
    db_path = tmp_path / "users.db"
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        cursor.executemany("INSERT INTO users (name) VALUES (?)", [("Ada",), ("Ben",)])
        conn.commit()
    finally:
        conn.close()

    os.environ["RAG_SQL_DATABASE_URI"] = f"sqlite:///{db_path}"

    async with get_client() as client:
        response = await client.post(
            "/query",
            json={
                "query": "SELECT id, name FROM users WHERE name = 'Ada'",
                "route": "sql",
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["route"] == "sql"
        assert payload["structured"]["type"] == "sql"
        assert payload["structured"]["records"]
