from __future__ import annotations

"""Tests for structured JSON outputs and routing behavior."""

import os

import httpx
import pytest

os.environ["RAG_VECTORSTORE"] = "memory"
os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["EMBEDDING_DIMENSION"] = "256"

from src.app.dependencies import reset_pipeline_cache
from src.app.main import app
from src.app.settings import settings

pytestmark = pytest.mark.anyio


def get_client() -> httpx.AsyncClient:
    """Build an ASGI test client."""
    reset_pipeline_cache()
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_structured_json_response() -> None:
    """Ensure structured JSON summaries are returned."""
    async with get_client() as client:
        ingest_response = await client.post(
            "/ingest",
            json={
                "documents": [
                    {
                        "doc_id": "policy",
                        "content": (
                            "Travel policy covers flights and hotels. "
                            "Manager approval is required for international trips."
                        ),
                        "metadata": {"source": "policy"},
                    }
                ]
            },
        )
        assert ingest_response.status_code == 200

        response = await client.post(
            "/query",
            json={"query": "What does the travel policy cover?"},
        )
    assert response.status_code == 200
    payload = response.json()
    structured = payload.get("structured")
    assert structured is not None
    assert structured.get("type") == "json"
    assert isinstance(structured.get("summary"), str)
    assert "Travel policy" in structured.get("summary")
    points = structured.get("points")
    assert isinstance(points, list)
    assert points
    assert any("Travel policy" in item for item in points)


async def test_llm_router_fallback_to_rules() -> None:
    """Ensure LLM router fallback uses rules on failure."""
    original = {
        "router_mode": settings.router_mode,
        "router_provider": settings.router_provider,
        "router_model": settings.router_model,
        "router_timeout": settings.router_timeout,
        "ollama_base_url": settings.ollama_base_url,
    }
    try:
        object.__setattr__(settings, "router_mode", "llm")
        object.__setattr__(settings, "router_provider", "ollama")
        object.__setattr__(settings, "router_model", "")
        object.__setattr__(settings, "router_timeout", 0.1)
        object.__setattr__(settings, "ollama_base_url", "http://127.0.0.1:9")
        async with get_client() as client:
            response = await client.post(
                "/query",
                json={"query": "Summarize the travel policy."},
            )
        assert response.status_code == 200
        payload = response.json()
        assert payload["route"] == "summarize"
    finally:
        for key, value in original.items():
            object.__setattr__(settings, key, value)

