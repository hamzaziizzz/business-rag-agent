from __future__ import annotations

import json

import httpx
import pytest

from src.loaders.api import APIIngestConfig, fetch_api_documents


@pytest.mark.anyio
async def test_fetch_api_documents_json_list() -> None:
    payload = [{"name": "Alpha"}, {"name": "Beta"}]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            content=json.dumps(payload).encode("utf-8"),
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        config = APIIngestConfig(
            url="http://test/items",
            method="GET",
            headers={},
            params={},
            json_body=None,
            timeout=5,
            max_bytes=1024,
            source_name="items",
            source_type="api",
        )
        documents = await fetch_api_documents(config, client=client)

    assert len(documents) == 2
    assert documents[0].metadata["source_type"] == "api"


@pytest.mark.anyio
async def test_fetch_api_documents_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"plain text")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        config = APIIngestConfig(
            url="http://test/text",
            method="GET",
            headers={},
            params={},
            json_body=None,
            timeout=5,
            max_bytes=1024,
            source_name="text",
            source_type="api",
        )
        documents = await fetch_api_documents(config, client=client)

    assert len(documents) == 1
    assert "plain text" in documents[0].content
