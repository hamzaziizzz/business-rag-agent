from __future__ import annotations

import httpx
import pytest

from src.loaders.api import APIIngestConfig, APILoaderError, fetch_api_documents


@pytest.mark.anyio
async def test_fetch_api_documents_limit_exceeded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"a" * 200)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        config = APIIngestConfig(
            url="http://test/large",
            method="GET",
            headers={},
            params={},
            json_body=None,
            timeout=5,
            max_bytes=100,
            source_name="large",
            source_type="api",
        )
        with pytest.raises(APILoaderError):
            await fetch_api_documents(config, client=client)
