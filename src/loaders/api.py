from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import urlparse

import httpx

from src.rag.types import Document


class APILoaderError(RuntimeError):
    pass


@dataclass(frozen=True)
class APIIngestConfig:
    url: str
    method: str
    headers: dict[str, str]
    params: dict[str, Any]
    json_body: Any | None
    timeout: float
    max_bytes: int
    source_name: str
    source_type: str = "api"


async def fetch_api_documents(
    config: APIIngestConfig, client: httpx.AsyncClient | None = None
) -> list[Document]:
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=config.timeout)
    try:
        response = await client.request(
            config.method,
            config.url,
            headers=config.headers,
            params=config.params,
            json=config.json_body,
        )
        response.raise_for_status()
        content = response.content
        if config.max_bytes and len(content) > config.max_bytes:
            raise APILoaderError("API response exceeds maximum size limit")
        return _response_to_documents(
            content,
            response.headers.get("content-type", ""),
            config,
        )
    except httpx.HTTPError as exc:
        raise APILoaderError(str(exc)) from exc
    finally:
        if owns_client and client is not None:
            await client.aclose()


def _response_to_documents(
    content: bytes,
    content_type: str,
    config: APIIngestConfig,
) -> list[Document]:
    text_content = content.decode("utf-8", errors="ignore")
    if "application/json" in content_type.lower():
        try:
            data = json.loads(text_content)
        except json.JSONDecodeError as exc:
            raise APILoaderError("Failed to parse JSON response") from exc
        return _json_to_documents(data, config)
    return [
        Document(
            doc_id=f"{config.source_name}-1",
            content=text_content,
            metadata=_base_metadata(config, item_index=1),
        )
    ]


def _json_to_documents(data: Any, config: APIIngestConfig) -> list[Document]:
    documents: list[Document] = []
    if isinstance(data, list):
        for idx, item in enumerate(data, start=1):
            payload = json.dumps(item, ensure_ascii=True, default=str)
            documents.append(
                Document(
                    doc_id=f"{config.source_name}-{idx}",
                    content=payload,
                    metadata=_base_metadata(config, item_index=idx),
                )
            )
        return documents
    payload = json.dumps(data, ensure_ascii=True, default=str)
    documents.append(
        Document(
            doc_id=f"{config.source_name}-1",
            content=payload,
            metadata=_base_metadata(config, item_index=1),
        )
    )
    return documents


def _base_metadata(config: APIIngestConfig, item_index: int) -> dict[str, Any]:
    parsed = urlparse(config.url)
    source = f"api:{parsed.netloc}{parsed.path}"
    return {
        "source": source,
        "source_type": config.source_type,
        "source_name": config.source_name,
        "item_index": item_index,
        "url_hash": _hash_text(config.url),
    }


def _hash_text(text_value: str) -> str:
    return hashlib.sha256(text_value.encode("utf-8")).hexdigest()
