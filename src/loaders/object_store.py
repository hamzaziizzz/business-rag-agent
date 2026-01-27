from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.rag.types import Document


class ObjectStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class ObjectStoreConfig:
    bucket: str
    key: str
    endpoint_url: str | None
    region: str | None
    access_key: str | None
    secret_key: str | None
    session_token: str | None
    max_bytes: int | None


def load_object_document(
    config: ObjectStoreConfig,
    loader: Callable[[bytes, str, str], Document],
    doc_id: str,
    source: str,
) -> Document:
    data = _fetch_object_bytes(config)
    if config.max_bytes is not None and len(data) > config.max_bytes:
        raise ObjectStoreError("Object exceeds configured max_bytes")
    return loader(data, doc_id=doc_id, source=source)


def _fetch_object_bytes(config: ObjectStoreConfig) -> bytes:
    try:
        import boto3
    except ImportError as exc:
        raise ObjectStoreError("boto3 is required for object storage ingestion") from exc

    session = boto3.session.Session(
        aws_access_key_id=config.access_key,
        aws_secret_access_key=config.secret_key,
        aws_session_token=config.session_token,
        region_name=config.region,
    )
    client = session.client("s3", endpoint_url=config.endpoint_url)
    try:
        response = client.get_object(Bucket=config.bucket, Key=config.key)
    except Exception as exc:
        raise ObjectStoreError(f"Failed to fetch object: {exc}") from exc
    body = response.get("Body")
    if body is None:
        raise ObjectStoreError("Object body missing in response")
    data = body.read()
    if not isinstance(data, (bytes, bytearray)):
        raise ObjectStoreError("Invalid object body data")
    return bytes(data)
