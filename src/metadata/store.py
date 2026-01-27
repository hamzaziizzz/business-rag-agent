from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse, urlunparse

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
)


class MetadataStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class IngestionMeta:
    source_type: str
    source_name: str
    source_uri: str
    status: str
    ingested_count: int | None = None
    chunk_count: int | None = None
    error: str | None = None
    extra: dict[str, Any] | None = None


class MetadataStore:
    def __init__(self, connection_uri: str) -> None:
        self._engine = create_engine(connection_uri)
        self._metadata = MetaData()
        self._table = Table(
            "ingestion_records",
            self._metadata,
            Column("id", String(36), primary_key=True),
            Column("source_type", String(64), nullable=False),
            Column("source_name", String(255), nullable=False),
            Column("source_uri", Text, nullable=False),
            Column("status", String(32), nullable=False),
            Column("ingested_count", Integer, nullable=True),
            Column("chunk_count", Integer, nullable=True),
            Column("error", Text, nullable=True),
            Column("extra", Text, nullable=True),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Column("completed_at", DateTime(timezone=True), nullable=True),
        )
        self._metadata.create_all(self._engine)

    def record_start(self, meta: IngestionMeta) -> str:
        record_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        payload = self._serialize(meta, created_at, None, record_id)
        with self._engine.begin() as conn:
            conn.execute(self._table.insert().values(**payload))
        return record_id

    def record_complete(self, record_id: str, ingested: int, chunk_count: int) -> None:
        completed_at = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                self._table.update()
                .where(self._table.c.id == record_id)
                .values(
                    status="completed",
                    ingested_count=ingested,
                    chunk_count=chunk_count,
                    completed_at=completed_at,
                )
            )

    def record_failure(self, record_id: str, error: str) -> None:
        completed_at = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                self._table.update()
                .where(self._table.c.id == record_id)
                .values(status="failed", error=error, completed_at=completed_at)
            )

    @staticmethod
    def redact_uri(uri: str) -> str:
        if "://" not in uri:
            return uri
        parsed = urlparse(uri)
        if parsed.password is None:
            return uri
        netloc = parsed.hostname or ""
        if parsed.username:
            netloc = f"{parsed.username}:***@{netloc}"
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        return urlunparse(
            (
                parsed.scheme,
                netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )

    def _serialize(
        self,
        meta: IngestionMeta,
        created_at: datetime,
        completed_at: datetime | None,
        record_id: str,
    ) -> dict[str, Any]:
        extra = None
        if meta.extra:
            extra = json.dumps(meta.extra, ensure_ascii=True, default=str)
        return {
            "id": record_id,
            "source_type": meta.source_type,
            "source_name": meta.source_name,
            "source_uri": meta.source_uri,
            "status": meta.status,
            "ingested_count": meta.ingested_count,
            "chunk_count": meta.chunk_count,
            "error": meta.error,
            "extra": extra,
            "created_at": created_at,
            "completed_at": completed_at,
        }
