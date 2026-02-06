from __future__ import annotations

"""Metadata persistence for ingestion events."""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse, urlunparse


class MetadataStoreError(RuntimeError):
    """Raised when metadata persistence fails."""
    pass


@dataclass(frozen=True)
class IngestionMeta:
    """Ingestion metadata tracked for a single source."""
    source_type: str
    source_name: str
    source_uri: str
    status: str
    ingested_count: int | None = None
    chunk_count: int | None = None
    error: str | None = None
    extra: dict[str, Any] | None = None


class MetadataStore:
    """Store ingestion metadata in a SQL database."""
    def __init__(self, connection_uri: str) -> None:
        """Initialize the metadata store and ensure tables exist."""
        try:
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
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise MetadataStoreError(
                "sqlalchemy is required to use the metadata store"
            ) from exc

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
        """Create a new ingestion record and return its ID."""
        record_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        payload = self._serialize(meta, created_at, None, record_id)
        with self._engine.begin() as conn:
            conn.execute(self._table.insert().values(**payload))
        return record_id

    def record_complete(self, record_id: str, ingested: int, chunk_count: int) -> None:
        """Mark a record as completed with counts."""
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
        """Mark a record as failed with an error message."""
        completed_at = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                self._table.update()
                .where(self._table.c.id == record_id)
                .values(status="failed", error=error, completed_at=completed_at)
            )

    @staticmethod
    def redact_uri(uri: str) -> str:
        """Redact credentials from connection URIs before storage."""
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
        """Prepare a metadata row for insertion."""
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
