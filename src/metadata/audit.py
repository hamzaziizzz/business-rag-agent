from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Column, DateTime, MetaData, String, Table, Text, create_engine


class AuditStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class AuditEvent:
    event_type: str
    request_id: str
    tenant_id: str
    actor: str
    status: str
    route: str | None = None
    detail: dict[str, Any] | None = None


class AuditStore:
    def __init__(self, connection_uri: str) -> None:
        self._engine = create_engine(connection_uri)
        self._metadata = MetaData()
        self._table = Table(
            "audit_events",
            self._metadata,
            Column("id", String(36), primary_key=True),
            Column("request_id", String(64), nullable=False),
            Column("event_type", String(64), nullable=False),
            Column("tenant_id", String(128), nullable=False),
            Column("actor", String(128), nullable=False),
            Column("route", String(64), nullable=True),
            Column("status", String(32), nullable=False),
            Column("detail", Text, nullable=True),
            Column("created_at", DateTime(timezone=True), nullable=False),
        )
        self._metadata.create_all(self._engine)

    def record_event(self, event: AuditEvent) -> None:
        created_at = datetime.now(timezone.utc)
        payload = {
            "id": str(uuid.uuid4()),
            "request_id": event.request_id,
            "event_type": event.event_type,
            "tenant_id": event.tenant_id,
            "actor": event.actor,
            "route": event.route,
            "status": event.status,
            "detail": json.dumps(event.detail or {}, ensure_ascii=True, default=str),
            "created_at": created_at,
        }
        with self._engine.begin() as conn:
            conn.execute(self._table.insert().values(**payload))


def hash_actor(api_key: str | None) -> str:
    if not api_key:
        return "anonymous"
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return digest[:12]
