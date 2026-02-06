from __future__ import annotations

"""Audit event storage and hashing utilities."""

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


class AuditStoreError(RuntimeError):
    """Raised when audit storage fails."""
    pass


@dataclass(frozen=True)
class AuditEvent:
    """Audit event payload captured during request processing."""
    event_type: str
    request_id: str
    tenant_id: str
    actor: str
    status: str
    route: str | None = None
    detail: dict[str, Any] | None = None


class AuditStore:
    """Persist audit events to a SQL database."""
    def __init__(self, connection_uri: str) -> None:
        """Initialize the audit store and ensure tables exist."""
        try:
            from sqlalchemy import Column, DateTime, MetaData, String, Table, Text, create_engine
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise AuditStoreError(
                "sqlalchemy is required to use the audit store"
            ) from exc

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
        """Insert a new audit event row."""
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
    """Hash an API key into a short actor token for audit logs."""
    if not api_key:
        return "anonymous"
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return digest[:12]
