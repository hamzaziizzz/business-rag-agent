from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.rag.types import Document


class DBLoaderError(RuntimeError):
    pass


@dataclass(frozen=True)
class DBIngestConfig:
    connection_uri: str
    query: str
    params: dict[str, Any]
    limit: int | None
    source_name: str
    source_type: str = "db"


@dataclass(frozen=True)
class DBTableIngestConfig:
    connection_uri: str
    table: str
    columns: list[str]
    filters: dict[str, Any]
    limit: int | None
    source_name: str
    allowed_tables: set[str]
    allowed_columns: set[str]
    source_type: str = "db"


def load_db_documents(config: DBIngestConfig) -> Iterable[Document]:
    engine = create_engine(config.connection_uri)
    return _fetch_documents(engine, config)


def load_db_table_documents(config: DBTableIngestConfig) -> Iterable[Document]:
    query, params = build_select_query(
        table=config.table,
        columns=config.columns,
        filters=config.filters,
        limit=config.limit,
        allowed_tables=config.allowed_tables,
        allowed_columns=config.allowed_columns,
    )
    ingest_config = DBIngestConfig(
        connection_uri=config.connection_uri,
        query=query,
        params=params,
        limit=config.limit,
        source_name=config.source_name,
        source_type=config.source_type,
    )
    engine = create_engine(config.connection_uri)
    return _fetch_documents(engine, ingest_config)


def _fetch_documents(engine: Engine, config: DBIngestConfig) -> Iterable[Document]:
    try:
        with engine.connect() as connection:
            result = connection.execute(text(config.query), config.params)
            rows = result.mappings()
            count = 0
            for row in rows:
                count += 1
                if config.limit is not None and count > config.limit:
                    break
                payload = json.dumps(dict(row), ensure_ascii=True, default=str)
                metadata = {
                    "source": f"db:{config.source_name}",
                    "source_type": config.source_type,
                    "source_name": config.source_name,
                    "row_index": count,
                    "query_hash": _hash_text(config.query),
                }
                yield Document(
                    doc_id=f"{config.source_name}-{count}",
                    content=payload,
                    metadata=metadata,
                )
    except Exception as exc:  # pragma: no cover - backend-specific
        raise DBLoaderError(str(exc)) from exc


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def build_select_query(
    table: str,
    columns: list[str],
    filters: dict[str, Any],
    limit: int | None,
    allowed_tables: set[str],
    allowed_columns: set[str],
) -> tuple[str, dict[str, Any]]:
    table_name = _validate_identifier(table)
    if allowed_tables and table_name.lower() not in allowed_tables:
        raise DBLoaderError("Table not allowed for ingestion")

    if not columns:
        raise DBLoaderError("At least one column must be specified")

    normalized_columns = [_validate_identifier(column) for column in columns]
    if allowed_columns:
        for column in normalized_columns:
            if column.lower() not in allowed_columns:
                raise DBLoaderError("Column not allowed for ingestion")

    select_clause = ", ".join(_quote_identifier(column) for column in normalized_columns)
    sql = f"SELECT {select_clause} FROM {_quote_identifier(table_name)}"
    params: dict[str, Any] = {}

    if filters:
        where_clauses: list[str] = []
        for idx, (key, value) in enumerate(filters.items(), start=1):
            column = _validate_identifier(key)
            if allowed_columns and column.lower() not in allowed_columns:
                raise DBLoaderError("Filter column not allowed for ingestion")
            param_key = f"filter_{idx}"
            where_clauses.append(f"{_quote_identifier(column)} = :{param_key}")
            params[param_key] = value
        sql += " WHERE " + " AND ".join(where_clauses)

    if limit is not None:
        sql += " LIMIT :limit"
        params["limit"] = limit

    return sql, params


def _validate_identifier(value: str) -> str:
    if not value or not _IDENTIFIER_RE.match(value):
        raise DBLoaderError("Invalid identifier in schema ingestion")
    return value


def _quote_identifier(value: str) -> str:
    return f"\"{value}\""


def _hash_text(text_value: str) -> str:
    return hashlib.sha256(text_value.encode("utf-8")).hexdigest()
