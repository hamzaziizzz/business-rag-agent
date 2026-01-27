from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import create_engine, text


class SQLValidationError(RuntimeError):
    pass


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SELECT_RE = re.compile(
    r"^select\s+(?P<columns>.+?)\s+from\s+(?P<table>\w+)"
    r"(?:\s+where\s+(?P<where>.+?))?"
    r"(?:\s+limit\s+(?P<limit>\d+))?$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class SQLQueryPlan:
    sql: str
    params: dict[str, Any]
    table: str
    columns: list[str]
    limit: int | None


def validate_select_query(
    query: str,
    allowed_tables: set[str],
    allowed_columns: set[str],
    max_rows: int,
) -> SQLQueryPlan:
    if not allowed_tables or not allowed_columns:
        raise SQLValidationError("SQL allow-lists must be configured")
    normalized = query.strip()
    if not normalized:
        raise SQLValidationError("SQL query is empty")
    lowered = normalized.lower()
    if ";" in normalized or "--" in lowered or "/*" in lowered or "*/" in lowered:
        raise SQLValidationError("Only single SELECT statements are allowed")
    match = _SELECT_RE.match(normalized)
    if not match:
        raise SQLValidationError("Only simple SELECT queries are allowed")

    table = match.group("table")
    if not _IDENTIFIER.match(table):
        raise SQLValidationError("Invalid table name")
    if allowed_tables and table.lower() not in allowed_tables:
        raise SQLValidationError("Table not allowed")

    columns_raw = match.group("columns")
    columns = _parse_columns(columns_raw)
    if not columns:
        raise SQLValidationError("No columns selected")
    if allowed_columns:
        if "*" in columns:
            raise SQLValidationError("Wildcard column not allowed")
        for column in columns:
            if column.lower() not in allowed_columns:
                raise SQLValidationError("Column not allowed")

    where_raw = match.group("where")
    params: dict[str, Any] = {}
    if where_raw:
        _validate_where(where_raw, allowed_columns)

    limit_raw = match.group("limit")
    limit = int(limit_raw) if limit_raw else None
    if limit is not None and limit > max_rows:
        limit = max_rows
    sql = _rebuild_query(normalized, limit, max_rows)
    return SQLQueryPlan(sql=sql, params=params, table=table, columns=columns, limit=limit)


def execute_select_query(
    connection_uri: str, plan: SQLQueryPlan, params: dict[str, Any], max_rows: int
) -> list[dict[str, Any]]:
    engine = create_engine(connection_uri)
    with engine.connect() as conn:
        result = conn.execute(text(plan.sql), dict(params))
        rows = result.mappings().fetchall()
        return [dict(row) for row in rows]


def _parse_columns(columns_raw: str) -> list[str]:
    columns: list[str] = []
    for item in columns_raw.split(","):
        token = item.strip()
        if not token:
            continue
        if token == "*":
            columns.append("*")
            continue
        token = token.split(" as ", 1)[0].split(" AS ", 1)[0].strip()
        token = token.split(" ", 1)[0]
        if "." in token:
            token = token.split(".", 1)[1]
        if not _IDENTIFIER.match(token):
            raise SQLValidationError("Invalid column name")
        columns.append(token)
    return columns


def _validate_where(where_raw: str, allowed_columns: set[str]) -> None:
    clauses = re.split(r"\s+and\s+", where_raw, flags=re.IGNORECASE)
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        match = re.match(r"^(\w+)\s*=\s*(:\w+|\d+|'.*?')$", clause)
        if not match:
            raise SQLValidationError("Only equality filters are allowed")
        column = match.group(1)
        if not _IDENTIFIER.match(column):
            raise SQLValidationError("Invalid filter column")
        if allowed_columns and column.lower() not in allowed_columns:
            raise SQLValidationError("Filter column not allowed")


def _rebuild_query(original: str, limit: int | None, max_rows: int) -> str:
    sql = original.strip()
    capped = max_rows if limit is None else min(limit, max_rows)
    if re.search(r"\blimit\s+\d+\b", sql, flags=re.IGNORECASE):
        return re.sub(r"\blimit\s+\d+\b", f"LIMIT {capped}", sql, flags=re.IGNORECASE)
    return f"{sql} LIMIT {capped}"
