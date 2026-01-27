from __future__ import annotations

import pytest

from src.rag.sql import SQLValidationError, validate_select_query


def test_validate_select_query_allows_simple_select() -> None:
    plan = validate_select_query(
        "SELECT id, name FROM users WHERE name = 'Ada' LIMIT 10",
        allowed_tables={"users"},
        allowed_columns={"id", "name"},
        max_rows=100,
    )
    assert plan.table == "users"
    assert plan.limit == 10


def test_validate_select_query_rejects_multi_statement() -> None:
    with pytest.raises(SQLValidationError):
        validate_select_query(
            "SELECT * FROM users; DROP TABLE users;",
            allowed_tables=set(),
            allowed_columns=set(),
            max_rows=100,
        )
