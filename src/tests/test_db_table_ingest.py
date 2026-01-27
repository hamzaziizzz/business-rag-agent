from __future__ import annotations

import sqlite3

from src.loaders.db import DBTableIngestConfig, build_select_query, load_db_table_documents


def test_build_select_query_respects_allowlist() -> None:
    query, params = build_select_query(
        table="users",
        columns=["id", "name"],
        filters={"name": "Ada"},
        limit=5,
        allowed_tables={"users"},
        allowed_columns={"id", "name"},
    )

    assert "SELECT" in query
    assert "FROM" in query
    assert "WHERE" in query
    assert params["filter_1"] == "Ada"


def test_load_db_table_documents_from_sqlite(tmp_path) -> None:
    db_path = tmp_path / "table.db"
    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.cursor()
        cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        cursor.executemany("INSERT INTO users (name) VALUES (?)", [("Ada",), ("Ben",)])
        connection.commit()
    finally:
        connection.close()

    config = DBTableIngestConfig(
        connection_uri=f"sqlite:///{db_path}",
        table="users",
        columns=["id", "name"],
        filters={},
        limit=10,
        source_name="users",
        allowed_tables=set(),
        allowed_columns=set(),
        source_type="db",
    )

    docs = list(load_db_table_documents(config))

    assert len(docs) == 2
    assert docs[0].metadata["source_type"] == "db"
