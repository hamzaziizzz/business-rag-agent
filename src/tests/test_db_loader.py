from __future__ import annotations

import sqlite3

from src.loaders.db import DBIngestConfig, load_db_documents


def test_load_db_documents_from_sqlite(tmp_path) -> None:
    db_path = tmp_path / "test.db"
    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.cursor()
        cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        cursor.executemany("INSERT INTO users (name) VALUES (?)", [("Ada",), ("Ben",)])
        connection.commit()
    finally:
        connection.close()

    config = DBIngestConfig(
        connection_uri=f"sqlite:///{db_path}",
        query="SELECT id, name FROM users",
        params={},
        limit=10,
        source_name="users",
        source_type="db",
    )

    docs = list(load_db_documents(config))

    assert len(docs) == 2
    assert docs[0].metadata["source_type"] == "db"
    assert "Ada" in docs[0].content or "Ben" in docs[0].content
