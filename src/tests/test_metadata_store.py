from __future__ import annotations

import sqlite3

from src.metadata.store import IngestionMeta, MetadataStore


def test_metadata_store_records_lifecycle(tmp_path) -> None:
    db_path = tmp_path / "meta.db"
    store = MetadataStore(f"sqlite:///{db_path}")
    meta = IngestionMeta(
        source_type="db",
        source_name="unit_test",
        source_uri="postgresql://user:***@localhost/db",
        status="started",
        extra={"note": "test"},
    )
    record_id = store.record_start(meta)
    store.record_complete(record_id, ingested=5, chunk_count=5)

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT status, ingested_count, chunk_count FROM ingestion_records WHERE id = ?",
            (record_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row == ("completed", 5, 5)
