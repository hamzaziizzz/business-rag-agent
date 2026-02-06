from __future__ import annotations

"""CSV/TSV loader for plain-text ingestion."""

from src.rag.types import Document


class CSVLoaderError(RuntimeError):
    """Raised when CSV loading fails."""
    pass


def load_csv_bytes(data: bytes, doc_id: str, source: str) -> Document:
    """Load CSV/TSV bytes into a Document."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
    content = text.strip()
    return Document(doc_id=doc_id, content=content, metadata={"source": source})
