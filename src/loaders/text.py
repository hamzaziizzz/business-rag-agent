from __future__ import annotations

"""Plain text loader for ingestion."""

from pathlib import Path

from src.rag.types import Document


def load_text_file(path: Path, doc_id: str | None = None) -> Document:
    """Load a text file from disk into a Document."""
    content = path.read_text(encoding="utf-8")
    return Document(
        doc_id=doc_id or path.stem,
        content=content,
        metadata={"source": str(path)},
    )


def load_text_bytes(data: bytes, doc_id: str, source: str) -> Document:
    """Load plain text bytes into a Document."""
    content = data.decode("utf-8", errors="ignore")
    return Document(doc_id=doc_id, content=content, metadata={"source": source})
