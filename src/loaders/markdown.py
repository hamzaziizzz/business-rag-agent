from __future__ import annotations

from pathlib import Path

from src.loaders.text import load_text_bytes, load_text_file
from src.rag.types import Document


def load_markdown_file(path: Path, doc_id: str | None = None) -> Document:
    return load_text_file(path, doc_id=doc_id)


def load_markdown_bytes(data: bytes, doc_id: str, source: str) -> Document:
    return load_text_bytes(data, doc_id=doc_id, source=source)
