from __future__ import annotations

from src.rag.types import Document


class CSVLoaderError(RuntimeError):
    pass


def load_csv_bytes(data: bytes, doc_id: str, source: str) -> Document:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
    content = text.strip()
    return Document(doc_id=doc_id, content=content, metadata={"source": source})
