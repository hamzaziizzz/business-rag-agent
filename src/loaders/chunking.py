from __future__ import annotations

import re

from src.rag.types import Document

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.replace("\r\n", "\n")).strip()


def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    if max_chars <= 0:
        return [cleaned]
    if overlap >= max_chars:
        overlap = max(0, max_chars // 4)
    step = max_chars - overlap
    if step <= 0:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    length = len(cleaned)
    while start < length:
        end = min(length, start + max_chars)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(0, end - overlap)
    return chunks


def chunk_document(document: Document, max_chars: int, overlap: int) -> list[Document]:
    chunks = chunk_text(document.content, max_chars=max_chars, overlap=overlap)
    if not chunks:
        return []
    if len(chunks) == 1:
        if chunks[0] == document.content:
            return [document]
        return [Document(doc_id=document.doc_id, content=chunks[0], metadata=document.metadata)]

    total = len(chunks)
    documents: list[Document] = []
    for idx, chunk in enumerate(chunks, start=1):
        metadata = dict(document.metadata)
        metadata.update({"chunk_index": idx, "chunk_count": total})
        documents.append(
            Document(
                doc_id=f"{document.doc_id}-{idx}",
                content=chunk,
                metadata=metadata,
            )
        )
    return documents
