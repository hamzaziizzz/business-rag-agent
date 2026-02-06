from __future__ import annotations

"""Text normalization and chunking utilities (char and token based)."""

import re

from src.rag.types import Document

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize whitespace and line endings in text."""
    return _WHITESPACE_RE.sub(" ", text.replace("\r\n", "\n")).strip()


def normalize_query_tokens(text: str, encoding_name: str = "cl100k_base") -> str:
    """Normalize a query via tokenizer round-trip for consistent retrieval."""
    cleaned = normalize_text(text)
    if not cleaned:
        return ""
    try:
        import tiktoken
    except ImportError:  # pragma: no cover - optional dependency
        return cleaned
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception:  # pragma: no cover - fallback to default encoding
        encoding = tiktoken.get_encoding("cl100k_base")
    return encoding.decode(encoding.encode(cleaned)).strip()


def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split text into overlapping character-based chunks."""
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


def chunk_text_tokens(
    text: str,
    max_tokens: int,
    overlap: int,
    encoding_name: str = "cl100k_base",
) -> list[str]:
    """Split text into overlapping token-based chunks."""
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    if max_tokens <= 0:
        return [cleaned]
    if overlap >= max_tokens:
        overlap = max(0, max_tokens // 4)
    step = max_tokens - overlap
    if step <= 0:
        return [cleaned]
    try:
        import tiktoken
    except ImportError:  # pragma: no cover - optional dependency
        return chunk_text(cleaned, max_chars=1000, overlap=0)
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception:  # pragma: no cover - fall back to default encoding
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(cleaned)
    if not tokens:
        return []
    chunks: list[str] = []
    start = 0
    length = len(tokens)
    while start < length:
        end = min(length, start + max_tokens)
        chunk_tokens = tokens[start:end]
        chunk = encoding.decode(chunk_tokens).strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(0, end - overlap)
    return chunks


def chunk_document(
    document: Document,
    max_chars: int,
    overlap: int,
    max_tokens: int | None = None,
    token_overlap: int | None = None,
    encoding_name: str = "cl100k_base",
) -> list[Document]:
    """Chunk a document into multiple Document records with metadata."""
    resolved_max_tokens = max_tokens if max_tokens and max_tokens > 0 else max_chars
    resolved_overlap = token_overlap if token_overlap is not None else overlap
    chunks = chunk_text_tokens(
        document.content,
        max_tokens=resolved_max_tokens,
        overlap=resolved_overlap,
        encoding_name=encoding_name,
    )
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
