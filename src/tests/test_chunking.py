from __future__ import annotations

"""Chunking behavior tests."""

from src.loaders.chunking import chunk_document
from src.rag.types import Document


def test_chunk_document_splits_and_adds_metadata() -> None:
    """Ensure chunking splits and annotates metadata."""
    content = "word " * 300
    document = Document(doc_id="policy", content=content, metadata={"source": "policy.txt"})

    chunks = chunk_document(document, max_chars=200, overlap=20)

    assert len(chunks) > 1
    assert chunks[0].doc_id == "policy-1"
    assert chunks[0].metadata["chunk_index"] == 1
    assert chunks[0].metadata["chunk_count"] == len(chunks)
