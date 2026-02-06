from __future__ import annotations

"""Core data types for documents and retrieval."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """Document chunk with metadata."""
    doc_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """Search result with similarity score."""
    document: Document
    score: float


@dataclass(frozen=True)
class ContextChunk:
    """Context chunk selected for answering."""
    document_id: str
    content: str
    metadata: dict[str, Any]
    score: float
