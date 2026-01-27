from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    doc_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    document: Document
    score: float


@dataclass(frozen=True)
class ContextChunk:
    document_id: str
    content: str
    metadata: dict[str, Any]
    score: float
