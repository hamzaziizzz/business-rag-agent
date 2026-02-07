from __future__ import annotations

"""Pydantic request/response schemas for the API."""

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request payload for query endpoint."""
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)


class SourceChunk(BaseModel):
    """Chunk returned with relevance score and metadata."""
    document_id: str
    content: str
    metadata: dict[str, Any]
    score: float


class Citation(BaseModel):
    """Citation metadata for a returned chunk."""
    label: str
    document_id: str
    source_type: str
    source_name: str


class QueryResponse(BaseModel):
    """Response payload for query endpoint."""
    answer: str
    sources: list[SourceChunk]
    refusal_reason: str | None = None
    request_id: str
    citations: list[Citation] = Field(default_factory=list)


class IngestDocument(BaseModel):
    """Single document payload for ingest."""
    doc_id: str | None = None
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    """Request payload for text ingest."""
    documents: list[IngestDocument]


class IngestResponse(BaseModel):
    """Response payload for ingest requests."""
    ingested: int

