from __future__ import annotations

"""Pydantic request/response schemas for the API."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request payload for query endpoint."""
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    source_types: list[str] | None = None
    source_names: list[str] | None = None
    source_filter_mode: Literal["and", "or"] = "and"
    route: Literal["rag", "summarize"] | None = None
    trace_id: str | None = None


class SourceChunk(BaseModel):
    """Chunk returned with relevance score and metadata."""
    document_id: str
    content: str
    metadata: dict[str, Any]
    score: float
    highlights: list[str] = Field(default_factory=list)


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
    route: str
    request_id: str
    answerer: str | None = None
    answerer_reason: str | None = None
    structured: dict[str, Any] | None = None
    citations: list[Citation] = Field(default_factory=list)


class ChatMessage(BaseModel):
    """Single chat message in a conversation."""
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    """Request payload for chat endpoint."""
    messages: list[ChatMessage]
    top_k: int | None = Field(default=None, ge=1, le=20)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    source_types: list[str] | None = None
    source_names: list[str] | None = None
    source_filter_mode: Literal["and", "or"] = "and"
    trace_id: str | None = None


class ChatResponse(BaseModel):
    """Response payload for chat endpoint."""
    answer: str
    sources: list[SourceChunk]
    refusal_reason: str | None = None
    route: str
    request_id: str
    answerer: str | None = None
    answerer_reason: str | None = None
    structured: dict[str, Any] | None = None
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


class StatsResponse(BaseModel):
    """Vector store stats response."""
    backend: str
    document_count: int
    embedding_dimension: int
    collection: str | None = None


class StatsHealthResponse(BaseModel):
    """Vector store health response."""
    backend: str
    ok: bool
    detail: str | None = None
    collection: str | None = None


class EmbeddingHealthResponse(BaseModel):
    """Embedding configuration health response."""
    provider: str
    model: str | None
    configured_dimension: int
    expected_dimension: int | None = None
    ok: bool
    status: str
    detail: str | None = None
    action: str | None = None



class DeleteSourceRequest(BaseModel):
    """Request payload for source deletion."""
    source_types: list[str] | None = None
    source_names: list[str] | None = None
    source_filter_mode: Literal["and", "or"] = "and"


class DeleteSourceResponse(BaseModel):
    """Response payload for source deletion."""
    deleted: int
