from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    source_types: list[str] | None = None
    source_names: list[str] | None = None
    source_filter_mode: Literal["and", "or"] = "and"
    route: Literal["rag", "sql", "summarize"] | None = None
    trace_id: str | None = None
    sql_query: str | None = None
    sql_params: dict[str, Any] = Field(default_factory=dict)


class SourceChunk(BaseModel):
    document_id: str
    content: str
    metadata: dict[str, Any]
    score: float
    highlights: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    refusal_reason: str | None = None
    route: str
    request_id: str
    structured: dict[str, Any] | None = None


class IngestDocument(BaseModel):
    doc_id: str | None = None
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[IngestDocument]


class IngestResponse(BaseModel):
    ingested: int


class StatsResponse(BaseModel):
    backend: str
    document_count: int
    embedding_dimension: int
    collection: str | None = None


class StatsHealthResponse(BaseModel):
    backend: str
    ok: bool
    detail: str | None = None
    collection: str | None = None


class EmbeddingHealthResponse(BaseModel):
    provider: str
    model: str | None
    configured_dimension: int
    expected_dimension: int | None = None
    ok: bool
    status: str
    detail: str | None = None
    action: str | None = None


class DBIngestRequest(BaseModel):
    connection_uri: str = Field(min_length=1)
    query: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    limit: int | None = Field(default=None, ge=1, le=10000)
    source_name: str | None = None


class DBTableIngestRequest(BaseModel):
    connection_uri: str = Field(min_length=1)
    table: str = Field(min_length=1)
    columns: list[str] = Field(min_length=1)
    filters: dict[str, Any] = Field(default_factory=dict)
    limit: int | None = Field(default=None, ge=1, le=10000)
    source_name: str | None = None


class APIIngestRequest(BaseModel):
    url: str = Field(min_length=1)
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "GET"
    headers: dict[str, str] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    json_body: Any | None = None
    timeout: float | None = Field(default=None, ge=1, le=60)
    max_bytes: int | None = Field(default=None, ge=1024, le=10_485_760)
    source_name: str | None = None


class ObjectIngestRequest(BaseModel):
    bucket: str = Field(min_length=1)
    key: str = Field(min_length=1)
    source_name: str | None = None
    endpoint_url: str | None = None
    region: str | None = None
    access_key: str | None = None
    secret_key: str | None = None
    session_token: str | None = None
    max_bytes: int | None = Field(default=None, ge=1024)


class DeleteSourceRequest(BaseModel):
    source_types: list[str] | None = None
    source_names: list[str] | None = None
    source_filter_mode: Literal["and", "or"] = "and"


class DeleteSourceResponse(BaseModel):
    deleted: int
