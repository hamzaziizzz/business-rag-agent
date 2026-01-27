from __future__ import annotations

import hashlib
import json
import logging
import uuid
from urllib.parse import urlparse, urlunparse
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile

from src.app.dependencies import (
    get_audit_store,
    get_embedding_config_report,
    get_metadata_store,
    get_pipeline,
)
from src.app.settings import settings
from src.app.schemas import (
    APIIngestRequest,
    ObjectIngestRequest,
    DBIngestRequest,
    DBTableIngestRequest,
    DeleteSourceRequest,
    DeleteSourceResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    StatsResponse,
    StatsHealthResponse,
    EmbeddingHealthResponse,
)
from src.app.security import AuthContext, require_api_key, require_roles, resolve_tenant_id
from src.app.metrics import metrics_middleware, metrics_response
from src.loaders.chunking import chunk_document
from src.loaders.db import (
    DBIngestConfig,
    DBLoaderError,
    DBTableIngestConfig,
    load_db_documents,
    load_db_table_documents,
)
from src.loaders.api import APIIngestConfig, APILoaderError, fetch_api_documents
from src.loaders.object_store import ObjectStoreConfig, ObjectStoreError, load_object_document
from src.loaders.markdown import load_markdown_bytes
from src.loaders.docx import DocxLoaderError, load_docx_bytes
from src.loaders.pdf import PDFLoaderError, load_pdf_bytes
from src.loaders.text import load_text_bytes
from src.loaders.xlsx import XlsxLoaderError, load_xlsx_bytes
from src.loaders.csv_loader import CSVLoaderError, load_csv_bytes
from src.rag.answerer import ExtractiveAnswerer, SummarizingAnswerer
from src.rag.formatters import extract_db_records, format_db_answer
from src.rag.guardrails import DEFAULT_REFUSAL, require_context
from src.rag.highlights import build_highlights
from src.rag.llm import LLMError, OllamaAnswerer
from src.rag.sql import SQLValidationError, execute_select_query, validate_select_query
from src.rag.types import ContextChunk, Document
from src.agents.router import AgentRouter
from src.metadata.store import IngestionMeta, MetadataStore
from src.metadata.audit import AuditEvent, hash_actor

logger = logging.getLogger(__name__)
router = AgentRouter()

app = FastAPI(title="Business RAG Agent", version="0.1.0")


def _sanitize_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme:
        return url
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def _record_audit_event(event: AuditEvent) -> None:
    audit_store = get_audit_store()
    if not audit_store:
        return
    audit_store.record_event(event)


def _safe_error_message(exc: Exception) -> str:
    return type(exc).__name__


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    return await metrics_middleware(request, call_next)


@app.get("/metrics")
async def metrics():
    return metrics_response()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/stats", response_model=StatsResponse)
async def stats(auth: AuthContext = Depends(require_api_key)) -> StatsResponse:
    require_roles(auth, {"admin"})
    pipeline = get_pipeline()
    return StatsResponse(**pipeline.vectorstore.stats())


@app.get("/stats/health", response_model=StatsHealthResponse)
async def stats_health(auth: AuthContext = Depends(require_api_key)) -> StatsHealthResponse:
    require_roles(auth, {"admin"})
    pipeline = get_pipeline()
    return StatsHealthResponse(**pipeline.vectorstore.health())


@app.get("/stats/embedding", response_model=EmbeddingHealthResponse)
async def embedding_health(
    auth: AuthContext = Depends(require_api_key),
) -> EmbeddingHealthResponse:
    require_roles(auth, {"admin"})
    report = get_embedding_config_report()
    return EmbeddingHealthResponse(**report.__dict__)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    http_request: Request,
    auth: AuthContext = Depends(require_api_key),
) -> IngestResponse:
    require_roles(auth, {"writer", "admin"})
    pipeline = get_pipeline()
    store = get_metadata_store()
    tenant_id = resolve_tenant_id(http_request, auth)
    request_id = getattr(http_request.state, "request_id", str(uuid.uuid4()))
    record_id = None
    if store:
        meta = IngestionMeta(
            source_type="manual",
            source_name=f"{tenant_id}:manual",
            source_uri="manual",
            status="started",
            extra={"doc_count": len(request.documents)},
        )
        record_id = store.record_start(meta)
    documents: list[Document] = []
    for idx, doc in enumerate(request.documents, start=1):
        metadata = dict(doc.metadata)
        metadata.setdefault("source_type", "manual")
        metadata.setdefault("source_name", "manual")
        metadata.setdefault("tenant_id", tenant_id)
        document = Document(
            doc_id=doc.doc_id or f"doc-{idx}",
            content=doc.content,
            metadata=metadata,
        )
        documents.extend(
            chunk_document(
                document,
                max_chars=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )
        )
    if not documents:
        _record_audit_event(
            AuditEvent(
                event_type="ingest",
                request_id=request_id,
                tenant_id=tenant_id,
                actor=hash_actor(auth.api_key),
                status="failed",
                route=None,
                detail={"source_type": "manual", "source_name": "manual", "reason": "empty"},
            )
        )
        if store and record_id:
            store.record_failure(record_id, "No documents provided")
        raise HTTPException(status_code=400, detail="No documents provided")
    ingested = pipeline.ingest(documents)
    _record_audit_event(
        AuditEvent(
            event_type="ingest",
            request_id=request_id,
            tenant_id=tenant_id,
            actor=hash_actor(auth.api_key),
            status="completed",
            route=None,
            detail={
                "source_type": "manual",
                "source_name": "manual",
                "ingested": ingested,
                "chunk_count": len(documents),
            },
        )
    )
    if store and record_id:
        store.record_complete(record_id, ingested=ingested, chunk_count=len(documents))
    return IngestResponse(ingested=ingested)


@app.post("/ingest/db", response_model=IngestResponse)
async def ingest_db(
    request: DBIngestRequest,
    http_request: Request,
    auth: AuthContext = Depends(require_api_key),
) -> IngestResponse:
    require_roles(auth, {"writer", "admin"})
    request_id = getattr(http_request.state, "request_id", str(uuid.uuid4()))
    pipeline = get_pipeline()
    store = get_metadata_store()
    tenant_id = resolve_tenant_id(http_request, auth)
    source_name = request.source_name or "database"
    record_id = None
    if store:
        meta = IngestionMeta(
            source_type="db",
            source_name=source_name,
            source_uri=MetadataStore.redact_uri(request.connection_uri),
            status="started",
            extra={
                "query_hash": hashlib.sha256(request.query.encode("utf-8")).hexdigest(),
                "limit": request.limit or settings.db_max_rows,
            },
        )
        record_id = store.record_start(meta)
    config = DBIngestConfig(
        connection_uri=request.connection_uri,
        query=request.query,
        params=request.params,
        limit=request.limit or settings.db_max_rows,
        source_name=source_name,
        source_type="db",
    )
    try:
        documents = []
        for document in load_db_documents(config):
            document.metadata.setdefault("tenant_id", tenant_id)
            documents.extend(
                chunk_document(
                    document,
                    max_chars=settings.db_chunk_size,
                    overlap=settings.db_chunk_overlap,
                )
            )
    except DBLoaderError as exc:
        _record_audit_event(
            AuditEvent(
                event_type="ingest",
                request_id=request_id,
                tenant_id=tenant_id,
                actor=hash_actor(auth.api_key),
                status="failed",
                route=None,
                detail={
                    "source_type": "db",
                    "source_name": source_name,
                    "error": _safe_error_message(exc),
                },
            )
        )
        if store and record_id:
            store.record_failure(record_id, _safe_error_message(exc))
        logger.error(
            "db_ingest_failed",
            extra={
                "request_id": request_id,
                "source_name": source_name,
                "detail": _safe_error_message(exc),
            },
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not documents:
        _record_audit_event(
            AuditEvent(
                event_type="ingest",
                request_id=request_id,
                tenant_id=tenant_id,
                actor=hash_actor(auth.api_key),
                status="failed",
                route=None,
                detail={"source_type": "db", "source_name": source_name, "reason": "empty"},
            )
        )
        if store and record_id:
            store.record_failure(record_id, "No database rows returned")
        logger.info(
            "db_ingest_empty",
            extra={"request_id": request_id, "source_name": source_name},
        )
        raise HTTPException(status_code=400, detail="No database rows returned")
    ingested = pipeline.ingest(documents)
    _record_audit_event(
        AuditEvent(
            event_type="ingest",
            request_id=request_id,
            tenant_id=tenant_id,
            actor=hash_actor(auth.api_key),
            status="completed",
            route=None,
            detail={
                "source_type": "db",
                "source_name": source_name,
                "ingested": ingested,
                "chunk_count": len(documents),
            },
        )
    )
    if store and record_id:
        store.record_complete(record_id, ingested=ingested, chunk_count=len(documents))
    logger.info(
        "db_ingest_complete",
        extra={"request_id": request_id, "source_name": source_name, "ingested": ingested},
    )
    return IngestResponse(ingested=ingested)


@app.post("/ingest/db/table", response_model=IngestResponse)
async def ingest_db_table(
    request: DBTableIngestRequest,
    http_request: Request,
    auth: AuthContext = Depends(require_api_key),
) -> IngestResponse:
    require_roles(auth, {"writer", "admin"})
    request_id = getattr(http_request.state, "request_id", str(uuid.uuid4()))
    pipeline = get_pipeline()
    store = get_metadata_store()
    tenant_id = resolve_tenant_id(http_request, auth)
    source_name = request.source_name or request.table
    record_id = None
    if store:
        meta = IngestionMeta(
            source_type="db",
            source_name=source_name,
            source_uri=MetadataStore.redact_uri(request.connection_uri),
            status="started",
            extra={
                "table": request.table,
                "columns": request.columns,
                "filters": list(request.filters.keys()),
                "limit": request.limit or settings.db_max_rows,
            },
        )
        record_id = store.record_start(meta)
    config = DBTableIngestConfig(
        connection_uri=request.connection_uri,
        table=request.table,
        columns=request.columns,
        filters=request.filters,
        limit=request.limit or settings.db_max_rows,
        source_name=source_name,
        allowed_tables=settings.db_allowed_tables,
        allowed_columns=settings.db_allowed_columns,
        source_type="db",
    )
    try:
        documents = []
        for document in load_db_table_documents(config):
            document.metadata.setdefault("tenant_id", tenant_id)
            documents.extend(
                chunk_document(
                    document,
                    max_chars=settings.db_chunk_size,
                    overlap=settings.db_chunk_overlap,
                )
            )
    except DBLoaderError as exc:
        _record_audit_event(
            AuditEvent(
                event_type="ingest",
                request_id=request_id,
                tenant_id=tenant_id,
                actor=hash_actor(auth.api_key),
                status="failed",
                route=None,
                detail={
                    "source_type": "db",
                    "source_name": source_name,
                    "error": _safe_error_message(exc),
                },
            )
        )
        if store and record_id:
            store.record_failure(record_id, _safe_error_message(exc))
        logger.error(
            "db_table_ingest_failed",
            extra={
                "request_id": request_id,
                "source_name": source_name,
                "detail": _safe_error_message(exc),
            },
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not documents:
        _record_audit_event(
            AuditEvent(
                event_type="ingest",
                request_id=request_id,
                tenant_id=tenant_id,
                actor=hash_actor(auth.api_key),
                status="failed",
                route=None,
                detail={"source_type": "db", "source_name": source_name, "reason": "empty"},
            )
        )
        if store and record_id:
            store.record_failure(record_id, "No database rows returned")
        logger.info(
            "db_table_ingest_empty",
            extra={"request_id": request_id, "source_name": source_name},
        )
        raise HTTPException(status_code=400, detail="No database rows returned")
    ingested = pipeline.ingest(documents)
    _record_audit_event(
        AuditEvent(
            event_type="ingest",
            request_id=request_id,
            tenant_id=tenant_id,
            actor=hash_actor(auth.api_key),
            status="completed",
            route=None,
            detail={
                "source_type": "db",
                "source_name": source_name,
                "ingested": ingested,
                "chunk_count": len(documents),
            },
        )
    )
    if store and record_id:
        store.record_complete(record_id, ingested=ingested, chunk_count=len(documents))
    logger.info(
        "db_table_ingest_complete",
        extra={"request_id": request_id, "source_name": source_name, "ingested": ingested},
    )
    return IngestResponse(ingested=ingested)


@app.post("/ingest/api", response_model=IngestResponse)
async def ingest_api(
    request: APIIngestRequest,
    http_request: Request,
    auth: AuthContext = Depends(require_api_key),
) -> IngestResponse:
    require_roles(auth, {"writer", "admin"})
    request_id = getattr(http_request.state, "request_id", str(uuid.uuid4()))
    pipeline = get_pipeline()
    store = get_metadata_store()
    tenant_id = resolve_tenant_id(http_request, auth)
    source_name = request.source_name or "api"
    record_id = None
    if store:
        meta = IngestionMeta(
            source_type="api",
            source_name=source_name,
            source_uri=_sanitize_url(request.url),
            status="started",
            extra={
                "method": request.method,
                "limit": request.max_bytes or settings.api_max_bytes,
            },
        )
        record_id = store.record_start(meta)
    config = APIIngestConfig(
        url=request.url,
        method=request.method,
        headers=request.headers,
        params=request.params,
        json_body=request.json_body,
        timeout=request.timeout or settings.api_timeout,
        max_bytes=request.max_bytes or settings.api_max_bytes,
        source_name=source_name,
        source_type="api",
    )
    try:
        documents = []
        raw_documents = await fetch_api_documents(config)
        for document in raw_documents:
            document.metadata.setdefault("tenant_id", tenant_id)
            documents.extend(
                chunk_document(
                    document,
                    max_chars=settings.chunk_size,
                    overlap=settings.chunk_overlap,
                )
            )
    except APILoaderError as exc:
        _record_audit_event(
            AuditEvent(
                event_type="ingest",
                request_id=request_id,
                tenant_id=tenant_id,
                actor=hash_actor(auth.api_key),
                status="failed",
                route=None,
                detail={
                    "source_type": "api",
                    "source_name": source_name,
                    "error": _safe_error_message(exc),
                },
            )
        )
        if store and record_id:
            store.record_failure(record_id, _safe_error_message(exc))
        logger.error(
            "api_ingest_failed",
            extra={
                "request_id": request_id,
                "source_name": source_name,
                "detail": _safe_error_message(exc),
            },
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not documents:
        _record_audit_event(
            AuditEvent(
                event_type="ingest",
                request_id=request_id,
                tenant_id=tenant_id,
                actor=hash_actor(auth.api_key),
                status="failed",
                route=None,
                detail={"source_type": "api", "source_name": source_name, "reason": "empty"},
            )
        )
        if store and record_id:
            store.record_failure(record_id, "No API content returned")
        logger.info(
            "api_ingest_empty",
            extra={"request_id": request_id, "source_name": source_name},
        )
        raise HTTPException(status_code=400, detail="No API content returned")
    ingested = pipeline.ingest(documents)
    _record_audit_event(
        AuditEvent(
            event_type="ingest",
            request_id=request_id,
            tenant_id=tenant_id,
            actor=hash_actor(auth.api_key),
            status="completed",
            route=None,
            detail={
                "source_type": "api",
                "source_name": source_name,
                "ingested": ingested,
                "chunk_count": len(documents),
            },
        )
    )
    if store and record_id:
        store.record_complete(record_id, ingested=ingested, chunk_count=len(documents))
    logger.info(
        "api_ingest_complete",
        extra={"request_id": request_id, "source_name": source_name, "ingested": ingested},
    )
    return IngestResponse(ingested=ingested)


@app.post("/ingest/object", response_model=IngestResponse)
async def ingest_object(
    request: ObjectIngestRequest,
    http_request: Request,
    auth: AuthContext = Depends(require_api_key),
) -> IngestResponse:
    require_roles(auth, {"writer", "admin"})
    request_id = getattr(http_request.state, "request_id", str(uuid.uuid4()))
    pipeline = get_pipeline()
    store = get_metadata_store()
    tenant_id = resolve_tenant_id(http_request, auth)
    bucket = request.bucket or settings.object_store_bucket
    if not bucket:
        raise HTTPException(status_code=400, detail="Bucket is required")
    key = request.key
    source_name = request.source_name or Path(key).stem
    source = f"s3://{bucket}/{key}"
    doc_id = f"{Path(key).stem}-{request_id[:8]}"
    suffix = Path(key).suffix.lower()
    if suffix in {".txt", ".text"}:
        loader = load_text_bytes
        source_type = "text"
    elif suffix in {".md", ".markdown"}:
        loader = load_markdown_bytes
        source_type = "markdown"
    elif suffix == ".pdf":
        loader = load_pdf_bytes
        source_type = "pdf"
    elif suffix == ".docx":
        loader = load_docx_bytes
        source_type = "docx"
    elif suffix == ".xlsx":
        loader = load_xlsx_bytes
        source_type = "xlsx"
    elif suffix in {".csv", ".tsv"}:
        loader = load_csv_bytes
        source_type = "csv"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported object type: {suffix}")

    config = ObjectStoreConfig(
        bucket=bucket,
        key=key,
        endpoint_url=request.endpoint_url or settings.object_store_endpoint_url,
        region=request.region or settings.object_store_region,
        access_key=request.access_key or settings.object_store_access_key,
        secret_key=request.secret_key or settings.object_store_secret_key,
        session_token=request.session_token or settings.object_store_session_token,
        max_bytes=request.max_bytes or settings.object_store_max_bytes,
    )
    record_id = None
    if store:
        meta = IngestionMeta(
            source_type="object",
            source_name=source_name,
            source_uri=source,
            status="started",
            extra={"bucket": bucket, "key": key, "max_bytes": config.max_bytes},
        )
        record_id = store.record_start(meta)
    try:
        document = load_object_document(
            config=config,
            loader=loader,
            doc_id=doc_id,
            source=source,
        )
    except (ObjectStoreError, PDFLoaderError, DocxLoaderError, XlsxLoaderError, CSVLoaderError) as exc:
        _record_audit_event(
            AuditEvent(
                event_type="ingest",
                request_id=request_id,
                tenant_id=tenant_id,
                actor=hash_actor(auth.api_key),
                status="failed",
                route=None,
                detail={
                    "source_type": "object",
                    "source_name": source_name,
                    "error": _safe_error_message(exc),
                },
            )
        )
        if store and record_id:
            store.record_failure(record_id, _safe_error_message(exc))
        logger.error(
            "object_ingest_failed",
            extra={
                "request_id": request_id,
                "source_name": source_name,
                "detail": _safe_error_message(exc),
            },
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    document.metadata["source_type"] = source_type
    document.metadata.setdefault("source_name", source_name)
    document.metadata.setdefault("tenant_id", tenant_id)
    chunks = chunk_document(
        document,
        max_chars=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    if not chunks:
        _record_audit_event(
            AuditEvent(
                event_type="ingest",
                request_id=request_id,
                tenant_id=tenant_id,
                actor=hash_actor(auth.api_key),
                status="failed",
                route=None,
                detail={"source_type": "object", "source_name": source_name, "reason": "empty"},
            )
        )
        if store and record_id:
            store.record_failure(record_id, "No object content to ingest")
        raise HTTPException(status_code=400, detail="No object content to ingest")
    ingested = pipeline.ingest(chunks)
    _record_audit_event(
        AuditEvent(
            event_type="ingest",
            request_id=request_id,
            tenant_id=tenant_id,
            actor=hash_actor(auth.api_key),
            status="completed",
            route=None,
            detail={
                "source_type": "object",
                "source_name": source_name,
                "ingested": ingested,
                "chunk_count": len(chunks),
            },
        )
    )
    if store and record_id:
        store.record_complete(record_id, ingested=ingested, chunk_count=len(chunks))
    logger.info(
        "object_ingest_complete",
        extra={"request_id": request_id, "source_name": source_name, "ingested": ingested},
    )
    return IngestResponse(ingested=ingested)


@app.post("/ingest/delete", response_model=DeleteSourceResponse)
async def delete_sources(
    request: DeleteSourceRequest,
    http_request: Request,
    auth: AuthContext = Depends(require_api_key),
) -> DeleteSourceResponse:
    require_roles(auth, {"admin"})
    if not (request.source_types or request.source_names):
        raise HTTPException(status_code=400, detail="source_types or source_names required")
    pipeline = get_pipeline()
    tenant_id = resolve_tenant_id(http_request, auth)
    deleted = pipeline.delete_by_source(
        source_types=request.source_types,
        source_names=request.source_names,
        tenant_id=tenant_id,
        source_filter_mode=request.source_filter_mode,
    )
    _record_audit_event(
        AuditEvent(
            event_type="delete",
            request_id=getattr(http_request.state, "request_id", str(uuid.uuid4())),
            tenant_id=tenant_id,
            actor=hash_actor(auth.api_key),
            status="completed",
            route=None,
            detail={
                "source_types": request.source_types or [],
                "source_names": request.source_names or [],
                "deleted": deleted,
            },
        )
    )
    return DeleteSourceResponse(deleted=deleted)


@app.post("/ingest/files", response_model=IngestResponse)
async def ingest_files(
    http_request: Request,
    files: list[UploadFile] = File(...),
    auth: AuthContext = Depends(require_api_key),
) -> IngestResponse:
    require_roles(auth, {"writer", "admin"})
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    pipeline = get_pipeline()
    store = get_metadata_store()
    tenant_id = resolve_tenant_id(http_request, auth)
    request_id = getattr(http_request.state, "request_id", str(uuid.uuid4()))
    record_id = None
    if store:
        meta = IngestionMeta(
            source_type="file",
            source_name=f"{tenant_id}:upload",
            source_uri="upload",
            status="started",
            extra={"file_count": len(files)},
        )
        record_id = store.record_start(meta)
    documents: list[Document] = []
    for idx, upload in enumerate(files, start=1):
        filename = upload.filename or f"upload-{idx}"
        suffix = Path(filename).suffix.lower()
        data = await upload.read()
        if not data:
            continue
        doc_id = f"{Path(filename).stem}-{idx}"
        source = filename
        try:
            if suffix in {".txt", ".text"}:
                document = load_text_bytes(data, doc_id=doc_id, source=source)
                document.metadata["source_type"] = "text"
                document.metadata.setdefault("source_name", Path(filename).stem)
            elif suffix in {".md", ".markdown"}:
                document = load_markdown_bytes(data, doc_id=doc_id, source=source)
                document.metadata["source_type"] = "markdown"
                document.metadata.setdefault("source_name", Path(filename).stem)
            elif suffix == ".pdf":
                document = load_pdf_bytes(data, doc_id=doc_id, source=source)
                document.metadata["source_type"] = "pdf"
                document.metadata.setdefault("source_name", Path(filename).stem)
            elif suffix == ".docx":
                document = load_docx_bytes(data, doc_id=doc_id, source=source)
                document.metadata["source_type"] = "docx"
                document.metadata.setdefault("source_name", Path(filename).stem)
            elif suffix == ".xlsx":
                document = load_xlsx_bytes(data, doc_id=doc_id, source=source)
                document.metadata["source_type"] = "xlsx"
                document.metadata.setdefault("source_name", Path(filename).stem)
            elif suffix in {".csv", ".tsv"}:
                document = load_csv_bytes(data, doc_id=doc_id, source=source)
                document.metadata["source_type"] = "csv"
                document.metadata.setdefault("source_name", Path(filename).stem)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
        except (PDFLoaderError, DocxLoaderError, XlsxLoaderError, CSVLoaderError) as exc:
            _record_audit_event(
                AuditEvent(
                    event_type="ingest",
                    request_id=request_id,
                    tenant_id=tenant_id,
                    actor=hash_actor(auth.api_key),
                    status="failed",
                    route=None,
                    detail={
                        "source_type": "file",
                        "source_name": filename,
                        "error": _safe_error_message(exc),
                    },
                )
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        document.metadata.setdefault("tenant_id", tenant_id)
        documents.extend(
            chunk_document(
                document,
                max_chars=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )
        )

    if not documents:
        _record_audit_event(
            AuditEvent(
                event_type="ingest",
                request_id=request_id,
                tenant_id=tenant_id,
                actor=hash_actor(auth.api_key),
                status="failed",
                route=None,
                detail={"source_type": "file", "source_name": "upload", "reason": "empty"},
            )
        )
        if store and record_id:
            store.record_failure(record_id, "No valid file content provided")
        raise HTTPException(status_code=400, detail="No valid file content provided")
    ingested = pipeline.ingest(documents)
    _record_audit_event(
        AuditEvent(
            event_type="ingest",
            request_id=request_id,
            tenant_id=tenant_id,
            actor=hash_actor(auth.api_key),
            status="completed",
            route=None,
            detail={
                "source_type": "file",
                "source_name": "upload",
                "ingested": ingested,
                "chunk_count": len(documents),
            },
        )
    )
    if store and record_id:
        store.record_complete(record_id, ingested=ingested, chunk_count=len(documents))
    return IngestResponse(ingested=ingested)


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    http_request: Request,
    auth: AuthContext = Depends(require_api_key),
) -> QueryResponse:
    require_roles(auth, {"reader", "writer", "admin"})
    request_id = request.trace_id or getattr(http_request.state, "request_id", str(uuid.uuid4()))
    tenant_id = resolve_tenant_id(http_request, auth)
    query_hash = hashlib.sha256(request.query.encode("utf-8")).hexdigest()
    if request.route:
        decision_tool = request.route
        decision_reason = "explicit_route"
    else:
        decision = router.route(request.query)
        decision_tool = decision.tool
        decision_reason = decision.reason

    logger.info(
        "query_received",
        extra={
            "request_id": request_id,
            "route": decision_tool,
            "route_reason": decision_reason,
            "query_length": len(request.query),
            "query_hash": query_hash,
            "top_k": request.top_k,
            "min_score": request.min_score,
            "source_types": request.source_types,
            "source_names": request.source_names,
            "source_filter_mode": request.source_filter_mode,
            "tenant_id": tenant_id,
            "actor": hash_actor(auth.api_key),
        },
    )

    pipeline = get_pipeline()
    if decision_tool == "sql":
        require_roles(auth, {"admin"})
        if not settings.sql_database_uri:
            answer = "SQL tool is not configured for this deployment."
            response = QueryResponse(
                answer=answer,
                sources=[],
                refusal_reason="sql_not_configured",
                route="sql",
                request_id=request_id,
            )
            _record_audit_event(
                AuditEvent(
                    event_type="query",
                    request_id=request_id,
                    tenant_id=tenant_id,
                    actor=hash_actor(auth.api_key),
                    status="failed",
                    route="sql",
                    detail={"reason": "sql_not_configured"},
                )
            )
            logger.info(
                "query_completed",
                extra={
                    "request_id": request_id,
                    "route": "sql",
                    "refusal_reason": response.refusal_reason,
                    "answer_length": len(response.answer),
                    "sources": 0,
                },
            )
            return response
        sql_query = request.sql_query or request.query
        try:
            plan = validate_select_query(
                sql_query,
                allowed_tables=settings.sql_allowed_tables,
                allowed_columns=settings.sql_allowed_columns,
                max_rows=settings.sql_max_rows,
            )
            records = execute_select_query(
                settings.sql_database_uri,
                plan=plan,
                params=request.sql_params,
                max_rows=settings.sql_max_rows,
            )
        except SQLValidationError as exc:
            _record_audit_event(
                AuditEvent(
                    event_type="query",
                    request_id=request_id,
                    tenant_id=tenant_id,
                    actor=hash_actor(auth.api_key),
                    status="failed",
                    route="sql",
                    detail={"error": _safe_error_message(exc)},
                )
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not records:
            response = QueryResponse(
                answer=DEFAULT_REFUSAL,
                sources=[],
                refusal_reason="no_context",
                route="sql",
                request_id=request_id,
            )
            _record_audit_event(
                AuditEvent(
                    event_type="query",
                    request_id=request_id,
                    tenant_id=tenant_id,
                    actor=hash_actor(auth.api_key),
                    status="refused",
                    route="sql",
                    detail={"reason": "no_context"},
                )
            )
            logger.info(
                "query_completed",
                extra={
                    "request_id": request_id,
                    "route": response.route,
                    "refusal_reason": response.refusal_reason,
                    "answer_length": len(response.answer),
                    "sources": 0,
                },
            )
            return response

        contexts = [
            ContextChunk(
                document_id=f"sql-{idx}",
                content=json.dumps(record, ensure_ascii=True, default=str),
                metadata={"source_type": "sql", "source_name": "sql"},
                score=1.0,
            )
            for idx, record in enumerate(records, start=1)
        ]
        structured = {"type": "sql", "records": records}
        answer = ""
        llm_refusal_reason = None
        if settings.answerer_mode.lower() == "llm":
            try:
                llm = OllamaAnswerer(
                    base_url=settings.ollama_base_url,
                    model=settings.ollama_model,
                    temperature=settings.ollama_temperature,
                    max_tokens=settings.ollama_max_tokens,
                    timeout=settings.ollama_timeout,
                    context_max_chars=settings.llm_context_max_chars,
                )
                llm_result = await llm.generate(request.query, contexts)
                if llm_result.refusal_reason:
                    llm_refusal_reason = llm_result.refusal_reason
                else:
                    answer = llm_result.answer
            except LLMError as exc:
                logger.error(
                    "llm_failed",
                    extra={
                        "request_id": request_id,
                        "detail": _safe_error_message(exc),
                    },
                )
        if not answer or answer == DEFAULT_REFUSAL:
            answer = SummarizingAnswerer().generate(request.query, contexts)
        if not answer:
            answer = DEFAULT_REFUSAL
        sources = [
            SourceChunk(
                document_id=chunk.document_id,
                content=chunk.content,
                metadata=chunk.metadata,
                score=chunk.score,
                highlights=build_highlights(chunk.content, request.query),
            )
            for chunk in contexts
        ]
        refusal_reason = None
        if answer == DEFAULT_REFUSAL:
            refusal_reason = llm_refusal_reason or "empty_answer"
        response = QueryResponse(
            answer=answer,
            sources=sources,
            refusal_reason=refusal_reason,
            route="sql",
            request_id=request_id,
            structured=structured,
        )
        _record_audit_event(
            AuditEvent(
                event_type="query",
                request_id=request_id,
                tenant_id=tenant_id,
                actor=hash_actor(auth.api_key),
                status="completed" if not response.refusal_reason else "refused",
                route="sql",
                detail={
                    "sources": len(response.sources),
                    "refusal_reason": response.refusal_reason,
                },
            )
        )
        logger.info(
            "query_completed",
            extra={
                "request_id": request_id,
                "route": response.route,
                "refusal_reason": response.refusal_reason,
                "answer_length": len(response.answer),
                "sources": len(response.sources),
            },
        )
        return response

    if decision_tool == "refuse":
        response = QueryResponse(
            answer=DEFAULT_REFUSAL,
            sources=[],
            refusal_reason="empty_query",
            route="refuse",
            request_id=request_id,
        )
        _record_audit_event(
            AuditEvent(
                event_type="query",
                request_id=request_id,
                tenant_id=tenant_id,
                actor=hash_actor(auth.api_key),
                status="refused",
                route="refuse",
                detail={"reason": "empty_query"},
            )
        )
        logger.info(
            "query_completed",
            extra={
                "request_id": request_id,
                "route": response.route,
                "refusal_reason": response.refusal_reason,
                "answer_length": len(response.answer),
                "sources": 0,
            },
        )
        return response

    if decision_tool == "summarize":
        base_limit = request.top_k or pipeline.max_chunks
        retrieval_limit = base_limit
        if request.source_types or request.source_names:
            retrieval_limit = min(base_limit * 5, 50)
        results = pipeline.retrieve(
            request.query,
            top_k=retrieval_limit,
            source_types=request.source_types,
            source_names=request.source_names,
            tenant_id=tenant_id,
            source_filter_mode=request.source_filter_mode,
        )
        contexts = pipeline.build_context(
            results,
            min_score=request.min_score,
            min_score_by_type=settings.min_score_by_type,
            min_score_by_source=settings.min_score_by_source,
            source_types=request.source_types,
            source_names=request.source_names,
            source_filter_mode=request.source_filter_mode,
            limit=base_limit,
        )
        guardrail = require_context(contexts)
        if not guardrail.allowed:
            response = QueryResponse(
                answer=DEFAULT_REFUSAL,
                sources=[],
                refusal_reason=guardrail.reason,
                route="summarize",
                request_id=request_id,
            )
        else:
            structured = None
            is_db = contexts and all(
                str(chunk.metadata.get("source_type", "")).lower() == "db"
                for chunk in contexts
            )
            if is_db:
                records = extract_db_records(contexts, max_items=base_limit)
                if records:
                    structured = {"type": "db", "records": records}
            answer = ""
            llm_refusal_reason = None
            if settings.answerer_mode.lower() == "llm":
                try:
                    llm = OllamaAnswerer(
                        base_url=settings.ollama_base_url,
                        model=settings.ollama_model,
                        temperature=settings.ollama_temperature,
                        max_tokens=settings.ollama_max_tokens,
                        timeout=settings.ollama_timeout,
                        context_max_chars=settings.llm_context_max_chars,
                    )
                    llm_result = await llm.generate(request.query, contexts)
                    if llm_result.refusal_reason:
                        llm_refusal_reason = llm_result.refusal_reason
                    else:
                        answer = llm_result.answer
                except LLMError as exc:
                    logger.error(
                        "llm_failed",
                        extra={
                            "request_id": request_id,
                            "detail": _safe_error_message(exc),
                        },
                    )
            else:
                summarizer = SummarizingAnswerer()
                answer = summarizer.generate(request.query, contexts)
            if not answer or answer == DEFAULT_REFUSAL:
                if is_db:
                    formatted = format_db_answer(contexts, max_items=base_limit)
                    if formatted:
                        answer = formatted
                elif settings.answerer_mode.lower() == "llm":
                    summarizer = SummarizingAnswerer()
                    answer = summarizer.generate(request.query, contexts)
            if not answer:
                response = QueryResponse(
                    answer=DEFAULT_REFUSAL,
                    sources=[],
                    refusal_reason=llm_refusal_reason or "empty_answer",
                    route="summarize",
                    request_id=request_id,
                )
            else:
                sources = [
                    SourceChunk(
                        document_id=chunk.document_id,
                        content=chunk.content,
                        metadata=chunk.metadata,
                        score=chunk.score,
                        highlights=build_highlights(chunk.content, request.query),
                    )
                    for chunk in contexts
                ]
                response = QueryResponse(
                    answer=answer,
                    sources=sources,
                    refusal_reason=None,
                    route="summarize",
                    request_id=request_id,
                    structured=structured,
                )
        logger.info(
            "query_completed",
            extra={
                "request_id": request_id,
                "route": response.route,
                "refusal_reason": response.refusal_reason,
                "answer_length": len(response.answer),
                "sources": len(response.sources),
            },
        )
        _record_audit_event(
            AuditEvent(
                event_type="query",
                request_id=request_id,
                tenant_id=tenant_id,
                actor=hash_actor(auth.api_key),
                status="completed" if not response.refusal_reason else "refused",
                route="summarize",
                detail={
                    "sources": len(response.sources),
                    "refusal_reason": response.refusal_reason,
                },
            )
        )
        return response

    structured = None
    if settings.answerer_mode.lower() == "llm":
        base_limit = request.top_k or pipeline.max_chunks
        retrieval_limit = base_limit
        if request.source_types or request.source_names:
            retrieval_limit = min(base_limit * 5, 50)
        results = pipeline.retrieve(
            request.query,
            top_k=retrieval_limit,
            source_types=request.source_types,
            source_names=request.source_names,
            tenant_id=tenant_id,
            source_filter_mode=request.source_filter_mode,
        )
        contexts = pipeline.build_context(
            results,
            min_score=request.min_score,
            min_score_by_type=settings.min_score_by_type,
            min_score_by_source=settings.min_score_by_source,
            source_types=request.source_types,
            source_names=request.source_names,
            source_filter_mode=request.source_filter_mode,
            limit=base_limit,
        )
        guardrail = require_context(contexts)
        if not guardrail.allowed:
            response = QueryResponse(
                answer=DEFAULT_REFUSAL,
                sources=[],
                refusal_reason=guardrail.reason,
                route="rag",
                request_id=request_id,
            )
            _record_audit_event(
                AuditEvent(
                    event_type="query",
                    request_id=request_id,
                    tenant_id=tenant_id,
                    actor=hash_actor(auth.api_key),
                    status="refused",
                    route="rag",
                    detail={"reason": guardrail.reason},
                )
            )
            logger.info(
                "query_completed",
                extra={
                    "request_id": request_id,
                    "route": response.route,
                    "refusal_reason": response.refusal_reason,
                    "answer_length": len(response.answer),
                    "sources": 0,
                },
            )
            return response
        answer = ""
        llm_refusal_reason = None
        is_db = contexts and all(
            str(chunk.metadata.get("source_type", "")).lower() == "db"
            for chunk in contexts
        )
        if is_db:
            records = extract_db_records(contexts, max_items=base_limit)
            if records:
                structured = {"type": "db", "records": records}
        try:
            llm = OllamaAnswerer(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=settings.ollama_temperature,
                max_tokens=settings.ollama_max_tokens,
                timeout=settings.ollama_timeout,
                context_max_chars=settings.llm_context_max_chars,
            )
            llm_result = await llm.generate(request.query, contexts)
            if llm_result.refusal_reason:
                llm_refusal_reason = llm_result.refusal_reason
            else:
                answer = llm_result.answer
        except LLMError as exc:
            logger.error(
                "llm_failed",
                extra={
                    "request_id": request_id,
                    "detail": _safe_error_message(exc),
                },
            )
        if not answer or answer == DEFAULT_REFUSAL:
            if is_db:
                formatted = format_db_answer(contexts, max_items=base_limit)
                if formatted:
                    answer = formatted
            else:
                answer = ExtractiveAnswerer().generate(request.query, contexts)
        if not answer:
            answer = DEFAULT_REFUSAL
        sources = [
            SourceChunk(
                document_id=chunk.document_id,
                content=chunk.content,
                metadata=chunk.metadata,
                score=chunk.score,
                highlights=build_highlights(chunk.content, request.query),
            )
            for chunk in contexts
        ]
        refusal_reason = None
        if answer == DEFAULT_REFUSAL:
            refusal_reason = llm_refusal_reason or "empty_answer"
        response = QueryResponse(
            answer=answer,
            sources=sources,
            refusal_reason=refusal_reason,
            route="rag",
            request_id=request_id,
            structured=structured,
        )
        _record_audit_event(
            AuditEvent(
                event_type="query",
                request_id=request_id,
                tenant_id=tenant_id,
                actor=hash_actor(auth.api_key),
                status="completed" if not response.refusal_reason else "refused",
                route="rag",
                detail={
                    "sources": len(response.sources),
                    "refusal_reason": response.refusal_reason,
                },
            )
        )
        logger.info(
            "query_completed",
            extra={
                "request_id": request_id,
                "route": response.route,
                "refusal_reason": response.refusal_reason,
                "answer_length": len(response.answer),
                "sources": len(response.sources),
            },
        )
        return response
    result = pipeline.answer(
        request.query,
        top_k=request.top_k,
        min_score=request.min_score,
        min_score_by_type=settings.min_score_by_type,
        min_score_by_source=settings.min_score_by_source,
        source_types=request.source_types,
        source_names=request.source_names,
        tenant_id=tenant_id,
        source_filter_mode=request.source_filter_mode,
    )
    answer = result.answer
    if result.sources and all(
        str(chunk.metadata.get("source_type", "")).lower() == "db"
        for chunk in result.sources
    ):
        records = extract_db_records(result.sources, max_items=request.top_k or pipeline.max_chunks)
        if records:
            structured = {"type": "db", "records": records}
        formatted = format_db_answer(
            result.sources, max_items=request.top_k or pipeline.max_chunks
        )
        if formatted:
            answer = formatted
    sources = [
        SourceChunk(
            document_id=chunk.document_id,
            content=chunk.content,
            metadata=chunk.metadata,
            score=chunk.score,
            highlights=build_highlights(chunk.content, request.query),
        )
        for chunk in result.sources
    ]
    response = QueryResponse(
        answer=answer,
        sources=sources,
        refusal_reason=result.refusal_reason,
        route="rag",
        request_id=request_id,
        structured=structured,
    )
    _record_audit_event(
        AuditEvent(
            event_type="query",
            request_id=request_id,
            tenant_id=tenant_id,
            actor=hash_actor(auth.api_key),
            status="completed" if not response.refusal_reason else "refused",
            route="rag",
            detail={
                "sources": len(response.sources),
                "refusal_reason": response.refusal_reason,
            },
        )
    )
    logger.info(
        "query_completed",
        extra={
            "request_id": request_id,
            "route": response.route,
            "refusal_reason": response.refusal_reason,
            "answer_length": len(response.answer),
            "sources": len(response.sources),
        },
    )
    return response
