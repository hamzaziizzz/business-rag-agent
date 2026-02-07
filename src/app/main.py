from __future__ import annotations

"""FastAPI application entrypoint for the document-only RAG service."""

import hashlib
import json
import logging
import uuid
from urllib.parse import urlparse, urlunparse
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse

from src.app.dependencies import (
    get_audit_store,
    get_embedding_config_report,
    get_metadata_store,
    get_pipeline,
    get_query_rewriter,
)
from src.app.settings import settings
from src.app.schemas import (
    DeleteSourceRequest,
    DeleteSourceResponse,
    IngestRequest,
    IngestResponse,
    ChatRequest,
    ChatResponse,
    ChatMessage,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    StatsResponse,
    StatsHealthResponse,
    EmbeddingHealthResponse,
)
from src.app.security import AuthContext, require_api_key, require_roles, resolve_tenant_id
from src.app.metrics import metrics_middleware, metrics_response
from src.loaders.chunking import chunk_document, normalize_query_tokens
from src.loaders.markdown import load_markdown_bytes
from src.loaders.docx import DocxLoaderError, load_docx_bytes
from src.loaders.pdf import PDFLoaderError, load_pdf_bytes
from src.loaders.text import load_text_bytes
from src.loaders.xlsx import XlsxLoaderError, load_xlsx_bytes
from src.loaders.csv_loader import CSVLoaderError, load_csv_bytes
from src.rag.answerer import ExtractiveAnswerer, SummarizingAnswerer
from src.rag.embeddings import EmbeddingConfigError
from src.rag.formatters import build_json_summary, extract_db_records, format_db_answer
from src.rag.guardrails import DEFAULT_REFUSAL, require_context
from src.rag.highlights import build_highlights
from src.rag.llm import LLMError, base_system_prompt, build_llm_answerer, build_llm_gate
from src.rag.citations import append_citation_footer, build_citations, order_contexts
from src.rag.rewriter import QueryRewriteError
from src.rag.types import ContextChunk, Document
from src.agents.router import AgentRouter
from src.metadata.store import IngestionMeta, MetadataStore
from src.metadata.audit import AuditEvent, hash_actor

logger = logging.getLogger(__name__)
router = AgentRouter()

app = FastAPI(title="Business RAG Agent", version="0.1.0")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_UI_PATH = PROJECT_ROOT / "fiverr_assets" / "simple_chat_ui.html"


def _configure_logging() -> None:
    """Configure root logging using environment settings."""
    level_name = settings.log_level.strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    logger.setLevel(level)


_configure_logging()


def _sanitize_url(url: str) -> str:
    """Remove query fragments from URLs before storing in metadata."""
    parsed = urlparse(url)
    if not parsed.scheme:
        return url
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def _record_audit_event(event: AuditEvent) -> None:
    """Persist an audit event if an audit store is configured."""
    audit_store = get_audit_store()
    if not audit_store:
        return
    audit_store.record_event(event)


def _safe_error_message(exc: Exception) -> str:
    """Return a safe error type name for logs and responses."""
    return type(exc).__name__


async def _rewrite_query(query: str) -> tuple[str, bool]:
    """Rewrite the query for retrieval when a rewriter is enabled."""
    if not settings.query_rewriter_enabled:
        return query, False
    try:
        rewriter = get_query_rewriter()
        rewritten = await rewriter.rewrite(query)
    except QueryRewriteError:
        logger.warning("query_rewrite_failed")
        return query, False
    if not rewritten or rewritten.strip() == query.strip():
        return query, False
    return rewritten, True


async def _llm_context_gate(
    query: str,
    contexts: list[ContextChunk],
    request_id: str,
) -> tuple[bool, str | None]:
    """Ask the LLM gate to decide if context is sufficient to answer."""
    if not settings.llm_gate_enabled:
        return True, None
    try:
        gate = build_llm_gate(
            settings.llm_provider,
            api_key_openai=settings.openai_api_key,
            api_key_gemini=settings.gemini_api_key,
            openai_base_url=settings.openai_base_url,
            openai_model=settings.openai_chat_model,
            gemini_model=settings.gemini_chat_model,
            ollama_base_url=settings.ollama_base_url,
            ollama_model=settings.ollama_model,
            temperature=0.0,
            max_tokens=settings.llm_gate_max_tokens,
            timeout=settings.llm_gate_timeout,
            context_max_chars=settings.llm_context_max_chars,
        )
        result = await gate.evaluate(query, contexts)
    except LLMError as exc:
        logger.error(
            "llm_gate_failed",
            extra={"request_id": request_id, "detail": _safe_error_message(exc)},
        )
        return True, None
    logger.info(
        "llm_gate_decision",
        extra={
            "request_id": request_id,
            "sufficient": result.sufficient,
            "reason": result.reason,
        },
    )
    return result.sufficient, result.reason


def _slice_history(messages: list[ChatMessage], max_turns: int) -> list[dict[str, str]]:
    """Return the most recent chat turns formatted for prompts."""
    if not messages:
        return []
    trimmed = messages[-max_turns:] if max_turns > 0 else messages
    return [{"role": msg.role, "content": msg.content} for msg in trimmed]


def _apply_citations(
    answer: str, contexts: list[ContextChunk], refusal_reason: str | None
) -> tuple[str, list]:
    """Append citations to the answer and serialize citation metadata."""
    if refusal_reason or answer == DEFAULT_REFUSAL:
        return answer, []
    citations = build_citations(contexts)
    answer = append_citation_footer(answer, citations)
    return answer, [citation.__dict__ for citation in citations]


def _build_structured_payload(
    contexts: list[ContextChunk],
    is_db: bool,
    max_items: int,
) -> dict[str, object] | None:
    """Build a structured response payload from retrieved contexts."""
    if is_db:
        records = extract_db_records(contexts, max_items=max_items)
        if records:
            return {"type": "db", "records": records}
        return None
    summary = build_json_summary(contexts, max_items=max_items)
    if summary:
        return {"type": "json", **summary}
    return None


def _role_system_prompt(role: str) -> str:
    """Augment the base system prompt with role-specific guidance."""
    base = base_system_prompt()
    role_normalized = role.strip().lower()
    if role_normalized == "admin":
        addon = "Provide detailed, complete answers with full context coverage."
    elif role_normalized == "writer":
        addon = "Focus on actionable, clear steps and policy-aligned guidance."
    else:
        addon = "Keep answers concise and easy to scan."
    return f"{base} {addon}"


def role_system_prompt_for_tests(role: str) -> str:
    """Expose role prompt generation for tests."""
    return _role_system_prompt(role)


def _log_top_scores(request_id: str, results: list["SearchResult"], limit: int = 5) -> None:
    """Log top retrieval scores for debugging threshold behavior."""
    from src.rag.types import SearchResult

    payload = []
    for result in results[:limit]:
        metadata = result.document.metadata or {}
        payload.append(
            {
                "doc_id": result.document.doc_id,
                "score": result.score,
                "source_type": metadata.get("source_type"),
                "source_name": metadata.get("source_name"),
            }
        )
    logger.info(
        "retrieval_top_scores",
        extra={
            "request_id": request_id,
            "count": len(results),
            "top": payload,
        },
    )




async def _read_upload_bytes(upload: UploadFile, max_bytes: int | None) -> bytes:
    """Stream upload bytes with a hard size limit."""
    if not max_bytes or max_bytes <= 0:
        return await upload.read()
    buffer = bytearray()
    while True:
        chunk = await upload.read(65536)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"File exceeds maximum size of {max_bytes} bytes",
            )
    return bytes(buffer)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Attach or create a request ID for traceability."""
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    """Capture request metrics before returning the response."""
    return await metrics_middleware(request, call_next)


@app.get("/metrics")
async def metrics():
    """Expose Prometheus-style metrics."""
    return metrics_response()


@app.get("/health")
async def health() -> dict[str, str]:
    """Simple health probe for uptime checks."""
    return {"status": "ok"}


@app.get("/demo", response_class=HTMLResponse)
async def demo() -> HTMLResponse:
    """Serve the static HTML demo UI."""
    if not DEMO_UI_PATH.exists():
        raise HTTPException(status_code=404, detail="Demo UI not found")
    return HTMLResponse(DEMO_UI_PATH.read_text(encoding="utf-8"))


@app.get("/stats", response_model=StatsResponse)
async def stats(auth: AuthContext = Depends(require_api_key)) -> StatsResponse:
    """Return vector store stats for admins."""
    require_roles(auth, {"admin"})
    pipeline = get_pipeline()
    return StatsResponse(**pipeline.vectorstore.stats())


@app.get("/stats/health", response_model=StatsHealthResponse)
async def stats_health(auth: AuthContext = Depends(require_api_key)) -> StatsHealthResponse:
    """Return vector store health status for admins."""
    require_roles(auth, {"admin"})
    pipeline = get_pipeline()
    return StatsHealthResponse(**pipeline.vectorstore.health())


@app.get("/stats/embedding", response_model=EmbeddingHealthResponse)
async def embedding_health(
    auth: AuthContext = Depends(require_api_key),
) -> EmbeddingHealthResponse:
    """Return embedding configuration health checks."""
    require_roles(auth, {"admin"})
    report = get_embedding_config_report()
    return EmbeddingHealthResponse(**report.__dict__)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    http_request: Request,
    auth: AuthContext = Depends(require_api_key),
) -> IngestResponse:
    """Ingest raw text documents into the vector store."""
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
                max_tokens=settings.chunk_tokens,
                token_overlap=settings.chunk_token_overlap,
                encoding_name=settings.tokenizer_encoding,
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




@app.post("/ingest/delete", response_model=DeleteSourceResponse)
async def delete_sources(
    request: DeleteSourceRequest,
    http_request: Request,
    auth: AuthContext = Depends(require_api_key),
) -> DeleteSourceResponse:
    """Delete ingested documents using source filters."""
    require_roles(auth, {"admin"})
    if not (request.source_types or request.source_names):
        raise HTTPException(status_code=400, detail="source_types or source_names required")
    pipeline = get_pipeline()
    tenant_id = resolve_tenant_id(http_request, auth)
    try:
        deleted = pipeline.delete_by_source(
            source_types=request.source_types,
            source_names=request.source_names,
            tenant_id=tenant_id,
            source_filter_mode=request.source_filter_mode,
        )
    except EmbeddingConfigError as exc:
        _record_audit_event(
            AuditEvent(
                event_type="delete",
                request_id=getattr(http_request.state, "request_id", str(uuid.uuid4())),
                tenant_id=tenant_id,
                actor=hash_actor(auth.api_key),
                status="failed",
                route=None,
                detail={"error": _safe_error_message(exc)},
            )
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
    """Ingest uploaded files into the vector store."""
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
        try:
            data = await _read_upload_bytes(upload, settings.file_max_bytes)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else "File exceeds size limit"
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
                        "error": detail,
                    },
                )
            )
            if store and record_id:
                store.record_failure(record_id, detail)
            logger.error(
                "file_ingest_failed",
                extra={
                    "request_id": request_id,
                    "source_name": filename,
                    "detail": detail,
                },
            )
            raise
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
                max_tokens=settings.chunk_tokens,
                token_overlap=settings.chunk_token_overlap,
                encoding_name=settings.tokenizer_encoding,
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
    """Route and answer a query using the RAG pipeline."""
    require_roles(auth, {"reader", "writer", "admin"})
    request_id = request.trace_id or getattr(http_request.state, "request_id", str(uuid.uuid4()))
    tenant_id = resolve_tenant_id(http_request, auth)
    query_hash = hashlib.sha256(request.query.encode("utf-8")).hexdigest()
    if request.route:
        decision_tool = request.route
        decision_reason = "explicit_route"
    else:
        decision = await router.route_async(request.query)
        decision_tool = decision.tool
        decision_reason = decision.reason
    retrieval_query = request.query
    rewrite_applied = False
    if decision_tool in {"rag", "summarize"}:
        retrieval_query, rewrite_applied = await _rewrite_query(request.query)
        retrieval_query = normalize_query_tokens(
            retrieval_query, encoding_name=settings.tokenizer_encoding
        )
    role_prompt = _role_system_prompt(auth.role)
    effective_min_score_by_type = (
        settings.min_score_by_type if request.min_score is None else {}
    )
    effective_min_score_by_source = (
        settings.min_score_by_source if request.min_score is None else {}
    )

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
            "rewrite_applied": rewrite_applied,
            "rewrite_length": len(retrieval_query) if rewrite_applied else None,
        },
    )
    logger.info(
        "thresholds_effective",
        extra={
            "request_id": request_id,
            "min_score": request.min_score,
            "min_score_by_type": effective_min_score_by_type,
            "min_score_by_source": effective_min_score_by_source,
        },
    )

    pipeline = get_pipeline()
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
            retrieval_query,
            top_k=retrieval_limit,
            source_types=request.source_types,
            source_names=request.source_names,
            tenant_id=tenant_id,
            source_filter_mode=request.source_filter_mode,
        )
        _log_top_scores(request_id, results)
        contexts = pipeline.build_context(
            results,
            min_score=request.min_score,
            min_score_by_type=effective_min_score_by_type,
            min_score_by_source=effective_min_score_by_source,
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
                answerer="guardrail",
                answerer_reason=guardrail.reason,
            )
        else:
            allowed, gate_reason = await _llm_context_gate(
                request.query, contexts, request_id
            )
            if not allowed:
                response = QueryResponse(
                    answer=DEFAULT_REFUSAL,
                    sources=[],
                    refusal_reason=gate_reason or "insufficient_context",
                    route="summarize",
                    request_id=request_id,
                    answerer="llm_gate",
                    answerer_reason=gate_reason or "insufficient_context",
                )
                logger.info(
                    "query_completed",
                    extra={
                        "request_id": request_id,
                        "route": "summarize",
                        "refusal_reason": response.refusal_reason,
                        "answer_length": len(response.answer),
                        "sources": 0,
                    },
                )
                _record_audit_event(
                    AuditEvent(
                        event_type="query",
                        request_id=request_id,
                        tenant_id=tenant_id,
                        actor=hash_actor(auth.api_key),
                        status="refused",
                        route="summarize",
                        detail={"reason": response.refusal_reason},
                    )
                )
                return response
            is_db = contexts and all(
                str(chunk.metadata.get("source_type", "")).lower() == "db"
                for chunk in contexts
            )
            structured = _build_structured_payload(contexts, is_db, base_limit)
            answer = ""
            llm_refusal_reason = None
            preferred_source_ids: list[str] | None = None
            llm_used = False
            fallback_used = False
            if settings.answerer_mode.lower() == "llm":
                try:
                    llm = build_llm_answerer(
                        settings.llm_provider,
                        api_key_openai=settings.openai_api_key,
                        api_key_gemini=settings.gemini_api_key,
                        openai_base_url=settings.openai_base_url,
                        openai_model=settings.openai_chat_model,
                        gemini_model=settings.gemini_chat_model,
                        ollama_base_url=settings.ollama_base_url,
                        ollama_model=settings.ollama_model,
                        temperature=settings.ollama_temperature,
                        max_tokens=settings.ollama_max_tokens,
                        timeout=settings.ollama_timeout,
                        context_max_chars=settings.llm_context_max_chars,
                        system_prompt=role_prompt,
                    )
                    llm_result = await llm.generate(request.query, contexts)
                    if llm_result.refusal_reason:
                        llm_refusal_reason = llm_result.refusal_reason
                    else:
                        answer = llm_result.answer
                        preferred_source_ids = llm_result.source_ids
                        llm_used = True
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
                fallback_used = True
            if not answer or answer == DEFAULT_REFUSAL:
                fallback_mode = settings.llm_fallback_mode.strip().lower()
                if is_db:
                    formatted = format_db_answer(contexts, max_items=base_limit)
                    if formatted:
                        answer = formatted
                        fallback_used = True
                elif settings.answerer_mode.lower() == "llm":
                    if fallback_mode == "refuse":
                        answer = ""
                    else:
                        summarizer = SummarizingAnswerer()
                        answer = summarizer.generate(request.query, contexts)
                        fallback_used = True
                elif not answer:
                    summarizer = SummarizingAnswerer()
                    answer = summarizer.generate(request.query, contexts)
                    fallback_used = True
            if llm_used:
                logger.info(
                    "answerer_llm_used",
                    extra={"request_id": request_id, "route": "summarize"},
                )
            elif fallback_used:
                logger.info(
                    "answerer_fallback_used",
                    extra={"request_id": request_id, "route": "summarize"},
                )
            if not answer:
                response = QueryResponse(
                    answer=DEFAULT_REFUSAL,
                    sources=[],
                    refusal_reason=llm_refusal_reason or "empty_answer",
                    route="summarize",
                    request_id=request_id,
                    answerer="llm" if settings.answerer_mode.lower() == "llm" else "summarize",
                    answerer_reason=llm_refusal_reason or "empty_answer",
                )
            else:
                ordered_contexts = order_contexts(contexts, preferred_source_ids)
                sources = [
                    SourceChunk(
                        document_id=chunk.document_id,
                        content=chunk.content,
                        metadata=chunk.metadata,
                        score=chunk.score,
                        highlights=build_highlights(chunk.content, request.query),
                    )
                    for chunk in ordered_contexts
                ]
                answer, citations = _apply_citations(answer, ordered_contexts, None)
                response = QueryResponse(
                    answer=answer,
                    sources=sources,
                    refusal_reason=None,
                    route="summarize",
                    request_id=request_id,
                    structured=structured,
                    citations=citations,
                    answerer="llm" if llm_used else "summarize",
                    answerer_reason=(
                        None
                        if llm_used
                        else (llm_refusal_reason or "fallback")
                    ),
                )
        logger.info(
            "query_completed",
            extra={
                "request_id": request_id,
                "route": response.route,
                "refusal_reason": response.refusal_reason,
                "answer_length": len(response.answer),
                "sources": len(response.sources),
                "llm_used": llm_used if "llm_used" in locals() else False,
                "fallback_used": fallback_used if "fallback_used" in locals() else False,
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
            retrieval_query,
            top_k=retrieval_limit,
            source_types=request.source_types,
            source_names=request.source_names,
            tenant_id=tenant_id,
            source_filter_mode=request.source_filter_mode,
        )
        _log_top_scores(request_id, results)
        contexts = pipeline.build_context(
            results,
            min_score=request.min_score,
            min_score_by_type=effective_min_score_by_type,
            min_score_by_source=effective_min_score_by_source,
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
                answerer="guardrail",
                answerer_reason=guardrail.reason,
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
                    "route": "rag",
                    "refusal_reason": response.refusal_reason,
                    "answer_length": len(response.answer),
                    "sources": 0,
                },
            )
            return response
        allowed, gate_reason = await _llm_context_gate(
            request.query, contexts, request_id
        )
        if not allowed:
            response = QueryResponse(
                answer=DEFAULT_REFUSAL,
                sources=[],
                refusal_reason=gate_reason or "insufficient_context",
                route="rag",
                request_id=request_id,
                answerer="llm_gate",
                answerer_reason=gate_reason or "insufficient_context",
            )
            _record_audit_event(
                AuditEvent(
                    event_type="query",
                    request_id=request_id,
                    tenant_id=tenant_id,
                    actor=hash_actor(auth.api_key),
                    status="refused",
                    route="rag",
                    detail={"reason": response.refusal_reason},
                )
            )
            logger.info(
                "query_completed",
                extra={
                    "request_id": request_id,
                    "route": "rag",
                    "refusal_reason": response.refusal_reason,
                    "answer_length": len(response.answer),
                    "sources": 0,
                },
            )
            return response
        answer = ""
        llm_refusal_reason = None
        preferred_source_ids: list[str] | None = None
        llm_used = False
        fallback_used = False
        is_db = contexts and all(
            str(chunk.metadata.get("source_type", "")).lower() == "db"
            for chunk in contexts
        )
        structured = _build_structured_payload(contexts, is_db, base_limit)
        try:
            llm = build_llm_answerer(
                settings.llm_provider,
                api_key_openai=settings.openai_api_key,
                api_key_gemini=settings.gemini_api_key,
                openai_base_url=settings.openai_base_url,
                openai_model=settings.openai_chat_model,
                gemini_model=settings.gemini_chat_model,
                ollama_base_url=settings.ollama_base_url,
                ollama_model=settings.ollama_model,
                temperature=settings.ollama_temperature,
                max_tokens=settings.ollama_max_tokens,
                timeout=settings.ollama_timeout,
                context_max_chars=settings.llm_context_max_chars,
                system_prompt=role_prompt,
            )
            llm_result = await llm.generate(request.query, contexts)
            if llm_result.refusal_reason:
                llm_refusal_reason = llm_result.refusal_reason
            else:
                answer = llm_result.answer
                preferred_source_ids = llm_result.source_ids
                llm_used = True
        except LLMError as exc:
            logger.error(
                "llm_failed",
                extra={
                    "request_id": request_id,
                    "detail": _safe_error_message(exc),
                },
            )
        if not answer or answer == DEFAULT_REFUSAL:
            fallback_mode = settings.llm_fallback_mode.strip().lower()
            if is_db:
                formatted = format_db_answer(contexts, max_items=base_limit)
                if formatted:
                    answer = formatted
                    fallback_used = True
            elif settings.answerer_mode.lower() == "llm" and fallback_mode == "summarize":
                summarizer = SummarizingAnswerer()
                answer = summarizer.generate(request.query, contexts)
                fallback_used = True
            elif settings.answerer_mode.lower() == "llm" and fallback_mode == "refuse":
                answer = ""
            else:
                answer = ExtractiveAnswerer().generate(request.query, contexts)
                fallback_used = True
        if not answer:
            answer = DEFAULT_REFUSAL
        if llm_used:
            logger.info(
                "answerer_llm_used",
                extra={"request_id": request_id, "route": "rag"},
            )
        elif fallback_used:
            logger.info(
                "answerer_fallback_used",
                extra={"request_id": request_id, "route": "rag"},
            )
        ordered_contexts = order_contexts(contexts, preferred_source_ids)
        sources = [
            SourceChunk(
                document_id=chunk.document_id,
                content=chunk.content,
                metadata=chunk.metadata,
                score=chunk.score,
                highlights=build_highlights(chunk.content, request.query),
            )
            for chunk in ordered_contexts
        ]
        refusal_reason = None
        if answer == DEFAULT_REFUSAL:
            refusal_reason = llm_refusal_reason or "empty_answer"
        answer, citations = _apply_citations(answer, ordered_contexts, refusal_reason)
        response = QueryResponse(
            answer=answer,
            sources=sources,
            refusal_reason=refusal_reason,
            route="rag",
            request_id=request_id,
            structured=structured,
            citations=citations,
            answerer="llm" if llm_used else "extractive",
            answerer_reason=(
                None
                if llm_used
                else (llm_refusal_reason or "fallback")
            ),
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
                "llm_used": llm_used,
                "fallback_used": fallback_used,
            },
        )
        return response
    result = pipeline.answer(
        request.query,
        top_k=request.top_k,
        min_score=request.min_score,
        min_score_by_type=effective_min_score_by_type,
        min_score_by_source=effective_min_score_by_source,
        source_types=request.source_types,
        source_names=request.source_names,
        tenant_id=tenant_id,
        source_filter_mode=request.source_filter_mode,
        retrieval_query=retrieval_query,
    )
    answer = result.answer
    is_db = result.sources and all(
        str(chunk.metadata.get("source_type", "")).lower() == "db"
        for chunk in result.sources
    )
    if is_db:
        records = extract_db_records(result.sources, max_items=request.top_k or pipeline.max_chunks)
        if records:
            structured = {"type": "db", "records": records}
        formatted = format_db_answer(
            result.sources, max_items=request.top_k or pipeline.max_chunks
        )
        if formatted:
            answer = formatted
    else:
        structured = _build_structured_payload(
            result.sources, False, request.top_k or pipeline.max_chunks
        )
    ordered_contexts = order_contexts(result.sources, None)
    sources = [
        SourceChunk(
            document_id=chunk.document_id,
            content=chunk.content,
            metadata=chunk.metadata,
            score=chunk.score,
            highlights=build_highlights(chunk.content, request.query),
        )
        for chunk in ordered_contexts
    ]
    answer, citations = _apply_citations(answer, ordered_contexts, result.refusal_reason)
    response = QueryResponse(
        answer=answer,
        sources=sources,
        refusal_reason=result.refusal_reason,
        route="rag",
        request_id=request_id,
        structured=structured,
        citations=citations,
        answerer="extractive",
        answerer_reason=None if not result.refusal_reason else result.refusal_reason,
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


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    http_request: Request,
    auth: AuthContext = Depends(require_api_key),
) -> ChatResponse:
    """Chat endpoint for multi-turn RAG conversations."""
    require_roles(auth, {"reader", "writer", "admin"})
    request_id = request.trace_id or getattr(http_request.state, "request_id", str(uuid.uuid4()))
    tenant_id = resolve_tenant_id(http_request, auth)
    messages = request.messages or []
    if not messages:
        return ChatResponse(
            answer=DEFAULT_REFUSAL,
            sources=[],
            refusal_reason="empty_messages",
            route="chat",
            request_id=request_id,
            answerer="guardrail",
            answerer_reason="empty_messages",
        )
    last_user = next((m.content for m in reversed(messages) if m.role == "user"), "").strip()
    if not last_user:
        return ChatResponse(
            answer=DEFAULT_REFUSAL,
            sources=[],
            refusal_reason="empty_query",
            route="chat",
            request_id=request_id,
            answerer="guardrail",
            answerer_reason="empty_query",
        )
    history = _slice_history(messages, settings.chat_history_turns)
    decision = await router.route_chat_async(last_user, history)
    logger.info(
        "chat_retrieval_decision",
        extra={
            "request_id": request_id,
            "retrieve": decision.retrieve,
            "reason": decision.reason,
        },
    )
    if not decision.retrieve:
        return ChatResponse(
            answer=DEFAULT_REFUSAL,
            sources=[],
            refusal_reason=decision.reason,
            route="chat",
            request_id=request_id,
            answerer="llm_router",
            answerer_reason=decision.reason,
        )

    pipeline = get_pipeline()
    retrieval_query, rewrite_applied = await _rewrite_query(last_user)
    retrieval_query = normalize_query_tokens(
        retrieval_query, encoding_name=settings.tokenizer_encoding
    )
    base_limit = request.top_k or pipeline.max_chunks
    retrieval_limit = base_limit
    if request.source_types or request.source_names:
        retrieval_limit = min(base_limit * 5, 50)
    results = pipeline.retrieve(
        retrieval_query,
        top_k=retrieval_limit,
        source_types=request.source_types,
        source_names=request.source_names,
        tenant_id=tenant_id,
        source_filter_mode=request.source_filter_mode,
    )
    _log_top_scores(request_id, results)
    effective_min_score_by_type = settings.min_score_by_type if request.min_score is None else {}
    effective_min_score_by_source = (
        settings.min_score_by_source if request.min_score is None else {}
    )
    contexts = pipeline.build_context(
        results,
        min_score=request.min_score,
        min_score_by_type=effective_min_score_by_type,
        min_score_by_source=effective_min_score_by_source,
        source_types=request.source_types,
        source_names=request.source_names,
        source_filter_mode=request.source_filter_mode,
        limit=base_limit,
    )
    guardrail = require_context(contexts)
    if not guardrail.allowed:
        return ChatResponse(
            answer=DEFAULT_REFUSAL,
            sources=[],
            refusal_reason=guardrail.reason,
            route="chat",
            request_id=request_id,
            answerer="guardrail",
            answerer_reason=guardrail.reason,
        )
    allowed, gate_reason = await _llm_context_gate(
        last_user, contexts, request_id
    )
    if not allowed:
        return ChatResponse(
            answer=DEFAULT_REFUSAL,
            sources=[],
            refusal_reason=gate_reason or "insufficient_context",
            route="chat",
            request_id=request_id,
            answerer="llm_gate",
            answerer_reason=gate_reason or "insufficient_context",
        )

    is_db = contexts and all(
        str(chunk.metadata.get("source_type", "")).lower() == "db"
        for chunk in contexts
    )
    structured = _build_structured_payload(contexts, is_db, base_limit)
    answer = ""
    llm_refusal_reason = None
    preferred_source_ids: list[str] | None = None
    llm_used = False
    fallback_used = False
    role_prompt = _role_system_prompt(auth.role)
    try:
        llm = build_llm_answerer(
            settings.llm_provider,
            api_key_openai=settings.openai_api_key,
            api_key_gemini=settings.gemini_api_key,
            openai_base_url=settings.openai_base_url,
            openai_model=settings.openai_chat_model,
            gemini_model=settings.gemini_chat_model,
            ollama_base_url=settings.ollama_base_url,
            ollama_model=settings.ollama_model,
            temperature=settings.ollama_temperature,
            max_tokens=settings.ollama_max_tokens,
            timeout=settings.ollama_timeout,
            context_max_chars=settings.llm_context_max_chars,
            system_prompt=role_prompt,
        )
        llm_result = await llm.generate(last_user, contexts, history=history)
        if llm_result.refusal_reason:
            llm_refusal_reason = llm_result.refusal_reason
        else:
            answer = llm_result.answer
            preferred_source_ids = llm_result.source_ids
            llm_used = True
    except LLMError as exc:
        logger.error(
            "llm_failed",
            extra={
                "request_id": request_id,
                "detail": _safe_error_message(exc),
            },
        )
    if not answer or answer == DEFAULT_REFUSAL:
        fallback_mode = settings.llm_fallback_mode.strip().lower()
        if is_db:
            formatted = format_db_answer(contexts, max_items=base_limit)
            if formatted:
                answer = formatted
                fallback_used = True
        elif settings.answerer_mode.lower() == "llm" and fallback_mode == "summarize":
            summarizer = SummarizingAnswerer()
            answer = summarizer.generate(last_user, contexts)
            fallback_used = True
        elif settings.answerer_mode.lower() == "llm" and fallback_mode == "refuse":
            answer = ""
        else:
            answer = ExtractiveAnswerer().generate(last_user, contexts)
            fallback_used = True
    if not answer:
        return ChatResponse(
            answer=DEFAULT_REFUSAL,
            sources=[],
            refusal_reason=llm_refusal_reason or "empty_answer",
            route="chat",
            request_id=request_id,
            answerer="llm",
            answerer_reason=llm_refusal_reason or "empty_answer",
        )

    ordered_contexts = order_contexts(contexts, preferred_source_ids)
    sources = [
        SourceChunk(
            document_id=chunk.document_id,
            content=chunk.content,
            metadata=chunk.metadata,
            score=chunk.score,
            highlights=build_highlights(chunk.content, last_user),
        )
        for chunk in ordered_contexts
    ]
    answerer = "llm" if llm_used else "extractive"
    if fallback_used and not llm_used:
        answerer = "extractive"
    answer, citations = _apply_citations(answer, ordered_contexts, None)
    return ChatResponse(
        answer=answer,
        sources=sources,
        refusal_reason=None,
        route="chat",
        request_id=request_id,
        structured=structured,
        citations=citations,
        answerer=answerer,
        answerer_reason=None if llm_used else (llm_refusal_reason or "fallback"),
    )
