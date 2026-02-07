from __future__ import annotations

"""FastAPI application entrypoint for the basic document-only RAG service."""

import logging
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile

from src.app.dependencies import get_pipeline
from src.app.schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse, SourceChunk
from src.app.security import AuthContext, require_api_key
from src.app.settings import settings
from src.loaders.chunking import chunk_document
from src.loaders.csv_loader import CSVLoaderError, load_csv_bytes
from src.loaders.docx import DocxLoaderError, load_docx_bytes
from src.loaders.markdown import load_markdown_bytes
from src.loaders.pdf import PDFLoaderError, load_pdf_bytes
from src.loaders.text import load_text_bytes
from src.loaders.xlsx import XlsxLoaderError, load_xlsx_bytes
from src.rag.citations import append_citation_footer, build_citations
from src.rag.guardrails import DEFAULT_REFUSAL
from src.rag.types import Document

logger = logging.getLogger(__name__)

app = FastAPI(title="Business RAG Agent", version="0.1.0")


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


@app.get("/health")
async def health() -> dict[str, str]:
    """Simple health probe for uptime checks."""
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    auth: AuthContext = Depends(require_api_key),
) -> IngestResponse:
    """Ingest raw text documents into the vector store."""
    _ = auth
    pipeline = get_pipeline()
    documents: list[Document] = []
    for idx, doc in enumerate(request.documents, start=1):
        metadata = dict(doc.metadata)
        metadata.setdefault("source_type", "manual")
        metadata.setdefault("source_name", "manual")
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
        raise HTTPException(status_code=400, detail="No documents provided")
    ingested = pipeline.ingest(documents)
    return IngestResponse(ingested=ingested)


@app.post("/ingest/files", response_model=IngestResponse)
async def ingest_files(
    files: list[UploadFile] = File(...),
    auth: AuthContext = Depends(require_api_key),
) -> IngestResponse:
    """Ingest uploaded files into the vector store."""
    _ = auth
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    pipeline = get_pipeline()
    documents: list[Document] = []
    for idx, upload in enumerate(files, start=1):
        filename = upload.filename or f"upload-{idx}"
        suffix = Path(filename).suffix.lower()
        data = await _read_upload_bytes(upload, settings.file_max_bytes)
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
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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
        raise HTTPException(status_code=400, detail="No valid file content provided")
    ingested = pipeline.ingest(documents)
    return IngestResponse(ingested=ingested)


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    http_request: Request,
    auth: AuthContext = Depends(require_api_key),
) -> QueryResponse:
    """Answer a query using the RAG pipeline."""
    _ = auth
    request_id = getattr(http_request.state, "request_id", str(uuid.uuid4()))
    pipeline = get_pipeline()
    result = pipeline.answer(
        request.query,
        top_k=request.top_k,
        min_score=request.min_score,
    )
    sources = [
        SourceChunk(
            document_id=chunk.document_id,
            content=chunk.content,
            metadata=chunk.metadata,
            score=chunk.score,
        )
        for chunk in result.sources
    ]
    answer = result.answer
    citations = []
    if result.refusal_reason is None and answer != DEFAULT_REFUSAL:
        citations = build_citations(result.sources)
        answer = append_citation_footer(answer, citations)
    return QueryResponse(
        answer=answer,
        sources=sources,
        refusal_reason=result.refusal_reason,
        request_id=request_id,
        citations=[citation.__dict__ for citation in citations],
    )
