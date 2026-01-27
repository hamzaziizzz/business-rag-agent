from __future__ import annotations

from io import BytesIO
from pathlib import Path

from src.rag.types import Document


class PDFLoaderError(RuntimeError):
    pass


def load_pdf_file(path: Path, doc_id: str | None = None) -> Document:
    try:
        from PyPDF2 import PdfReader
    except ImportError as exc:
        raise PDFLoaderError("PyPDF2 is required to load PDF files") from exc

    reader = PdfReader(str(path))
    text_parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text_parts.append(text)
    content = "\n".join(text_parts).strip()
    return Document(
        doc_id=doc_id or path.stem,
        content=content,
        metadata={"source": str(path)},
    )


def load_pdf_bytes(data: bytes, doc_id: str, source: str) -> Document:
    try:
        from PyPDF2 import PdfReader
    except ImportError as exc:
        raise PDFLoaderError("PyPDF2 is required to load PDF files") from exc

    reader = PdfReader(BytesIO(data))
    text_parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text_parts.append(text)
    content = "\n".join(text_parts).strip()
    return Document(doc_id=doc_id, content=content, metadata={"source": source})
