from __future__ import annotations

"""PDF text extraction and cleanup."""

from io import BytesIO
import re
from pathlib import Path

from src.rag.types import Document


class PDFLoaderError(RuntimeError):
    """Raised when PDF loading fails."""
    pass


_WHITESPACE_RE = re.compile(r"\s+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def _clean_pdf_text(text: str) -> str:
    """Aggressively clean PDF-extracted text for better chunking."""
    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n")
    cleaned = re.sub(r"(\w)-\n(\w)", r"\1\2", cleaned)
    cleaned = cleaned.replace("\n", " ")
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    tokens = cleaned.split()
    merged: list[str] = []
    idx = 0
    while idx < len(tokens):
        current = tokens[idx]
        if idx + 1 < len(tokens):
            nxt = tokens[idx + 1]
            curr_lower = current.lower()
            if (
                current.isalpha()
                and nxt.isalpha()
                and (len(current) <= 3 or len(nxt) <= 3)
                and curr_lower not in _STOPWORDS
            ):
                merged.append(current + nxt)
                idx += 2
                continue
        merged.append(current)
        idx += 1
    return " ".join(merged)


def load_pdf_file(path: Path, doc_id: str | None = None) -> Document:
    """Load a PDF from disk and return a Document."""
    try:
        import fitz
    except ImportError as exc:
        raise PDFLoaderError("PyMuPDF is required to load PDF files") from exc

    reader = fitz.open(str(path))
    text_parts: list[str] = []
    for page in reader:
        text = page.get_text() or ""
        text_parts.append(text)
    content = _clean_pdf_text("\n".join(text_parts))
    return Document(
        doc_id=doc_id or path.stem,
        content=content,
        metadata={"source": str(path)},
    )


def load_pdf_bytes(data: bytes, doc_id: str, source: str) -> Document:
    """Load a PDF from bytes and return a Document."""
    try:
        import fitz
    except ImportError as exc:
        raise PDFLoaderError("PyMuPDF is required to load PDF files") from exc

    reader = fitz.open(stream=BytesIO(data).read(), filetype="pdf")
    text_parts: list[str] = []
    for page in reader:
        text = page.get_text() or ""
        text_parts.append(text)
    content = _clean_pdf_text("\n".join(text_parts))
    return Document(doc_id=doc_id, content=content, metadata={"source": source})
