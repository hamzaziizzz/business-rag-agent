from __future__ import annotations

"""Citation helpers for attaching sources to answers."""

from dataclasses import dataclass

from src.rag.types import ContextChunk


@dataclass(frozen=True)
class Citation:
    """Citation metadata for a single context chunk."""
    label: str
    document_id: str
    source_type: str
    source_name: str


def order_contexts(
    contexts: list[ContextChunk], preferred_ids: list[str] | None
) -> list[ContextChunk]:
    """Order contexts based on preferred document IDs."""
    if not preferred_ids:
        return contexts
    preferred_map = {doc_id: idx for idx, doc_id in enumerate(preferred_ids)}
    preferred: list[ContextChunk] = []
    remaining: list[ContextChunk] = []
    for chunk in contexts:
        if chunk.document_id in preferred_map:
            preferred.append(chunk)
        else:
            remaining.append(chunk)
    preferred.sort(key=lambda chunk: preferred_map.get(chunk.document_id, 0))
    return preferred + remaining


def build_citations(contexts: list[ContextChunk]) -> list[Citation]:
    """Build citation labels for contexts."""
    citations: list[Citation] = []
    for idx, chunk in enumerate(contexts, start=1):
        source_type = str(chunk.metadata.get("source_type", "unknown"))
        source_name = str(chunk.metadata.get("source_name", "unknown"))
        citations.append(
            Citation(
                label=f"[{idx}]",
                document_id=chunk.document_id,
                source_type=source_type,
                source_name=source_name,
            )
        )
    return citations


def append_citation_footer(answer: str, citations: list[Citation]) -> str:
    """Append citation labels to the answer."""
    if not citations:
        return answer
    labels = " ".join(citation.label for citation in citations)
    return f"{answer}\n\nSources: {labels}"
