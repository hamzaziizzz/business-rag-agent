from __future__ import annotations

"""In-memory vector store for local testing and small datasets."""

import math
from dataclasses import dataclass, field
from typing import Iterable

from src.rag.embeddings import EmbeddingProvider
from src.rag.types import Document, SearchResult


@dataclass
class InMemoryVectorStore:
    """Simple in-memory vector store with cosine similarity search."""
    embedder: EmbeddingProvider
    documents: list[Document] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)

    def add_documents(self, documents: Iterable[Document]) -> int:
        """Embed and store documents."""
        added = 0
        for document in documents:
            vector = self.embedder.embed(document.content)
            self.documents.append(document)
            self.vectors.append(vector)
            added += 1
        return added

    def search(
        self,
        query: str,
        top_k: int = 4,
        source_types: list[str] | None = None,
        source_names: list[str] | None = None,
        tenant_id: str | None = None,
        source_filter_mode: str = "and",
    ) -> list[SearchResult]:
        """Search stored vectors and apply optional filters."""
        if not self.documents:
            return []
        query_vector = self.embedder.embed(query)
        allowed_types = {value.lower() for value in source_types or []}
        allowed_names = {value.lower() for value in source_names or []}
        scored = [
            SearchResult(document=doc, score=self._cosine_similarity(query_vector, vec))
            for doc, vec in zip(self.documents, self.vectors)
        ]
        if allowed_types or allowed_names:
            scored = [
                result
                for result in scored
                if self._matches_filters(
                    result.document.metadata,
                    allowed_types,
                    allowed_names,
                    tenant_id=tenant_id,
                    mode=source_filter_mode,
                )
            ]
        if tenant_id and tenant_id != "*":
            scored = [
                result
                for result in scored
                if str(result.document.metadata.get("tenant_id", "")).lower()
                == tenant_id.lower()
            ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _matches_filters(
        self,
        metadata: dict[str, object],
        allowed_types: set[str],
        allowed_names: set[str],
        tenant_id: str | None,
        mode: str = "and",
    ) -> bool:
        """Check metadata against filter constraints."""
        type_matches = True
        name_matches = True
        tenant_matches = True
        if tenant_id and tenant_id != "*":
            tenant_matches = str(metadata.get("tenant_id", "")).lower() == tenant_id.lower()
        if allowed_types:
            source_type = str(metadata.get("source_type", "")).lower()
            type_matches = source_type in allowed_types
        if allowed_names:
            source_name = str(metadata.get("source_name", "")).lower()
            source = str(metadata.get("source", "")).lower()
            candidates = {value for value in (source_name, source) if value}
            if ":" in source:
                candidates.add(source.split(":", 1)[1])
            name_matches = bool(candidates.intersection(allowed_names))
        if not (allowed_types or allowed_names):
            return tenant_matches
        if mode == "or" and allowed_types and allowed_names:
            return tenant_matches and (type_matches or name_matches)
        return tenant_matches and type_matches and name_matches

    def stats(self) -> dict[str, int | str]:
        """Return basic stats for the vector store."""
        return {
            "backend": "memory",
            "document_count": len(self.documents),
            "embedding_dimension": self.embedder.dimension,
        }

    def health(self) -> dict[str, str | bool]:
        """Return health information for the vector store."""
        return {
            "backend": "memory",
            "ok": True,
        }

    def delete_by_source(
        self,
        source_types: list[str] | None,
        source_names: list[str] | None,
        tenant_id: str | None = None,
        source_filter_mode: str = "and",
    ) -> int:
        """Delete documents matching source filters."""
        if not (source_types or source_names):
            return 0
        allowed_types = {value.lower() for value in source_types or []}
        allowed_names = {value.lower() for value in source_names or []}
        kept_documents: list[Document] = []
        kept_vectors: list[list[float]] = []
        removed = 0
        for doc, vector in zip(self.documents, self.vectors):
            if self._matches_filters(
                doc.metadata,
                allowed_types,
                allowed_names,
                tenant_id=tenant_id,
                mode=source_filter_mode,
            ):
                removed += 1
                continue
            kept_documents.append(doc)
            kept_vectors.append(vector)
        self.documents = kept_documents
        self.vectors = kept_vectors
        return removed
