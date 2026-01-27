from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from src.rag.answerer import ExtractiveAnswerer
from src.rag.guardrails import DEFAULT_REFUSAL, require_context
from src.rag.types import ContextChunk, Document, SearchResult
from src.vectorstore.inmemory import InMemoryVectorStore

logger = logging.getLogger(__name__)


def _matches_source_name(metadata: dict[str, object], allowed: set[str]) -> bool:
    source_name = str(metadata.get("source_name", "")).lower()
    source = str(metadata.get("source", "")).lower()
    candidates = {value for value in (source_name, source) if value}
    if ":" in source:
        candidates.add(source.split(":", 1)[1])
    return bool(candidates.intersection(allowed))


def _extract_source_name(metadata: dict[str, object]) -> str:
    source_name = str(metadata.get("source_name", "")).strip().lower()
    source = str(metadata.get("source", "")).strip().lower()
    if source_name:
        return source_name
    if ":" in source:
        return source.split(":", 1)[1]
    return source


@dataclass
class RAGResponse:
    answer: str
    sources: list[ContextChunk]
    refusal_reason: str | None = None


@dataclass
class RAGPipeline:
    vectorstore: InMemoryVectorStore
    answerer: ExtractiveAnswerer
    max_chunks: int = 4
    min_score: float = 0.2
    min_score_by_type: dict[str, float] | None = None
    min_score_by_source: dict[str, float] | None = None
    source_weights: dict[str, float] | None = None
    source_name_weights: dict[str, float] | None = None

    def ingest(self, documents: Iterable[Document]) -> int:
        return self.vectorstore.add_documents(documents)

    def delete_by_source(
        self,
        source_types: list[str] | None,
        source_names: list[str] | None,
        tenant_id: str | None = None,
        source_filter_mode: str = "and",
    ) -> int:
        if not (source_types or source_names):
            return 0
        return self.vectorstore.delete_by_source(
            source_types=source_types,
            source_names=source_names,
            tenant_id=tenant_id,
            source_filter_mode=source_filter_mode,
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        source_types: list[str] | None = None,
        source_names: list[str] | None = None,
        tenant_id: str | None = None,
        source_filter_mode: str = "and",
    ) -> list[SearchResult]:
        results = self.vectorstore.search(
            query,
            top_k=top_k or self.max_chunks,
            source_types=source_types,
            source_names=source_names,
            tenant_id=tenant_id,
            source_filter_mode=source_filter_mode,
        )
        logger.info(
            "retrieval_complete",
            extra={
                "results": len(results),
                "query_length": len(query),
            },
        )
        return results

    def build_context(
        self,
        results: list[SearchResult],
        min_score: float | None = None,
        min_score_by_type: dict[str, float] | None = None,
        min_score_by_source: dict[str, float] | None = None,
        source_types: list[str] | None = None,
        source_names: list[str] | None = None,
        source_filter_mode: str = "and",
        limit: int | None = None,
    ) -> list[ContextChunk]:
        contexts: list[ContextChunk] = []
        threshold_default = self.min_score if min_score is None else min_score
        thresholds_by_type = min_score_by_type or self.min_score_by_type or {}
        thresholds_by_source = min_score_by_source or self.min_score_by_source or {}
        weights_by_type = self.source_weights or {}
        weights_by_source = self.source_name_weights or {}
        allowed_types = {value.lower() for value in source_types or []}
        allowed_names = {value.lower() for value in source_names or []}
        mode = source_filter_mode.lower()
        weighted = []
        for result in results:
            source_type = str(result.document.metadata.get("source_type", "default")).lower()
            source_name = _extract_source_name(result.document.metadata)
            weight = weights_by_type.get(source_type, 1.0) * weights_by_source.get(
                source_name, 1.0
            )
            weighted.append((result, result.score * weight))
        weighted.sort(key=lambda item: item[1], reverse=True)
        for result, _ in weighted:
            source_type = str(result.document.metadata.get("source_type", "default")).lower()
            type_matches = not allowed_types or source_type in allowed_types
            name_matches = not allowed_names or _matches_source_name(
                result.document.metadata, allowed_names
            )
            if allowed_types or allowed_names:
                if mode == "or":
                    if allowed_types and allowed_names:
                        if not (type_matches or name_matches):
                            continue
                    elif not type_matches and not name_matches:
                        continue
                elif not (type_matches and name_matches):
                    continue
            source_name = _extract_source_name(result.document.metadata)
            threshold = thresholds_by_source.get(
                source_name, thresholds_by_type.get(source_type, threshold_default)
            )
            if result.score < threshold:
                continue
            content = result.document.content.strip()
            if not content:
                continue
            contexts.append(
                ContextChunk(
                    document_id=result.document.doc_id,
                    content=content,
                    metadata=result.document.metadata,
                    score=result.score,
                )
            )
            if limit is not None and len(contexts) >= limit:
                break
        return contexts

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        min_score_by_type: dict[str, float] | None = None,
        min_score_by_source: dict[str, float] | None = None,
        source_types: list[str] | None = None,
        source_names: list[str] | None = None,
        tenant_id: str | None = None,
        source_filter_mode: str = "and",
    ) -> RAGResponse:
        base_limit = top_k or self.max_chunks
        retrieval_limit = base_limit
        if source_types or source_names:
            retrieval_limit = min(base_limit * 5, 50)
        results = self.retrieve(
            query,
            top_k=retrieval_limit,
            source_types=source_types,
            source_names=source_names,
            tenant_id=tenant_id,
            source_filter_mode=source_filter_mode,
        )
        contexts = self.build_context(
            results,
            min_score=min_score,
            min_score_by_type=min_score_by_type,
            min_score_by_source=min_score_by_source,
            source_types=source_types,
            source_names=source_names,
            source_filter_mode=source_filter_mode,
            limit=base_limit,
        )
        guardrail = require_context(contexts)
        if not guardrail.allowed:
            return RAGResponse(answer=DEFAULT_REFUSAL, sources=[], refusal_reason=guardrail.reason)
        answer = self.answerer.generate(query, contexts)
        if not answer:
            return RAGResponse(answer=DEFAULT_REFUSAL, sources=[], refusal_reason="empty_answer")
        return RAGResponse(answer=answer, sources=contexts)
