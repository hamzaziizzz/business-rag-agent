from __future__ import annotations

from functools import lru_cache

from src.app.settings import settings
from src.rag.answerer import ExtractiveAnswerer
from src.rag.embeddings import (
    EmbeddingConfigError,
    EmbeddingProvider,
    GeminiEmbedder,
    HashEmbedder,
    OpenAIEmbedder,
    build_embedding_config_report,
    EmbeddingConfigReport,
)
from src.rag.pipeline import RAGPipeline
from src.metadata.store import MetadataStore
from src.metadata.audit import AuditStore
from src.vectorstore.inmemory import InMemoryVectorStore
from src.vectorstore.milvus import MilvusConfig, MilvusVectorStore


@lru_cache
def get_pipeline() -> RAGPipeline:
    embedder = build_embedder()
    vectorstore = build_vectorstore(embedder)
    answerer = ExtractiveAnswerer()
    return RAGPipeline(
        vectorstore=vectorstore,
        answerer=answerer,
        max_chunks=settings.max_chunks,
        min_score=settings.min_score,
        min_score_by_type=settings.min_score_by_type,
        min_score_by_source=settings.min_score_by_source,
        source_weights=settings.source_weights,
        source_name_weights=settings.source_name_weights,
    )


def reset_pipeline_cache() -> None:
    get_pipeline.cache_clear()


@lru_cache
def get_metadata_store() -> MetadataStore | None:
    if not settings.metadata_db_uri:
        return None
    return MetadataStore(settings.metadata_db_uri)


@lru_cache
def get_audit_store() -> AuditStore | None:
    if settings.audit_db_uri:
        return AuditStore(settings.audit_db_uri)
    if settings.metadata_db_uri:
        return AuditStore(settings.metadata_db_uri)
    return None


def get_embedding_config_report() -> EmbeddingConfigReport:
    provider = settings.embedding_provider
    model = None
    if provider.lower().strip() == "openai":
        model = settings.openai_embedding_model
    elif provider.lower().strip() in {"gemini", "google"}:
        model = settings.gemini_embedding_model
    return build_embedding_config_report(provider, model, settings.embedding_dimension)


def build_embedder() -> EmbeddingProvider:
    provider = settings.embedding_provider.lower().strip()
    if provider == "hash":
        return HashEmbedder(dimension=settings.embedding_dimension)
    if provider == "openai":
        return OpenAIEmbedder(
            api_key=settings.openai_api_key or "",
            model=settings.openai_embedding_model or "",
            dimension=settings.embedding_dimension,
        )
    if provider in {"gemini", "google"}:
        if settings.embedding_dimension <= 0:
            raise EmbeddingConfigError("EMBEDDING_DIMENSION must be set for Gemini embeddings")
        return GeminiEmbedder(
            api_key=settings.gemini_api_key or "",
            model=settings.gemini_embedding_model or "",
            dimension=settings.embedding_dimension,
        )
    raise EmbeddingConfigError(f"Unsupported embedding provider: {provider}")


def build_vectorstore(embedder: EmbeddingProvider) -> InMemoryVectorStore | MilvusVectorStore:
    backend = settings.vectorstore_backend.lower().strip()
    if backend == "milvus":
        config = MilvusConfig(
            uri=settings.milvus_uri,
            token=settings.milvus_token,
            collection=settings.milvus_collection,
            consistency=settings.milvus_consistency,
            index_type=settings.milvus_index_type,
            metric_type=settings.milvus_metric_type,
            nlist=settings.milvus_nlist,
            nprobe=settings.milvus_nprobe,
        )
        return MilvusVectorStore(embedder=embedder, config=config)
    return InMemoryVectorStore(embedder=embedder)
