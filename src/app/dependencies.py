from __future__ import annotations

"""Dependency providers and shared singletons for the app."""

from functools import lru_cache

from src.app.settings import settings
from src.rag.answerer import ExtractiveAnswerer
from src.rag.embeddings import (
    EmbeddingConfigError,
    EmbeddingProvider,
    GeminiEmbedder,
    HashEmbedder,
    OpenAIEmbedder,
)
from src.rag.pipeline import RAGPipeline
from src.vectorstore.inmemory import InMemoryVectorStore
from src.vectorstore.milvus import MilvusConfig, MilvusVectorStore


@lru_cache
def get_pipeline() -> RAGPipeline:
    """Build or return the cached RAG pipeline."""
    embedder = build_embedder()
    vectorstore = build_vectorstore(embedder)
    answerer = ExtractiveAnswerer()
    return RAGPipeline(
        vectorstore=vectorstore,
        answerer=answerer,
        max_chunks=settings.max_chunks,
        min_score=settings.min_score,
    )


def reset_pipeline_cache() -> None:
    """Clear cached pipeline to force rebuild."""
    get_pipeline.cache_clear()


def build_embedder() -> EmbeddingProvider:
    """Construct the embedding provider from settings."""
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
    """Construct the vector store backend based on settings."""
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
            hnsw_m=settings.milvus_hnsw_m,
            hnsw_ef_construction=settings.milvus_hnsw_ef_construction,
            hnsw_ef=settings.milvus_hnsw_ef,
            sparse_index_algo=settings.milvus_sparse_index_algo,
            hybrid_search=settings.hybrid_search_enabled,
        )
        return MilvusVectorStore(embedder=embedder, config=config)
    return InMemoryVectorStore(embedder=embedder)
