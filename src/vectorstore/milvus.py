from __future__ import annotations

"""Milvus-backed vector store with optional hybrid search."""

import json
from dataclasses import dataclass
from typing import Any, Iterable

from src.rag.embeddings import EmbeddingConfigError, EmbeddingProvider
from src.rag.types import Document, SearchResult


class MilvusDependencyError(RuntimeError):
    """Raised when Milvus dependencies are missing."""
    pass


@dataclass
class MilvusConfig:
    """Configuration for Milvus connection and indexing."""
    uri: str
    token: str | None
    collection: str
    consistency: str
    index_type: str
    metric_type: str
    nlist: int
    nprobe: int
    hnsw_m: int
    hnsw_ef_construction: int
    hnsw_ef: int
    sparse_index_algo: str
    hybrid_search: bool
    max_content_length: int = 65535


@dataclass
class MilvusVectorStore:
    """Milvus vector store with dense and sparse (BM25) fields."""
    embedder: EmbeddingProvider
    config: MilvusConfig

    def __post_init__(self) -> None:
        """Connect to Milvus and ensure collection exists."""
        try:
            from pymilvus import connections
        except ImportError as exc:
            raise MilvusDependencyError("pymilvus is required for MilvusVectorStore") from exc
        if self.embedder.dimension <= 0:
            raise EmbeddingConfigError(
                "Embedding dimension must be set before initializing MilvusVectorStore"
            )
        connections.connect(
            alias="default",
            uri=self.config.uri,
            token=self.config.token,
        )
        self.ensure_collection()

    def ensure_collection(self) -> None:
        """Create collection schema and indexes when missing."""
        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            Function,
            FunctionType,
            utility,
        )

        if utility.has_collection(self.config.collection):
            self.collection = Collection(self.config.collection, consistency_level=self.config.consistency)
            existing_dim = self._existing_embedding_dim()
            if existing_dim is not None and existing_dim != self.embedder.dimension:
                raise EmbeddingConfigError(
                    "Milvus collection embedding dimension mismatch: "
                    f"{existing_dim} (collection) vs {self.embedder.dimension} (embedder). "
                    "Update EMBEDDING_DIMENSION or use a new MILVUS_COLLECTION."
                )
            return

        metadata_field = self._metadata_field(DataType, FieldSchema)

        fields = [
            FieldSchema(
                name="doc_id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                max_length=256,
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=self.config.max_content_length,
                enable_analyzer=True,
            ),
            metadata_field,
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.embedder.dimension,
            ),
        ]
        functions = []
        if self.config.hybrid_search:
            fields.append(
                FieldSchema(
                    name="text_sparse",
                    dtype=DataType.SPARSE_FLOAT_VECTOR,
                )
            )
            functions.append(
                Function(
                    name="content_bm25",
                    input_field_names=["content"],
                    output_field_names=["text_sparse"],
                    function_type=FunctionType.BM25,
                )
            )
        schema = CollectionSchema(
            fields=fields,
            description="Business RAG documents",
            functions=functions,
        )
        self.collection = Collection(
            self.config.collection,
            schema,
            consistency_level=self.config.consistency,
        )
        self._create_index()

    def _metadata_field(self, data_type: Any, field_schema: Any):
        """Build metadata field with JSON support when available."""
        if hasattr(data_type, "JSON"):
            return field_schema(
                name="metadata",
                dtype=data_type.JSON,
            )
        return field_schema(
            name="metadata",
            dtype=data_type.VARCHAR,
            max_length=8192,
        )

    def _create_index(self) -> None:
        """Create dense and sparse indexes on the collection."""
        index_params = {
            "index_type": self.config.index_type,
            "metric_type": self.config.metric_type,
            "params": {"nlist": self.config.nlist},
        }
        if self.config.index_type.upper() == "HNSW":
            index_params = {
                "index_type": "HNSW",
                "metric_type": self.config.metric_type,
                "params": {
                    "M": self.config.hnsw_m,
                    "efConstruction": self.config.hnsw_ef_construction,
                },
            }
        try:
            self.collection.create_index(field_name="embedding", index_params=index_params)
        except Exception:
            # Collection may already have an index or use a different config.
            pass
        if self.config.hybrid_search:
            try:
                self.collection.create_index(
                    field_name="text_sparse",
                    index_params={
                        "index_type": "SPARSE_INVERTED_INDEX",
                        "metric_type": "BM25",
                        "params": {"inverted_index_algo": self.config.sparse_index_algo},
                    },
                )
            except Exception:
                pass

    def _existing_embedding_dim(self) -> int | None:
        """Read embedding dimension from existing collection schema."""
        try:
            fields = self.collection.schema.fields
        except Exception:
            return None
        for field in fields:
            if field.name != "embedding":
                continue
            dim = None
            params = getattr(field, "params", None)
            if isinstance(params, dict):
                dim = params.get("dim")
            if dim is None:
                dim = getattr(field, "dim", None)
            if dim is None:
                return None
            try:
                return int(dim)
            except (TypeError, ValueError):
                return None
        return None

    def add_documents(self, documents: Iterable[Document]) -> int:
        """Insert documents with dense embeddings and BM25 text."""
        rows: list[dict[str, Any]] = []
        for document in documents:
            content = document.content[: self.config.max_content_length]
            rows.append(
                {
                    "doc_id": document.doc_id,
                    "content": content,
                    "metadata": self._serialize_metadata(document.metadata),
                    "embedding": self.embedder.embed(content),
                }
            )

        if not rows:
            return 0

        self.collection.insert(rows)
        self.collection.flush()
        return len(rows)

    def search(
        self,
        query: str,
        top_k: int = 4,
        source_types: list[str] | None = None,
        source_names: list[str] | None = None,
        tenant_id: str | None = None,
        source_filter_mode: str = "and",
    ) -> list[SearchResult]:
        """Search using dense or hybrid (dense+BM25) retrieval."""
        if top_k <= 0:
            return []
        query_vector = self.embedder.embed(query)
        self.collection.load()
        expr = self._build_filter_expr(source_types, source_names, tenant_id, source_filter_mode)

        if self.config.hybrid_search:
            from pymilvus import AnnSearchRequest, RRFRanker

            dense_param = {
                "metric_type": self.config.metric_type,
                "params": {"ef": self.config.hnsw_ef}
                if self.config.index_type.upper() == "HNSW"
                else {"nprobe": self.config.nprobe},
            }
            req_dense = AnnSearchRequest(
                data=[query_vector],
                anns_field="embedding",
                param=dense_param,
                limit=top_k,
                expr=expr,
            )
            req_sparse = AnnSearchRequest(
                data=[query],
                anns_field="text_sparse",
                param={"metric_type": "BM25"},
                limit=top_k,
                expr=expr,
            )
            results = self.collection.hybrid_search(
                [req_dense, req_sparse],
                RRFRanker(),
                limit=top_k,
                output_fields=["doc_id", "content", "metadata"],
            )
        else:
            search_params = {"metric_type": self.config.metric_type, "params": {"nprobe": self.config.nprobe}}
            if self.config.index_type.upper() == "HNSW":
                search_params = {
                    "metric_type": self.config.metric_type,
                    "params": {"ef": self.config.hnsw_ef},
                }
            results = self.collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["doc_id", "content", "metadata"],
            )

        search_results: list[SearchResult] = []
        for hit in results[0]:
            entity = hit.entity
            metadata = self._deserialize_metadata(entity.get("metadata"))
            if tenant_id and tenant_id != "*":
                if str(metadata.get("tenant_id", "")).lower() != tenant_id.lower():
                    continue
            document = Document(
                doc_id=entity.get("doc_id"),
                content=entity.get("content"),
                metadata=metadata,
            )
            search_results.append(SearchResult(document=document, score=float(hit.score)))
        return search_results

    def _serialize_metadata(self, metadata: dict[str, Any]) -> Any:
        """Serialize metadata for storage."""
        if not metadata:
            return {} if self._supports_json() else "{}"
        if self._supports_json():
            return metadata
        return json.dumps(metadata, ensure_ascii=True)

    def _deserialize_metadata(self, value: Any) -> dict[str, Any]:
        """Deserialize metadata from storage."""
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {"raw": value}
        return {"raw": value}

    def _supports_json(self) -> bool:
        """Return True if Milvus supports JSON fields."""
        try:
            from pymilvus import DataType
        except Exception:
            return False
        return hasattr(DataType, "JSON")

    def _build_filter_expr(
        self,
        source_types: list[str] | None,
        source_names: list[str] | None,
        tenant_id: str | None,
        mode: str,
    ) -> str | None:
        """Build Milvus filter expression for metadata."""
        if not self._supports_json():
            return None
        clauses: list[str] = []
        if tenant_id and tenant_id != "*":
            clauses.append(f'(metadata["tenant_id"] == "{tenant_id}")')
        type_values = [value.lower() for value in source_types or [] if value]
        name_values = [value.lower() for value in source_names or [] if value]
        if type_values:
            clauses.append(f"({self._json_in_expr('source_type', type_values)})")
        if name_values:
            expanded_names = set(name_values)
            for source_type in type_values:
                for name in name_values:
                    if ":" not in name:
                        expanded_names.add(f"{source_type}:{name}")
            name_clause = self._json_in_expr("source_name", sorted(expanded_names))
            source_clause = self._json_in_expr("source", sorted(expanded_names))
            clauses.append(f"({name_clause} or {source_clause})")
        if not clauses:
            return None
        joiner = " or " if mode == "or" else " and "
        return joiner.join(clauses)

    def _json_in_expr(self, key: str, values: list[str]) -> str:
        """Build a JSON IN expression for metadata filters."""
        quoted = ", ".join(f'\"{value}\"' for value in values)
        return f'metadata["{key}"] in [{quoted}]'

    def stats(self) -> dict[str, int | str]:
        """Return collection stats."""
        try:
            count = int(self.collection.num_entities)
        except Exception:
            count = 0
        return {
            "backend": "milvus",
            "document_count": count,
            "embedding_dimension": self.embedder.dimension,
            "collection": self.config.collection,
        }

    def health(self) -> dict[str, str | bool]:
        """Return collection health info."""
        try:
            _ = self.collection.num_entities
        except Exception as exc:
            return {
                "backend": "milvus",
                "ok": False,
                "detail": str(exc),
            }
        return {
            "backend": "milvus",
            "ok": True,
            "collection": self.config.collection,
        }

    def delete_by_source(
        self,
        source_types: list[str] | None,
        source_names: list[str] | None,
        tenant_id: str | None = None,
        source_filter_mode: str = "and",
    ) -> int:
        """Delete documents matching source filters."""
        expr = self._build_filter_expr(
            source_types, source_names, tenant_id, source_filter_mode
        )
        if not expr:
            raise EmbeddingConfigError(
                "Delete filter requires JSON metadata support and at least one filter."
            )
        result = self.collection.delete(expr)
        self.collection.flush()
        try:
            return int(result.delete_count)
        except Exception:
            return 0
