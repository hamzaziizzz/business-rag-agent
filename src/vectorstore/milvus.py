from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from src.rag.embeddings import EmbeddingConfigError, EmbeddingProvider
from src.rag.types import Document, SearchResult


class MilvusDependencyError(RuntimeError):
    pass


@dataclass
class MilvusConfig:
    uri: str
    token: str | None
    collection: str
    consistency: str
    index_type: str
    metric_type: str
    nlist: int
    nprobe: int
    max_content_length: int = 65535


@dataclass
class MilvusVectorStore:
    embedder: EmbeddingProvider
    config: MilvusConfig

    def __post_init__(self) -> None:
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
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

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
            ),
            metadata_field,
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.embedder.dimension,
            ),
        ]
        schema = CollectionSchema(fields=fields, description="Business RAG documents")
        self.collection = Collection(
            self.config.collection,
            schema,
            consistency_level=self.config.consistency,
        )
        self._create_index()

    def _metadata_field(self, data_type: Any, field_schema: Any):
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
        index_params = {
            "index_type": self.config.index_type,
            "metric_type": self.config.metric_type,
            "params": {"nlist": self.config.nlist},
        }
        try:
            self.collection.create_index(field_name="embedding", index_params=index_params)
        except Exception:
            # Collection may already have an index or use a different config.
            pass

    def _existing_embedding_dim(self) -> int | None:
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
        doc_ids: list[str] = []
        contents: list[str] = []
        metadata_values: list[Any] = []
        embeddings: list[list[float]] = []
        for document in documents:
            content = document.content[: self.config.max_content_length]
            doc_ids.append(document.doc_id)
            contents.append(content)
            metadata_values.append(self._serialize_metadata(document.metadata))
            embeddings.append(self.embedder.embed(content))

        if not doc_ids:
            return 0

        self.collection.insert([doc_ids, contents, metadata_values, embeddings])
        self.collection.flush()
        return len(doc_ids)

    def search(
        self,
        query: str,
        top_k: int = 4,
        source_types: list[str] | None = None,
        source_names: list[str] | None = None,
        tenant_id: str | None = None,
        source_filter_mode: str = "and",
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []
        query_vector = self.embedder.embed(query)
        self.collection.load()
        search_params = {"metric_type": self.config.metric_type, "params": {"nprobe": self.config.nprobe}}
        expr = self._build_filter_expr(source_types, source_names, tenant_id, source_filter_mode)
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
        if not metadata:
            return {} if self._supports_json() else "{}"
        if self._supports_json():
            return metadata
        return json.dumps(metadata, ensure_ascii=True)

    def _deserialize_metadata(self, value: Any) -> dict[str, Any]:
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
        quoted = ", ".join(f'\"{value}\"' for value in values)
        return f'metadata["{key}"] in [{quoted}]'

    def stats(self) -> dict[str, int | str]:
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
