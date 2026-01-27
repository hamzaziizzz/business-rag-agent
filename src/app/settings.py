from __future__ import annotations

import json
import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in minimal setups
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


@dataclass(frozen=True)
class Settings:
    max_chunks: int = int(os.getenv("RAG_MAX_CHUNKS", "4"))
    min_score: float = float(os.getenv("RAG_MIN_SCORE", "0.2"))
    min_score_by_type_raw: str = os.getenv("RAG_MIN_SCORE_BY_TYPE", "")
    min_score_by_source_raw: str = os.getenv("RAG_MIN_SCORE_BY_SOURCE", "")
    vectorstore_backend: str = os.getenv("RAG_VECTORSTORE", "memory")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "hash")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "256"))
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_embedding_model: str | None = os.getenv("OPENAI_EMBEDDING_MODEL")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    gemini_embedding_model: str | None = os.getenv("GEMINI_EMBEDDING_MODEL")
    milvus_uri: str = os.getenv("MILVUS_URI", "http://localhost:19530")
    milvus_token: str | None = os.getenv("MILVUS_TOKEN")
    milvus_collection: str = os.getenv("MILVUS_COLLECTION", "business_documents")
    milvus_consistency: str = os.getenv("MILVUS_CONSISTENCY", "Strong")
    milvus_index_type: str = os.getenv("MILVUS_INDEX_TYPE", "IVF_FLAT")
    milvus_metric_type: str = os.getenv("MILVUS_METRIC_TYPE", "COSINE")
    milvus_nlist: int = int(os.getenv("MILVUS_NLIST", "1024"))
    milvus_nprobe: int = int(os.getenv("MILVUS_NPROBE", "10"))
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
    db_max_rows: int = int(os.getenv("RAG_DB_MAX_ROWS", "1000"))
    api_timeout: float = float(os.getenv("RAG_API_TIMEOUT", "15"))
    api_max_bytes: int = int(os.getenv("RAG_API_MAX_BYTES", "1048576"))
    metrics_enabled: bool = os.getenv("RAG_METRICS_ENABLED", "true").lower() in {"1", "true", "yes"}
    object_store_endpoint_url: str | None = os.getenv("RAG_OBJECT_STORE_ENDPOINT_URL")
    object_store_region: str | None = os.getenv("RAG_OBJECT_STORE_REGION")
    object_store_access_key: str | None = os.getenv("RAG_OBJECT_STORE_ACCESS_KEY")
    object_store_secret_key: str | None = os.getenv("RAG_OBJECT_STORE_SECRET_KEY")
    object_store_session_token: str | None = os.getenv("RAG_OBJECT_STORE_SESSION_TOKEN")
    object_store_bucket: str | None = os.getenv("RAG_OBJECT_STORE_BUCKET")
    object_store_max_bytes: int = int(os.getenv("RAG_OBJECT_STORE_MAX_BYTES", "52428800"))
    db_chunk_size: int = int(os.getenv("RAG_DB_CHUNK_SIZE", "8000"))
    db_chunk_overlap: int = int(os.getenv("RAG_DB_CHUNK_OVERLAP", "0"))
    db_allowed_tables_raw: str = os.getenv("RAG_DB_ALLOWED_TABLES", "")
    db_allowed_columns_raw: str = os.getenv("RAG_DB_ALLOWED_COLUMNS", "")
    api_keys_raw: str = os.getenv("RAG_API_KEYS", "")
    api_key_map_raw: str = os.getenv("RAG_API_KEY_MAP", "")
    default_tenant_id: str = os.getenv("RAG_DEFAULT_TENANT_ID", "default")
    answerer_mode_raw: str = os.getenv("RAG_ANSWERER", "extractive")
    llm_provider: str = os.getenv("RAG_LLM_PROVIDER", "ollama")
    llm_context_max_chars: int = int(os.getenv("RAG_LLM_CONTEXT_MAX_CHARS", "12000"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")
    ollama_temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
    ollama_max_tokens: int = int(os.getenv("OLLAMA_MAX_TOKENS", "512"))
    ollama_timeout: float = float(os.getenv("OLLAMA_TIMEOUT", "60"))
    source_weights_raw: str = os.getenv("RAG_SOURCE_WEIGHTS", "")
    source_name_weights_raw: str = os.getenv("RAG_SOURCE_NAME_WEIGHTS", "")
    sql_database_uri_raw: str | None = os.getenv("RAG_SQL_DATABASE_URI")
    sql_max_rows: int = int(os.getenv("RAG_SQL_MAX_ROWS", "500"))
    sql_allowed_tables_raw: str = os.getenv("RAG_SQL_ALLOWED_TABLES", "")
    sql_allowed_columns_raw: str = os.getenv("RAG_SQL_ALLOWED_COLUMNS", "")
    metadata_db_uri: str | None = os.getenv("RAG_METADATA_DB_URI")
    audit_db_uri: str | None = os.getenv("RAG_AUDIT_DB_URI")

    @property
    def db_allowed_tables(self) -> set[str]:
        raw = os.getenv("RAG_DB_ALLOWED_TABLES", self.db_allowed_tables_raw)
        return {value.strip().lower() for value in raw.split(",") if value.strip()}

    @property
    def db_allowed_columns(self) -> set[str]:
        raw = os.getenv("RAG_DB_ALLOWED_COLUMNS", self.db_allowed_columns_raw)
        return {value.strip().lower() for value in raw.split(",") if value.strip()}

    @property
    def api_keys(self) -> set[str]:
        raw = os.getenv("RAG_API_KEYS", self.api_keys_raw)
        return {value.strip() for value in raw.split(",") if value.strip()}

    @property
    def api_key_map(self) -> dict[str, dict[str, str]]:
        raw = os.getenv("RAG_API_KEY_MAP", self.api_key_map_raw).strip()
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}
        result: dict[str, dict[str, str]] = {}
        for key, value in data.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            role = value.get("role")
            tenant_id = value.get("tenant_id")
            if isinstance(role, str) and isinstance(tenant_id, str):
                result[key] = {"role": role, "tenant_id": tenant_id}
            elif isinstance(role, str):
                result[key] = {"role": role, "tenant_id": self.default_tenant_id}
            elif isinstance(tenant_id, str):
                result[key] = {"role": "reader", "tenant_id": tenant_id}
        return result

    @property
    def answerer_mode(self) -> str:
        return os.getenv("RAG_ANSWERER", self.answerer_mode_raw)

    @property
    def source_weights(self) -> dict[str, float]:
        mapping: dict[str, float] = {}
        raw = os.getenv("RAG_SOURCE_WEIGHTS", self.source_weights_raw).strip()
        if not raw:
            return mapping
        for part in raw.split(","):
            part = part.strip()
            if not part or "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if not key or not value:
                continue
            try:
                mapping[key] = float(value)
            except ValueError:
                continue
        return mapping

    @property
    def source_name_weights(self) -> dict[str, float]:
        mapping: dict[str, float] = {}
        raw = os.getenv("RAG_SOURCE_NAME_WEIGHTS", self.source_name_weights_raw).strip()
        if not raw:
            return mapping
        for part in raw.split(","):
            part = part.strip()
            if not part or "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if not key or not value:
                continue
            try:
                mapping[key] = float(value)
            except ValueError:
                continue
        return mapping

    @property
    def sql_allowed_tables(self) -> set[str]:
        raw = os.getenv("RAG_SQL_ALLOWED_TABLES", self.sql_allowed_tables_raw)
        return {value.strip().lower() for value in raw.split(",") if value.strip()}

    @property
    def sql_allowed_columns(self) -> set[str]:
        raw = os.getenv("RAG_SQL_ALLOWED_COLUMNS", self.sql_allowed_columns_raw)
        return {value.strip().lower() for value in raw.split(",") if value.strip()}

    @property
    def sql_database_uri(self) -> str | None:
        return os.getenv("RAG_SQL_DATABASE_URI", self.sql_database_uri_raw or "")

    @property
    def min_score_by_type(self) -> dict[str, float]:
        mapping: dict[str, float] = {}
        raw = os.getenv("RAG_MIN_SCORE_BY_TYPE", self.min_score_by_type_raw).strip()
        if not raw:
            return mapping
        for part in raw.split(","):
            part = part.strip()
            if not part or "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if not key or not value:
                continue
            try:
                mapping[key] = float(value)
            except ValueError:
                continue
        return mapping

    @property
    def min_score_by_source(self) -> dict[str, float]:
        mapping: dict[str, float] = {}
        raw = os.getenv("RAG_MIN_SCORE_BY_SOURCE", self.min_score_by_source_raw).strip()
        if not raw:
            return mapping
        for part in raw.split(","):
            part = part.strip()
            if not part or "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if not key or not value:
                continue
            try:
                mapping[key] = float(value)
            except ValueError:
                continue
        return mapping


settings = Settings()
