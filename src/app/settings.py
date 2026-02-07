from __future__ import annotations

"""Settings loader for environment-driven configuration."""

import json
import os
import sys
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in minimal setups
    load_dotenv = None

if (
    load_dotenv is not None
    and not os.getenv("RAG_DISABLE_DOTENV")
    and "PYTEST_CURRENT_TEST" not in os.environ
    and "pytest" not in sys.modules
):
    load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Application settings sourced from environment variables."""
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
    milvus_hnsw_m: int = int(os.getenv("MILVUS_HNSW_M", "30"))
    milvus_hnsw_ef_construction: int = int(os.getenv("MILVUS_HNSW_EF_CONSTRUCTION", "360"))
    milvus_hnsw_ef: int = int(os.getenv("MILVUS_HNSW_EF", "64"))
    milvus_sparse_index_algo: str = os.getenv("MILVUS_SPARSE_INDEX_ALGO", "DAAT_MAXSCORE")
    hybrid_search_enabled: bool = os.getenv("RAG_HYBRID_SEARCH", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
    chunk_tokens: int = int(os.getenv("RAG_CHUNK_TOKENS", "0"))
    chunk_token_overlap: int = int(os.getenv("RAG_CHUNK_TOKEN_OVERLAP", "0"))
    tokenizer_encoding: str = os.getenv("RAG_TOKENIZER_ENCODING", "cl100k_base")
    metrics_enabled: bool = os.getenv("RAG_METRICS_ENABLED", "true").lower() in {"1", "true", "yes"}
    file_max_bytes: int = int(os.getenv("RAG_FILE_MAX_BYTES", "10485760"))
    api_keys_raw: str = os.getenv("RAG_API_KEYS", "")
    api_key_map_raw: str = os.getenv("RAG_API_KEY_MAP", "")
    default_tenant_id: str = os.getenv("RAG_DEFAULT_TENANT_ID", "default")
    allow_anonymous: bool = os.getenv("RAG_ALLOW_ANONYMOUS", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    answerer_mode_raw: str = os.getenv("RAG_ANSWERER", "extractive")
    llm_provider: str = os.getenv("RAG_LLM_PROVIDER", "ollama")
    llm_context_max_chars: int = int(os.getenv("RAG_LLM_CONTEXT_MAX_CHARS", "12000"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")
    ollama_temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
    ollama_max_tokens: int = int(os.getenv("OLLAMA_MAX_TOKENS", "512"))
    ollama_timeout: float = float(os.getenv("OLLAMA_TIMEOUT", "60"))
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_chat_model: str | None = os.getenv("OPENAI_CHAT_MODEL")
    openai_timeout: float = float(os.getenv("OPENAI_TIMEOUT", "30"))
    gemini_chat_model: str | None = os.getenv("GEMINI_CHAT_MODEL")
    gemini_timeout: float = float(os.getenv("GEMINI_TIMEOUT", "30"))
    llm_gate_enabled: bool = os.getenv("RAG_LLM_GATE_ENABLED", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    llm_gate_max_tokens: int = int(os.getenv("RAG_LLM_GATE_MAX_TOKENS", "128"))
    llm_gate_timeout: float = float(os.getenv("RAG_LLM_GATE_TIMEOUT", "20"))
    llm_fallback_mode: str = os.getenv("RAG_LLM_FALLBACK", "extractive")
    source_weights_raw: str = os.getenv("RAG_SOURCE_WEIGHTS", "")
    source_name_weights_raw: str = os.getenv("RAG_SOURCE_NAME_WEIGHTS", "")
    query_rewriter_raw: str = os.getenv("RAG_QUERY_REWRITER", "off")
    query_rewriter_provider: str = os.getenv("RAG_QUERY_REWRITER_PROVIDER", "ollama")
    query_rewriter_model: str = os.getenv("RAG_QUERY_REWRITER_MODEL", "")
    query_rewriter_timeout: float = float(os.getenv("RAG_QUERY_REWRITER_TIMEOUT", "10"))
    query_rewriter_max_tokens: int = int(os.getenv("RAG_QUERY_REWRITER_MAX_TOKENS", "64"))
    router_mode: str = os.getenv("RAG_ROUTER_MODE", "rules")
    router_provider: str = os.getenv("RAG_ROUTER_PROVIDER", "ollama")
    router_model: str = os.getenv("RAG_ROUTER_MODEL", "")
    router_timeout: float = float(os.getenv("RAG_ROUTER_TIMEOUT", "10"))
    router_max_tokens: int = int(os.getenv("RAG_ROUTER_MAX_TOKENS", "64"))
    log_level: str = os.getenv("RAG_LOG_LEVEL", "INFO")
    metadata_db_uri: str | None = os.getenv("RAG_METADATA_DB_URI")
    audit_db_uri: str | None = os.getenv("RAG_AUDIT_DB_URI")
    chat_history_turns: int = int(os.getenv("RAG_CHAT_HISTORY_TURNS", "6"))
    demo_role_header: bool = os.getenv("RAG_DEMO_ROLE_HEADER", "false").lower() in {
        "1",
        "true",
        "yes",
    }


    @property
    def api_keys(self) -> set[str]:
        """Return the set of API keys allowed to access the service."""
        raw = os.getenv("RAG_API_KEYS", self.api_keys_raw)
        return {value.strip() for value in raw.split(",") if value.strip()}

    @property
    def api_key_map(self) -> dict[str, dict[str, str]]:
        """Return API key to role/tenant mappings parsed from JSON."""
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
        """Return the configured answerer mode, supporting runtime overrides."""
        return os.getenv("RAG_ANSWERER", self.answerer_mode_raw)

    @property
    def source_weights(self) -> dict[str, float]:
        """Return per-source-type weights for scoring adjustments."""
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
        """Return per-source-name weights for scoring adjustments."""
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
    def min_score_by_type(self) -> dict[str, float]:
        """Return per-source-type score thresholds."""
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
        """Return per-source-name score thresholds."""
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

    @property
    def query_rewriter_enabled(self) -> bool:
        """Return True when query rewriting is enabled."""
        raw = os.getenv("RAG_QUERY_REWRITER", self.query_rewriter_raw)
        return raw.strip().lower() in {"1", "true", "yes", "on", "enabled"}


settings = Settings()
