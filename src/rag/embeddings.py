from __future__ import annotations

"""Embedding providers and configuration validation."""

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class EmbeddingError(RuntimeError):
    """Raised when embeddings fail or are invalid."""
    pass


class EmbeddingConfigError(RuntimeError):
    """Raised when embedding configuration is invalid."""
    pass


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    dimension: int

    def embed(self, text: str) -> list[float]:
        """Return an embedding vector for the provided text."""
        raise NotImplementedError


def validate_vector(vector: list[float], dimension: int) -> list[float]:
    """Validate and normalize embedding vectors."""
    if len(vector) != dimension:
        raise EmbeddingError(
            f"Embedding dimension mismatch: expected {dimension}, got {len(vector)}"
        )
    cleaned: list[float] = []
    for value in vector:
        if not isinstance(value, (int, float)):
            raise EmbeddingError("Embedding contains a non-numeric value")
        if not math.isfinite(value):
            raise EmbeddingError("Embedding contains a non-finite value")
        cleaned.append(float(value))
    return cleaned


@dataclass
class HashEmbedder:
    """Deterministic hash-based embedder for testing or offline use."""
    dimension: int = 256

    def embed(self, text: str) -> list[float]:
        """Embed text using token hashing and L2 normalization."""
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return validate_vector([0.0] * self.dimension, self.dimension)
        vector = [0.0] * self.dimension
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = digest[0] % self.dimension
            vector[idx] += 1.0
        return validate_vector(self._l2_normalize(vector), self.dimension)

    def _l2_normalize(self, vector: list[float]) -> list[float]:
        """Normalize vector magnitude to 1.0."""
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


def resolve_openai_dimension(model: str) -> int | None:
    """Return expected dimension for OpenAI embedding model."""
    mapping = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    return mapping.get(model)


def resolve_gemini_dimension(model: str) -> int | None:
    """Return expected dimension for Gemini embedding model."""
    return None


@dataclass(frozen=True)
class EmbeddingConfigReport:
    """Validation report for embedding configuration."""
    provider: str
    model: str | None
    configured_dimension: int
    expected_dimension: int | None
    ok: bool
    status: str
    detail: str | None = None
    action: str | None = None


def build_embedding_config_report(
    provider: str, model: str | None, dimension: int
) -> EmbeddingConfigReport:
    """Build a validation report for embedding settings."""
    normalized = provider.lower().strip()
    expected: int | None = None

    if normalized in {"", "hash"}:
        if dimension <= 0:
            return EmbeddingConfigReport(
                provider="hash",
                model=None,
                configured_dimension=dimension,
                expected_dimension=None,
                ok=False,
                status="error",
                detail="EMBEDDING_DIMENSION must be greater than zero for hash embeddings.",
                action="Set EMBEDDING_DIMENSION to a positive integer.",
            )
        return EmbeddingConfigReport(
            provider="hash",
            model=None,
            configured_dimension=dimension,
            expected_dimension=dimension,
            ok=True,
            status="ok",
        )

    if normalized == "openai":
        if not model:
            return EmbeddingConfigReport(
                provider="openai",
                model=None,
                configured_dimension=dimension,
                expected_dimension=None,
                ok=False,
                status="error",
                detail="OPENAI_EMBEDDING_MODEL is required for OpenAI embeddings.",
                action="Set OPENAI_EMBEDDING_MODEL in .env.",
            )
        expected = resolve_openai_dimension(model)
        if dimension <= 0:
            if expected is not None:
                return EmbeddingConfigReport(
                    provider="openai",
                    model=model,
                    configured_dimension=dimension,
                    expected_dimension=expected,
                    ok=False,
                    status="error",
                    detail="EMBEDDING_DIMENSION is missing for the configured OpenAI model.",
                    action=f"Set EMBEDDING_DIMENSION to {expected}.",
                )
            return EmbeddingConfigReport(
                provider="openai",
                model=model,
                configured_dimension=dimension,
                expected_dimension=None,
                ok=False,
                status="error",
                detail="EMBEDDING_DIMENSION must be set for the configured OpenAI model.",
                action="Set EMBEDDING_DIMENSION based on the OpenAI model documentation.",
            )
        if expected is not None and dimension != expected:
            return EmbeddingConfigReport(
                provider="openai",
                model=model,
                configured_dimension=dimension,
                expected_dimension=expected,
                ok=False,
                status="error",
                detail="EMBEDDING_DIMENSION does not match the OpenAI model dimension.",
                action=f"Set EMBEDDING_DIMENSION to {expected}.",
            )
        if expected is None:
            return EmbeddingConfigReport(
                provider="openai",
                model=model,
                configured_dimension=dimension,
                expected_dimension=None,
                ok=True,
                status="warning",
                detail="Model dimension cannot be auto-validated. Confirm EMBEDDING_DIMENSION manually.",
            )
        return EmbeddingConfigReport(
            provider="openai",
            model=model,
            configured_dimension=dimension,
            expected_dimension=expected,
            ok=True,
            status="ok",
        )

    if normalized in {"gemini", "google"}:
        if not model:
            return EmbeddingConfigReport(
                provider="gemini",
                model=None,
                configured_dimension=dimension,
                expected_dimension=None,
                ok=False,
                status="error",
                detail="GEMINI_EMBEDDING_MODEL is required for Gemini embeddings.",
                action="Set GEMINI_EMBEDDING_MODEL in .env.",
            )
        expected = resolve_gemini_dimension(model)
        if dimension <= 0:
            return EmbeddingConfigReport(
                provider="gemini",
                model=model,
                configured_dimension=dimension,
                expected_dimension=expected,
                ok=False,
                status="error",
                detail="EMBEDDING_DIMENSION must be set for the configured Gemini model.",
                action="Set EMBEDDING_DIMENSION based on the Gemini model documentation.",
            )
        if expected is not None and dimension != expected:
            return EmbeddingConfigReport(
                provider="gemini",
                model=model,
                configured_dimension=dimension,
                expected_dimension=expected,
                ok=False,
                status="error",
                detail="EMBEDDING_DIMENSION does not match the Gemini model dimension.",
                action=f"Set EMBEDDING_DIMENSION to {expected}.",
            )
        if expected is None:
            return EmbeddingConfigReport(
                provider="gemini",
                model=model,
                configured_dimension=dimension,
                expected_dimension=None,
                ok=True,
                status="warning",
                detail="Model dimension cannot be auto-validated. Confirm EMBEDDING_DIMENSION manually.",
            )
        return EmbeddingConfigReport(
            provider="gemini",
            model=model,
            configured_dimension=dimension,
            expected_dimension=expected,
            ok=True,
            status="ok",
        )

    return EmbeddingConfigReport(
        provider=normalized,
        model=model,
        configured_dimension=dimension,
        expected_dimension=None,
        ok=False,
        status="error",
        detail="Unsupported embedding provider.",
        action="Set EMBEDDING_PROVIDER to hash, openai, or gemini.",
    )


@dataclass
class OpenAIEmbedder:
    """Embedding provider using OpenAI embeddings API."""
    api_key: str
    model: str
    dimension: int
    client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate OpenAI configuration and create a client."""
        if not self.api_key:
            raise EmbeddingConfigError("OPENAI_API_KEY is required for OpenAIEmbedder")
        if not self.model:
            raise EmbeddingConfigError("OPENAI_EMBEDDING_MODEL is required for OpenAIEmbedder")
        if self.dimension <= 0:
            resolved = resolve_openai_dimension(self.model)
            if resolved is None:
                raise EmbeddingConfigError(
                    "EMBEDDING_DIMENSION must be set for OpenAI embeddings when model is unknown"
                )
            self.dimension = resolved
        else:
            resolved = resolve_openai_dimension(self.model)
            if resolved is not None and self.dimension != resolved:
                raise EmbeddingConfigError(
                    f"EMBEDDING_DIMENSION should be {resolved} for model {self.model}"
                )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise EmbeddingError("openai package is required for OpenAIEmbedder") from exc
        self.client = OpenAI(api_key=self.api_key)

    def embed(self, text: str) -> list[float]:
        """Embed text using the OpenAI embeddings API."""
        response = self.client.embeddings.create(model=self.model, input=text)
        vector = list(response.data[0].embedding)
        return validate_vector(vector, self.dimension)


@dataclass
class GeminiEmbedder:
    """Embedding provider using Gemini embeddings API."""
    api_key: str
    model: str
    dimension: int
    client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate Gemini configuration and create a client."""
        if not self.api_key:
            raise EmbeddingConfigError("GEMINI_API_KEY is required for GeminiEmbedder")
        if not self.model:
            raise EmbeddingConfigError("GEMINI_EMBEDDING_MODEL is required for GeminiEmbedder")
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise EmbeddingError("google-generativeai package is required for GeminiEmbedder") from exc
        genai.configure(api_key=self.api_key)
        self.client = genai

    def embed(self, text: str) -> list[float]:
        """Embed text using the Gemini embeddings API."""
        result = self.client.embed_content(model=self.model, content=text)
        embedding = None
        if isinstance(result, dict):
            embedding = result.get("embedding")
        if embedding is None:
            embedding = getattr(result, "embedding", None)
        if embedding is None:
            raise EmbeddingError("Gemini embedding response missing embedding vector")
        return validate_vector(list(embedding), self.dimension)
