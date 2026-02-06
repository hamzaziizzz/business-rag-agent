from __future__ import annotations

"""Query rewriting helpers for retrieval optimization."""

from dataclasses import dataclass

import asyncio
import httpx

from src.app.settings import settings


class QueryRewriteError(RuntimeError):
    """Raised when query rewriting fails."""
    pass


class QueryRewriter:
    """Base class for query rewriters."""
    async def rewrite(self, query: str) -> str:
        """Return a rewritten query or the original if unchanged."""
        raise NotImplementedError


@dataclass(frozen=True)
class NoopRewriter(QueryRewriter):
    """Rewriter that returns the input unchanged."""
    async def rewrite(self, query: str) -> str:
        """Return the input query without modification."""
        return query


@dataclass(frozen=True)
class OllamaQueryRewriter(QueryRewriter):
    """Rewriter backed by Ollama."""
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout: float

    async def rewrite(self, query: str) -> str:
        """Rewrite a query using the Ollama chat API."""
        if not query.strip():
            return query
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Rewrite the user query for semantic search. "
                        "Preserve intent, remove filler, keep it concise. "
                        "Return only the rewritten query text."
                    ),
                },
                {"role": "user", "content": query},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise QueryRewriteError(str(exc)) from exc

        message = data.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise QueryRewriteError("Invalid query rewriter response")
        rewritten = content.strip()
        if not rewritten:
            return query
        return rewritten


@dataclass(frozen=True)
class OpenAIQueryRewriter(QueryRewriter):
    """Rewriter backed by OpenAI chat completions."""
    api_key: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout: float

    async def rewrite(self, query: str) -> str:
        """Rewrite a query using OpenAI chat completions."""
        if not query.strip():
            return query
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Rewrite the user query for semantic search. "
                        "Preserve intent, remove filler, keep it concise. "
                        "Return only the rewritten query text."
                    ),
                },
                {"role": "user", "content": query},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise QueryRewriteError(str(exc)) from exc

        choices = data.get("choices") or []
        if not choices:
            raise QueryRewriteError("Invalid OpenAI response")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise QueryRewriteError("Invalid OpenAI response content")
        rewritten = content.strip()
        return rewritten or query


@dataclass(frozen=True)
class GeminiQueryRewriter(QueryRewriter):
    """Rewriter backed by Gemini models."""
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    timeout: float

    async def rewrite(self, query: str) -> str:
        """Rewrite a query using the Gemini API."""
        if not query.strip():
            return query
        prompt = (
            "Rewrite the user query for semantic search. "
            "Preserve intent, remove filler, keep it concise. "
            "Return only the rewritten query text.\n\n"
            f"Query: {query}"
        )
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise QueryRewriteError("google-generativeai is required for Gemini rewriter") from exc

        def _run() -> str:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                },
            )
            return getattr(response, "text", "") or ""

        try:
            content = await asyncio.wait_for(asyncio.to_thread(_run), timeout=self.timeout)
        except Exception as exc:
            raise QueryRewriteError(str(exc)) from exc
        rewritten = content.strip()
        return rewritten or query


def build_rewriter() -> QueryRewriter:
    """Factory for query rewriters based on settings."""
    if not settings.query_rewriter_enabled:
        return NoopRewriter()
    provider = settings.query_rewriter_provider.strip().lower()
    if provider in {"", "ollama"}:
        model = settings.query_rewriter_model or settings.ollama_model
        return OllamaQueryRewriter(
            base_url=settings.ollama_base_url,
            model=model,
            temperature=settings.ollama_temperature,
            max_tokens=settings.query_rewriter_max_tokens,
            timeout=settings.query_rewriter_timeout,
        )
    if provider in {"openai"}:
        model = settings.query_rewriter_model or settings.openai_chat_model
        if not settings.openai_api_key or not model:
            raise QueryRewriteError("OpenAI rewriter requires OPENAI_API_KEY and OPENAI_CHAT_MODEL")
        return OpenAIQueryRewriter(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url.rstrip("/"),
            model=model,
            temperature=settings.ollama_temperature,
            max_tokens=settings.query_rewriter_max_tokens,
            timeout=settings.query_rewriter_timeout,
        )
    if provider in {"gemini", "google"}:
        model = settings.query_rewriter_model or settings.gemini_chat_model
        if not settings.gemini_api_key or not model:
            raise QueryRewriteError("Gemini rewriter requires GEMINI_API_KEY and GEMINI_CHAT_MODEL")
        return GeminiQueryRewriter(
            api_key=settings.gemini_api_key,
            model=model,
            temperature=settings.ollama_temperature,
            max_tokens=settings.query_rewriter_max_tokens,
            timeout=settings.query_rewriter_timeout,
        )
    return NoopRewriter()
