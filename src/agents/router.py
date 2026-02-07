from __future__ import annotations

"""Query routing logic for selecting RAG vs summarization."""

from dataclasses import dataclass
import asyncio
import json
import re

import httpx

from src.app.settings import settings


SUMMARY_HINTS = ("summarize", "summary", "tl;dr", "tldr")
_ALLOWED_TOOLS = {"rag", "summarize", "refuse"}
_JSON_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


@dataclass(frozen=True)
class RouteDecision:
    """Routing decision result."""
    tool: str
    reason: str


class AgentRouter:
    """Router that supports rule-based and LLM-based routing."""
    def route(self, query: str) -> RouteDecision:
        """Return a rule-based routing decision."""
        if not query.strip():
            return RouteDecision(tool="refuse", reason="empty_query")
        lowered = query.strip().lower()
        if any(hint in lowered for hint in SUMMARY_HINTS):
            return RouteDecision(tool="summarize", reason="summary_hint")
        return RouteDecision(tool="rag", reason="default_rag")

    async def route_async(self, query: str) -> RouteDecision:
        """Return an LLM-routed decision with rule-based fallback."""
        mode = settings.router_mode.strip().lower()
        if mode != "llm":
            return self.route(query)
        if not query.strip():
            return RouteDecision(tool="refuse", reason="empty_query")
        try:
            return await _route_with_llm(query)
        except Exception:
            return self.route(query)

    async def route_chat_async(
        self, query: str, history: list[dict[str, str]] | None = None
    ) -> "ChatRouteDecision":
        """Decide whether chat should retrieve context or refuse."""
        mode = settings.router_mode.strip().lower()
        if not query.strip():
            return ChatRouteDecision(retrieve=False, reason="empty_query")
        if mode != "llm":
            return ChatRouteDecision(retrieve=True, reason="default_retrieve")
        try:
            return await _route_chat_with_llm(query, history)
        except Exception:
            return ChatRouteDecision(retrieve=True, reason="fallback_retrieve")


async def _route_with_llm(query: str) -> RouteDecision:
    """Dispatch to the configured LLM router provider."""
    provider = settings.router_provider.strip().lower()
    model = settings.router_model
    if provider in {"", "ollama"}:
        model = model or settings.ollama_model
        return await _route_with_ollama(query, model)
    if provider == "openai":
        model = model or settings.openai_chat_model
        if not settings.openai_api_key or not model:
            raise RuntimeError("OpenAI router requires OPENAI_API_KEY and OPENAI_CHAT_MODEL")
        return await _route_with_openai(query, model)
    if provider in {"gemini", "google"}:
        model = model or settings.gemini_chat_model
        if not settings.gemini_api_key or not model:
            raise RuntimeError("Gemini router requires GEMINI_API_KEY and GEMINI_CHAT_MODEL")
        return await _route_with_gemini(query, model)
    raise RuntimeError("Unsupported router provider")


@dataclass(frozen=True)
class ChatRouteDecision:
    """Decision for chat retrieval."""
    retrieve: bool
    reason: str


def _router_prompt(query: str) -> str:
    """Build the classification prompt for routing."""
    return (
        "Classify the user query into one tool: rag, summarize, refuse.\n"
        "- Use summarize when the user asks to summarize or provide a TL;DR.\n"
        "- Use refuse only if the query is empty or meaningless.\n"
        "- Otherwise use rag.\n"
        "Return JSON only: {\"tool\": \"...\", \"reason\": \"...\"}.\n\n"
        f"Query: {query}"
    )


def _chat_router_prompt(query: str, history: list[dict[str, str]] | None) -> str:
    """Prompt for deciding if retrieval is required in chat."""
    history_block = ""
    if history:
        lines = []
        for item in history:
            role = item.get("role", "user").strip().lower()
            content = item.get("content", "").strip()
            if not content:
                continue
            if role not in {"user", "assistant", "system"}:
                role = "user"
            lines.append(f"{role}: {content}")
        if lines:
            history_block = "Conversation:\n" + "\n".join(lines) + "\n\n"
    return (
        "Decide if you must retrieve documents to answer the user's next message. "
        "Return JSON only: {\"retrieve\": true|false, \"reason\": \"...\"}.\n\n"
        f"{history_block}User: {query}"
    )


def _parse_router_json(content: str) -> RouteDecision:
    """Parse and validate router JSON output."""
    raw = content.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = _JSON_RE.search(raw)
        if not match:
            raise ValueError("Invalid router JSON")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Router JSON must be an object")
    tool = str(data.get("tool", "")).strip().lower()
    reason = str(data.get("reason", "")).strip() or "llm_router"
    if tool not in _ALLOWED_TOOLS:
        raise ValueError("Invalid tool from router")
    return RouteDecision(tool=tool, reason=reason)


def _parse_chat_router_json(content: str) -> ChatRouteDecision:
    """Parse chat retrieval decision from JSON output."""
    raw = content.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = _JSON_RE.search(raw)
        if not match:
            raise ValueError("Invalid chat router JSON")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Chat router JSON must be an object")
    retrieve = data.get("retrieve")
    if not isinstance(retrieve, bool):
        raise ValueError("Chat router JSON missing retrieve")
    reason = str(data.get("reason", "")).strip() or "chat_router"
    return ChatRouteDecision(retrieve=retrieve, reason=reason)


async def _route_with_ollama(query: str, model: str) -> RouteDecision:
    """Route using Ollama chat endpoint."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a routing classifier."},
            {"role": "user", "content": _router_prompt(query)},
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": settings.router_max_tokens,
        },
    }
    async with httpx.AsyncClient(timeout=settings.router_timeout) as client:
        response = await client.post(f"{settings.ollama_base_url}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
    message = data.get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Invalid Ollama router response")
    return _parse_router_json(content)


async def _route_chat_with_ollama(
    query: str, history: list[dict[str, str]] | None
) -> ChatRouteDecision:
    """Route chat retrieval using Ollama."""
    payload = {
        "model": settings.router_model or settings.ollama_model,
        "messages": [
            {"role": "system", "content": "You are a retrieval classifier."},
            {"role": "user", "content": _chat_router_prompt(query, history)},
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": settings.router_max_tokens,
        },
    }
    async with httpx.AsyncClient(timeout=settings.router_timeout) as client:
        response = await client.post(f"{settings.ollama_base_url}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
    message = data.get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Invalid Ollama chat router response")
    return _parse_chat_router_json(content)


async def _route_with_openai(query: str, model: str) -> RouteDecision:
    """Route using OpenAI chat completions."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a routing classifier."},
            {"role": "user", "content": _router_prompt(query)},
        ],
        "temperature": 0.0,
        "max_tokens": settings.router_max_tokens,
    }
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
    async with httpx.AsyncClient(timeout=settings.router_timeout) as client:
        response = await client.post(
            f"{settings.openai_base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("Invalid OpenAI router response")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Invalid OpenAI router content")
    return _parse_router_json(content)


async def _route_chat_with_openai(
    query: str, history: list[dict[str, str]] | None
) -> ChatRouteDecision:
    """Route chat retrieval using OpenAI chat completions."""
    model = settings.router_model or settings.openai_chat_model
    if not settings.openai_api_key or not model:
        raise RuntimeError("OpenAI chat router requires OPENAI_API_KEY and OPENAI_CHAT_MODEL")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a retrieval classifier."},
            {"role": "user", "content": _chat_router_prompt(query, history)},
        ],
        "temperature": 0.0,
        "max_tokens": settings.router_max_tokens,
    }
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
    async with httpx.AsyncClient(timeout=settings.router_timeout) as client:
        response = await client.post(
            f"{settings.openai_base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("Invalid OpenAI chat router response")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Invalid OpenAI chat router content")
    return _parse_chat_router_json(content)


async def _route_with_gemini(query: str, model: str) -> RouteDecision:
    """Route using Gemini generative model."""
    prompt = _router_prompt(query)
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError("google-generativeai is required for Gemini router") from exc

    def _run() -> str:
        genai.configure(api_key=settings.gemini_api_key)
        client = genai.GenerativeModel(model)
        response = client.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": settings.router_max_tokens,
            },
        )
        return getattr(response, "text", "") or ""

    content = await asyncio.wait_for(asyncio.to_thread(_run), timeout=settings.router_timeout)
    return _parse_router_json(content)


async def _route_chat_with_gemini(
    query: str, history: list[dict[str, str]] | None
) -> ChatRouteDecision:
    """Route chat retrieval using Gemini."""
    model = settings.router_model or settings.gemini_chat_model
    if not settings.gemini_api_key or not model:
        raise RuntimeError("Gemini chat router requires GEMINI_API_KEY and GEMINI_CHAT_MODEL")
    prompt = _chat_router_prompt(query, history)
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError("google-generativeai is required for Gemini chat router") from exc

    def _run() -> str:
        genai.configure(api_key=settings.gemini_api_key)
        client = genai.GenerativeModel(model)
        response = client.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": settings.router_max_tokens,
            },
        )
        return getattr(response, "text", "") or ""

    content = await asyncio.wait_for(asyncio.to_thread(_run), timeout=settings.router_timeout)
    return _parse_chat_router_json(content)


async def _route_chat_with_llm(
    query: str, history: list[dict[str, str]] | None
) -> ChatRouteDecision:
    """Dispatch to configured LLM for chat retrieval decision."""
    provider = settings.router_provider.strip().lower()
    if provider in {"", "ollama"}:
        return await _route_chat_with_ollama(query, history)
    if provider == "openai":
        return await _route_chat_with_openai(query, history)
    if provider in {"gemini", "google"}:
        return await _route_chat_with_gemini(query, history)
    raise RuntimeError("Unsupported chat router provider")
