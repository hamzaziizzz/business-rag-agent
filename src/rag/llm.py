from __future__ import annotations

"""LLM answerers and gating utilities."""

from dataclasses import dataclass
import asyncio
import json
import logging
import re

import httpx

from src.rag.types import ContextChunk


class LLMError(RuntimeError):
    """Raised when LLM requests fail or responses are invalid."""
    pass


logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are a production RAG assistant. "
    "Answer only from the provided context. "
    "If the context is insufficient, say: "
    "\"I don't know based on the provided context.\" "
    "Do not use external knowledge. "
    "Write fluent, natural language answers and synthesize information. "
    "Do not copy long verbatim passages; keep any direct quotes under 25 words. "
    "Prefer a short paragraph, and use bullets only when it improves clarity. "
    "Keep the answer concise. "
    "Return JSON only with keys: "
    "\"answer\" (string), \"refusal_reason\" (string or null), "
    "\"source_ids\" (array of strings)."
)

_GATE_PROMPT = (
    "You are a strict RAG gatekeeper. "
    "Decide whether the provided context is sufficient to answer the user's question. "
    "If the context does not directly contain the answer or strong evidence, return insufficient. "
    "Return JSON only with keys: \"sufficient\" (boolean) and \"reason\" (string)."
)


def base_system_prompt() -> str:
    """Return the default system prompt for answer generation."""
    return _SYSTEM_PROMPT


@dataclass(frozen=True)
class LLMResult:
    """Parsed LLM response for answers and citations."""
    answer: str
    refusal_reason: str | None
    source_ids: list[str]
    raw: str


@dataclass(frozen=True)
class GateResult:
    """Parsed LLM gate decision."""
    sufficient: bool
    reason: str
    raw: str


@dataclass(frozen=True)
class OllamaAnswerer:
    """LLM answerer backed by Ollama chat API."""
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout: float
    context_max_chars: int
    system_prompt: str = _SYSTEM_PROMPT

    async def generate(
        self,
        query: str,
        contexts: list[ContextChunk],
        history: list[dict[str, str]] | None = None,
    ) -> LLMResult:
        """Generate a grounded answer using Ollama."""
        if not contexts:
            return LLMResult(answer="", refusal_reason="no_context", source_ids=[], raw="")
        async def _request(system_prompt: str, context_limit: int) -> str:
            context_block = _build_context_block(contexts, context_limit)
            history_block = _format_history(history)
            user_prompt = (
                f"{history_block}Question: {query}\n\n"
                f"Context:\n{context_block}\n\n"
                "Instructions: Use only the context above to answer. "
                "Respond with JSON only."
            )
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
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
                raise LLMError(str(exc)) from exc
            message = data.get("message") or {}
            content = message.get("content")
            if not isinstance(content, str):
                raise LLMError("Invalid LLM response")
            return content

        content = await _request(self.system_prompt, self.context_max_chars)
        try:
            parsed = _parse_json_response(content)
        except Exception as exc:
            if not _is_json_parse_error(exc):
                raise
            logger.warning(
                "llm_json_retry",
                extra={
                    "provider": "ollama",
                    "model": self.model,
                },
            )
            retry_prompt = _strict_system_prompt(self.system_prompt)
            retry_context = max(2000, self.context_max_chars // 2)
            content = await _request(retry_prompt, retry_context)
            parsed = _parse_json_response(content)
        answer = parsed.get("answer")
        if not isinstance(answer, str):
            raise LLMError("Invalid LLM JSON response: missing answer")
        refusal_reason = parsed.get("refusal_reason")
        if refusal_reason is not None and not isinstance(refusal_reason, str):
            raise LLMError("Invalid LLM JSON response: refusal_reason")
        source_ids = parsed.get("source_ids") or []
        if not isinstance(source_ids, list) or any(
            not isinstance(item, str) for item in source_ids
        ):
            raise LLMError("Invalid LLM JSON response: source_ids")
        return LLMResult(
            answer=answer.strip(),
            refusal_reason=refusal_reason,
            source_ids=source_ids,
            raw=content.strip(),
        )


@dataclass(frozen=True)
class OpenAIAnswerer:
    """LLM answerer backed by OpenAI chat completions."""
    api_key: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout: float
    context_max_chars: int
    system_prompt: str = _SYSTEM_PROMPT

    async def generate(
        self,
        query: str,
        contexts: list[ContextChunk],
        history: list[dict[str, str]] | None = None,
    ) -> LLMResult:
        """Generate a grounded answer using OpenAI chat completions."""
        if not contexts:
            return LLMResult(answer="", refusal_reason="no_context", source_ids=[], raw="")
        async def _request(system_prompt: str, context_limit: int) -> str:
            context_block = _build_context_block(contexts, context_limit)
            history_block = _format_history(history)
            user_prompt = (
                f"{history_block}Question: {query}\n\n"
                f"Context:\n{context_block}\n\n"
                "Instructions: Use only the context above to answer. "
                "Respond with JSON only."
            )
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
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
                raise LLMError(str(exc)) from exc

            choices = data.get("choices") or []
            if not choices:
                raise LLMError("Invalid OpenAI response")
            message = choices[0].get("message") or {}
            content = message.get("content")
            if not isinstance(content, str):
                raise LLMError("Invalid OpenAI response content")
            return content

        content = await _request(self.system_prompt, self.context_max_chars)
        try:
            parsed = _parse_json_response(content)
        except Exception as exc:
            if not _is_json_parse_error(exc):
                raise
            logger.warning(
                "llm_json_retry",
                extra={
                    "provider": "openai",
                    "model": self.model,
                },
            )
            retry_prompt = _strict_system_prompt(self.system_prompt)
            retry_context = max(2000, self.context_max_chars // 2)
            content = await _request(retry_prompt, retry_context)
            parsed = _parse_json_response(content)
        answer = parsed.get("answer")
        if not isinstance(answer, str):
            raise LLMError("Invalid LLM JSON response: missing answer")
        refusal_reason = parsed.get("refusal_reason")
        if refusal_reason is not None and not isinstance(refusal_reason, str):
            raise LLMError("Invalid LLM JSON response: refusal_reason")
        source_ids = parsed.get("source_ids") or []
        if not isinstance(source_ids, list) or any(
            not isinstance(item, str) for item in source_ids
        ):
            raise LLMError("Invalid LLM JSON response: source_ids")
        return LLMResult(
            answer=answer.strip(),
            refusal_reason=refusal_reason,
            source_ids=source_ids,
            raw=content.strip(),
        )


@dataclass(frozen=True)
class GeminiAnswerer:
    """LLM answerer backed by Gemini generative models."""
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    timeout: float
    context_max_chars: int
    system_prompt: str = _SYSTEM_PROMPT

    async def generate(
        self,
        query: str,
        contexts: list[ContextChunk],
        history: list[dict[str, str]] | None = None,
    ) -> LLMResult:
        """Generate a grounded answer using Gemini."""
        if not contexts:
            return LLMResult(answer="", refusal_reason="no_context", source_ids=[], raw="")
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise LLMError("google-generativeai is required for GeminiAnswerer") from exc

        async def _request(system_prompt: str, context_limit: int) -> str:
            context_block = _build_context_block(contexts, context_limit)
            history_block = _format_history(history)
            prompt = (
                f"{system_prompt}\n\n"
                f"{history_block}"
                f"Question: {query}\n\n"
                f"Context:\n{context_block}\n\n"
                "Instructions: Use only the context above to answer. "
                "Respond with JSON only."
            )

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
                return await asyncio.wait_for(asyncio.to_thread(_run), timeout=self.timeout)
            except Exception as exc:
                raise LLMError(str(exc)) from exc

        content = await _request(self.system_prompt, self.context_max_chars)
        try:
            parsed = _parse_json_response(content)
        except Exception as exc:
            if not _is_json_parse_error(exc):
                raise
            logger.warning(
                "llm_json_retry",
                extra={
                    "provider": "gemini",
                    "model": self.model,
                },
            )
            retry_prompt = _strict_system_prompt(self.system_prompt)
            retry_context = max(2000, self.context_max_chars // 2)
            content = await _request(retry_prompt, retry_context)
            parsed = _parse_json_response(content)
        answer = parsed.get("answer")
        if not isinstance(answer, str):
            raise LLMError("Invalid LLM JSON response: missing answer")
        refusal_reason = parsed.get("refusal_reason")
        if refusal_reason is not None and not isinstance(refusal_reason, str):
            raise LLMError("Invalid LLM JSON response: refusal_reason")
        source_ids = parsed.get("source_ids") or []
        if not isinstance(source_ids, list) or any(
            not isinstance(item, str) for item in source_ids
        ):
            raise LLMError("Invalid LLM JSON response: source_ids")
        return LLMResult(
            answer=answer.strip(),
            refusal_reason=refusal_reason,
            source_ids=source_ids,
            raw=content.strip(),
        )


def _build_context_block(contexts: list[ContextChunk], max_chars: int) -> str:
    """Build a context block annotated with source metadata."""
    chunks: list[str] = []
    total = 0
    for idx, chunk in enumerate(contexts, start=1):
        source_type = chunk.metadata.get("source_type", "unknown")
        source_name = chunk.metadata.get("source_name", "unknown")
        header = (
            "Source "
            f"{idx} (id={chunk.document_id}, score={chunk.score:.3f}, "
            f"type={source_type}, name={source_name}):\n"
        )
        content = chunk.content.strip()
        snippet = header + content
        if total + len(snippet) > max_chars:
            remaining = max_chars - total
            if remaining <= len(header):
                break
            snippet = header + content[: remaining - len(header)]
        chunks.append(snippet)
        total += len(snippet)
        if total >= max_chars:
            break
    return "\n\n".join(chunks)


def _format_history(history: list[dict[str, str]] | None) -> str:
    """Format chat history for inclusion in prompts."""
    if not history:
        return ""
    lines: list[str] = []
    for item in history:
        role = item.get("role", "user").strip().lower()
        content = item.get("content", "").strip()
        if not content:
            continue
        if role not in {"user", "assistant", "system"}:
            role = "user"
        label = role.capitalize()
        lines.append(f"{label}: {content}")
    if not lines:
        return ""
    return "Conversation so far:\n" + "\n".join(lines) + "\n\n"


def _parse_json_response(content: str) -> dict[str, object]:
    """Parse a JSON object from model output."""
    text = content.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    raise LLMError("LLM response is not valid JSON")


def _is_json_parse_error(exc: Exception) -> bool:
    """Return True when an exception indicates invalid JSON output."""
    return isinstance(exc, LLMError) and "valid JSON" in str(exc)


def _strict_system_prompt(base_prompt: str) -> str:
    """Return a stricter system prompt for JSON-only retries."""
    return (
        f"{base_prompt} "
        "Return a single JSON object and nothing else. "
        "Do not use markdown or code fences. "
        "If you are unsure, set refusal_reason to \"no_context\"."
    )


def _parse_gate_response(content: str) -> GateResult:
    """Parse a gate decision from model output."""
    parsed = _parse_json_response(content)
    sufficient = parsed.get("sufficient")
    if not isinstance(sufficient, bool):
        raise LLMError("Invalid gate response: sufficient")
    reason = parsed.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = "unspecified"
    return GateResult(sufficient=sufficient, reason=reason.strip(), raw=content.strip())


@dataclass(frozen=True)
class OllamaGate:
    """Gatekeeper using Ollama to assess context sufficiency."""
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout: float
    context_max_chars: int

    async def evaluate(self, query: str, contexts: list[ContextChunk]) -> GateResult:
        """Evaluate sufficiency using Ollama."""
        if not contexts:
            return GateResult(sufficient=False, reason="no_context", raw="")
        context_block = _build_context_block(contexts, self.context_max_chars)
        user_prompt = (
            f"Question: {query}\n\n"
            f"Context:\n{context_block}\n\n"
            "Return JSON only."
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _GATE_PROMPT},
                {"role": "user", "content": user_prompt},
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
            raise LLMError(str(exc)) from exc

        message = data.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise LLMError("Invalid gate response")
        return _parse_gate_response(content)


@dataclass(frozen=True)
class OpenAIGate:
    """Gatekeeper using OpenAI to assess context sufficiency."""
    api_key: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout: float
    context_max_chars: int

    async def evaluate(self, query: str, contexts: list[ContextChunk]) -> GateResult:
        """Evaluate sufficiency using OpenAI chat completions."""
        if not contexts:
            return GateResult(sufficient=False, reason="no_context", raw="")
        context_block = _build_context_block(contexts, self.context_max_chars)
        user_prompt = (
            f"Question: {query}\n\n"
            f"Context:\n{context_block}\n\n"
            "Return JSON only."
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _GATE_PROMPT},
                {"role": "user", "content": user_prompt},
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
            raise LLMError(str(exc)) from exc

        choices = data.get("choices") or []
        if not choices:
            raise LLMError("Invalid OpenAI gate response")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise LLMError("Invalid OpenAI gate content")
        return _parse_gate_response(content)


@dataclass(frozen=True)
class GeminiGate:
    """Gatekeeper using Gemini to assess context sufficiency."""
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    timeout: float
    context_max_chars: int

    async def evaluate(self, query: str, contexts: list[ContextChunk]) -> GateResult:
        """Evaluate sufficiency using Gemini."""
        if not contexts:
            return GateResult(sufficient=False, reason="no_context", raw="")
        context_block = _build_context_block(contexts, self.context_max_chars)
        prompt = (
            f"{_GATE_PROMPT}\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context_block}\n\n"
            "Return JSON only."
        )
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise LLMError("google-generativeai is required for Gemini gate") from exc

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
            raise LLMError(str(exc)) from exc
        return _parse_gate_response(content)


def build_llm_answerer(
    provider: str,
    *,
    api_key_openai: str | None,
    api_key_gemini: str | None,
    openai_base_url: str,
    openai_model: str | None,
    gemini_model: str | None,
    ollama_base_url: str,
    ollama_model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    context_max_chars: int,
    system_prompt: str | None = None,
) -> OllamaAnswerer | OpenAIAnswerer | GeminiAnswerer:
    """Factory for LLM answerers based on provider."""
    resolved_prompt = system_prompt or _SYSTEM_PROMPT
    normalized = provider.strip().lower()
    if normalized in {"openai"}:
        if not api_key_openai:
            raise LLMError("OPENAI_API_KEY is required for OpenAI provider")
        if not openai_model:
            raise LLMError("OPENAI_CHAT_MODEL is required for OpenAI provider")
        return OpenAIAnswerer(
            api_key=api_key_openai,
            base_url=openai_base_url.rstrip("/"),
            model=openai_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            context_max_chars=context_max_chars,
            system_prompt=resolved_prompt,
        )
    if normalized in {"gemini", "google"}:
        if not api_key_gemini:
            raise LLMError("GEMINI_API_KEY is required for Gemini provider")
        if not gemini_model:
            raise LLMError("GEMINI_CHAT_MODEL is required for Gemini provider")
        return GeminiAnswerer(
            api_key=api_key_gemini,
            model=gemini_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            context_max_chars=context_max_chars,
            system_prompt=resolved_prompt,
        )
    return OllamaAnswerer(
        base_url=ollama_base_url,
        model=ollama_model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        context_max_chars=context_max_chars,
        system_prompt=resolved_prompt,
    )


def build_llm_gate(
    provider: str,
    *,
    api_key_openai: str | None,
    api_key_gemini: str | None,
    openai_base_url: str,
    openai_model: str | None,
    gemini_model: str | None,
    ollama_base_url: str,
    ollama_model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    context_max_chars: int,
) -> OllamaGate | OpenAIGate | GeminiGate:
    """Factory for LLM gatekeepers based on provider."""
    normalized = provider.strip().lower()
    if normalized in {"openai"}:
        if not api_key_openai:
            raise LLMError("OPENAI_API_KEY is required for OpenAI gate")
        if not openai_model:
            raise LLMError("OPENAI_CHAT_MODEL is required for OpenAI gate")
        return OpenAIGate(
            api_key=api_key_openai,
            base_url=openai_base_url.rstrip("/"),
            model=openai_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            context_max_chars=context_max_chars,
        )
    if normalized in {"gemini", "google"}:
        if not api_key_gemini:
            raise LLMError("GEMINI_API_KEY is required for Gemini gate")
        if not gemini_model:
            raise LLMError("GEMINI_CHAT_MODEL is required for Gemini gate")
        return GeminiGate(
            api_key=api_key_gemini,
            model=gemini_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            context_max_chars=context_max_chars,
        )
    return OllamaGate(
        base_url=ollama_base_url,
        model=ollama_model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        context_max_chars=context_max_chars,
    )
