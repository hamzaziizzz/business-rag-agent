from __future__ import annotations

from dataclasses import dataclass
import json
import re

import httpx

from src.rag.types import ContextChunk


class LLMError(RuntimeError):
    pass


_SYSTEM_PROMPT = (
    "You are a production RAG assistant. "
    "Answer only from the provided context. "
    "If the context is insufficient, say: "
    "\"I don't know based on the provided context.\" "
    "Do not use external knowledge. Keep the answer concise. "
    "Return JSON only with keys: "
    "\"answer\" (string), \"refusal_reason\" (string or null), "
    "\"source_ids\" (array of strings)."
)


@dataclass(frozen=True)
class LLMResult:
    answer: str
    refusal_reason: str | None
    source_ids: list[str]
    raw: str


@dataclass(frozen=True)
class OllamaAnswerer:
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout: float
    context_max_chars: int

    async def generate(self, query: str, contexts: list[ContextChunk]) -> LLMResult:
        if not contexts:
            return LLMResult(answer="", refusal_reason="no_context", source_ids=[], raw="")
        context_block = _build_context_block(contexts, self.context_max_chars)
        user_prompt = (
            f"Question: {query}\n\n"
            f"Context:\n{context_block}\n\n"
            "Instructions: Use only the context above to answer. "
            "Respond with JSON only."
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
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


def _parse_json_response(content: str) -> dict[str, object]:
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
