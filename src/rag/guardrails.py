from __future__ import annotations

from dataclasses import dataclass

from src.rag.types import ContextChunk


DEFAULT_REFUSAL = "I don't know based on the provided context."


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    reason: str


def require_context(contexts: list[ContextChunk]) -> GuardrailResult:
    if not contexts:
        return GuardrailResult(allowed=False, reason="no_context")
    if all(not chunk.content.strip() for chunk in contexts):
        return GuardrailResult(allowed=False, reason="empty_context")
    return GuardrailResult(allowed=True, reason="ok")
