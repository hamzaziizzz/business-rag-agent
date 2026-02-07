from __future__ import annotations

"""Simple non-LLM answerer for extractive outputs."""

from dataclasses import dataclass

from src.rag.types import ContextChunk


@dataclass
class ExtractiveAnswerer:
    """Return a short extract from the highest scoring chunk."""
    max_chars: int = 480

    def generate(self, query: str, contexts: list[ContextChunk]) -> str:
        """Generate an extractive answer from context."""
        if not contexts:
            return ""
        best = max(contexts, key=lambda chunk: chunk.score)
        snippet = self._truncate(best.content.strip())
        return f"Based on the provided context: {snippet}"

    def _truncate(self, text: str) -> str:
        """Trim text to the max character budget without cutting words."""
        if len(text) <= self.max_chars:
            return text
        return text[: self.max_chars].rsplit(" ", 1)[0] + "..."

