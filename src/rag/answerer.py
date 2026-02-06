from __future__ import annotations

"""Simple non-LLM answerers for extractive and summary outputs."""

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


@dataclass
class SummarizingAnswerer:
    """Return a stitched summary from top chunks."""
    max_chars: int = 800
    max_chunks: int = 3

    def generate(self, query: str, contexts: list[ContextChunk]) -> str:
        """Generate a summary answer from context."""
        if not contexts:
            return ""
        ranked = sorted(contexts, key=lambda chunk: chunk.score, reverse=True)
        pieces: list[str] = []
        for chunk in ranked[: self.max_chunks]:
            text = chunk.content.strip()
            if not text:
                continue
            pieces.append(self._truncate(text, limit=240))
        summary = " ".join(pieces).strip()
        if not summary:
            return ""
        summary = self._truncate(summary, limit=self.max_chars)
        return f"Summary based on the provided context: {summary}"

    def _truncate(self, text: str, limit: int) -> str:
        """Trim text to the provided character limit."""
        if len(text) <= limit:
            return text
        return text[:limit].rsplit(" ", 1)[0] + "..."
