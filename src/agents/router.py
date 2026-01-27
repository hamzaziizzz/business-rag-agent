from __future__ import annotations

from dataclasses import dataclass


SQL_HINTS = ("select ", "insert ", "update ", "delete ", "sql:")
SUMMARY_HINTS = ("summarize", "summary", "tl;dr", "tldr")


@dataclass(frozen=True)
class RouteDecision:
    tool: str
    reason: str


class AgentRouter:
    def route(self, query: str) -> RouteDecision:
        if not query.strip():
            return RouteDecision(tool="refuse", reason="empty_query")
        lowered = query.strip().lower()
        if any(lowered.startswith(hint) for hint in SQL_HINTS):
            return RouteDecision(tool="sql", reason="sql_hint")
        if any(hint in lowered for hint in SUMMARY_HINTS):
            return RouteDecision(tool="summarize", reason="summary_hint")
        return RouteDecision(tool="rag", reason="default_rag")
