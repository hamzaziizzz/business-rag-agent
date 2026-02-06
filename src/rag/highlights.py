from __future__ import annotations

"""Highlight extraction for query terms in context."""

import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9]{3,}")


def build_highlights(
    content: str,
    query: str,
    max_snippets: int = 3,
    window: int = 80,
) -> list[str]:
    """Extract highlighted snippets from content."""
    cleaned = content.strip()
    if not cleaned or not query.strip():
        return []
    tokens = _ordered_unique_tokens(query)
    if not tokens:
        return []

    highlights: list[str] = []
    lower = cleaned.lower()
    pattern = re.compile("|".join(re.escape(token) for token in tokens), re.IGNORECASE)

    for token in tokens:
        idx = lower.find(token)
        if idx == -1:
            continue
        start = max(0, idx - window)
        end = min(len(cleaned), idx + len(token) + window)
        snippet = cleaned[start:end].strip()
        if not snippet:
            continue
        snippet = pattern.sub(lambda match: f"[[{match.group(0)}]]", snippet)
        highlights.append(snippet)
        if len(highlights) >= max_snippets:
            break
    return highlights


def _ordered_unique_tokens(query: str) -> list[str]:
    """Return unique query tokens in order of appearance."""
    seen: set[str] = set()
    tokens: list[str] = []
    for token in _TOKEN_RE.findall(query.lower()):
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens
