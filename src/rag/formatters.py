from __future__ import annotations

import json
from typing import Any

from src.rag.types import ContextChunk


def format_db_answer(
    contexts: list[ContextChunk],
    max_items: int = 3,
    max_field_chars: int = 200,
) -> str | None:
    records = extract_db_records(contexts, max_items=max_items)
    if not records:
        return None

    lines: list[str] = []
    for record in records[:max_items]:
        line = _format_record(record, max_field_chars)
        if line:
            lines.append(line)
    if not lines:
        return None
    if len(lines) == 1:
        return f"Announcement: {lines[0]}"
    return "Announcements:\n" + "\n".join(f"- {line}" for line in lines)


def extract_db_records(
    contexts: list[ContextChunk],
    max_items: int = 10,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not contexts:
        return records
    for combined in _combine_db_chunks(contexts):
        parsed = _extract_json(combined)
        if parsed is None:
            continue
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    records.append(item)
        elif isinstance(parsed, dict):
            records.append(parsed)
        if len(records) >= max_items:
            break
    return records[:max_items]


def _format_record(record: dict[str, Any], max_field_chars: int) -> str:
    headline = _first_value(record, ["headline", "title", "NEWSSUB", "HEADLINE"])
    company = _first_value(record, ["company_name", "SLONGNAME", "COMPANY"])
    category = _first_value(record, ["category", "CATEGORYNAME"])
    date = _first_value(record, ["filing_date", "News_submission_dt", "NEWS_DT", "created_at"])

    if isinstance(record.get("raw_json"), dict):
        raw = record["raw_json"]
        if not headline:
            headline = _first_value(raw, ["HEADLINE", "NEWSSUB"])
        if not company:
            company = _first_value(raw, ["SLONGNAME", "COMPANY"])
        if not date:
            date = _first_value(raw, ["NEWS_DT", "DissemDT"])

    parts: list[str] = []
    if headline:
        parts.append(_truncate(str(headline), max_field_chars))
    if company:
        parts.append(f"Company: {_truncate(str(company), max_field_chars)}")
    if category:
        parts.append(f"Category: {_truncate(str(category), max_field_chars)}")
    if date:
        parts.append(f"Date: {_truncate(str(date), max_field_chars)}")
    return " | ".join(parts)


def _first_value(record: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return str(value)
    return None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def _extract_json(text: str) -> Any | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start_obj = cleaned.find("{")
    end_obj = cleaned.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        try:
            return json.loads(cleaned[start_obj : end_obj + 1])
        except json.JSONDecodeError:
            return None

    start_list = cleaned.find("[")
    end_list = cleaned.rfind("]")
    if start_list != -1 and end_list != -1 and end_list > start_list:
        try:
            return json.loads(cleaned[start_list : end_list + 1])
        except json.JSONDecodeError:
            return None
    return None


def _combine_db_chunks(contexts: list[ContextChunk]) -> list[str]:
    grouped: dict[str, list[ContextChunk]] = {}
    fallback: list[str] = []

    for chunk in contexts:
        if str(chunk.metadata.get("source_type", "")).lower() != "db":
            continue
        chunk_index = chunk.metadata.get("chunk_index")
        chunk_count = chunk.metadata.get("chunk_count")
        if chunk_index and chunk_count:
            base_id = _base_doc_id(chunk.document_id)
            grouped.setdefault(base_id, []).append(chunk)
        else:
            fallback.append(chunk.content)

    combined: list[str] = []
    for chunks in grouped.values():
        chunks.sort(key=lambda item: int(item.metadata.get("chunk_index", 0)))
        combined.append("".join(chunk.content for chunk in chunks))
    combined.extend(fallback)
    return combined


def _base_doc_id(doc_id: str) -> str:
    if "-" not in doc_id:
        return doc_id
    return doc_id.rsplit("-", 1)[0]
