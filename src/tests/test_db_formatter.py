from __future__ import annotations

import json

from src.rag.formatters import format_db_answer
from src.rag.types import ContextChunk


def test_format_db_answer_uses_known_fields() -> None:
    record = {
        "company_name": "JSW Steel Ltd",
        "headline": "Announcement under Regulation 30",
        "category": "Company Update",
        "filing_date": "2025-12-24",
    }
    chunk = ContextChunk(
        document_id="doc-1",
        content=json.dumps(record),
        metadata={"source_type": "db"},
        score=0.9,
    )

    answer = format_db_answer([chunk])

    assert answer is not None
    assert "JSW Steel Ltd" in answer
    assert "Company Update" in answer


def test_format_db_answer_combines_chunks() -> None:
    part1 = '{"company_name": "JSW", "headline": "Credit Rating",'
    part2 = '"category": "Company Update", "filing_date": "2025-12-24"}'
    chunks = [
        ContextChunk(
            document_id="row-1-1",
            content=part1,
            metadata={"source_type": "db", "chunk_index": 1, "chunk_count": 2},
            score=0.8,
        ),
        ContextChunk(
            document_id="row-1-2",
            content=part2,
            metadata={"source_type": "db", "chunk_index": 2, "chunk_count": 2},
            score=0.7,
        ),
    ]

    answer = format_db_answer(chunks)

    assert answer is not None
    assert "JSW" in answer
    assert "Credit Rating" in answer
