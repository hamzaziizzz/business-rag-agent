from __future__ import annotations

"""Shared pytest fixtures and test environment defaults."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("RAG_ALLOW_ANONYMOUS", "true")
os.environ["RAG_ANSWERER"] = "extractive"
os.environ.pop("RAG_API_KEYS", None)
os.environ.pop("OPENAI_API_KEY", None)
os.environ.setdefault("RAG_VECTORSTORE", "memory")
os.environ.setdefault("RAG_HYBRID_SEARCH", "false")
os.environ.setdefault("RAG_DISABLE_TIKTOKEN", "true")
