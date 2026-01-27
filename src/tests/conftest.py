from __future__ import annotations

import sys
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("RAG_DB_ALLOWED_TABLES", "")
os.environ.setdefault("RAG_DB_ALLOWED_COLUMNS", "")
os.environ.setdefault("RAG_ALLOW_ANONYMOUS", "true")
os.environ["RAG_ANSWERER"] = "extractive"
