# Basic Package (Document‑Only RAG)

This package delivers a **production‑ready, document‑only RAG system** with citations and a clean demo UI.

## What’s Included
- Document ingestion: `/ingest` and `/ingest/files`
- Grounded Q&A: `/query`
- Grounded chat endpoint: `/chat` (multi-turn with last-N history)
- Source citations + structured JSON output
- Streamlit demo UI
- Milvus HNSW + BM25 hybrid search

## Quick Start
1. **Set env vars**
   - Copy `.env.basic` → `.env` and fill in values.
2. **Start Milvus (docker compose)**
   ```bash
   docker compose up -d milvus etcd minio
   ```
3. **Reset collection (required after schema changes)**
   ```bash
   /home/hamza/.local/share/virtualenvs/business-rag-agent-C7T_qDqk/bin/python3 \
     tools/reset_milvus_collection.py --collection business_documents
   ```
4. **Run API**
   ```bash
   /home/hamza/.local/share/virtualenvs/business-rag-agent-C7T_qDqk/bin/python3 \
     -m uvicorn src.app.main:app --host 0.0.0.0 --port 8010
   ```
5. **Run Streamlit UI**
   ```bash
   /home/hamza/.local/share/virtualenvs/business-rag-agent-C7T_qDqk/bin/python3 \
     -m streamlit run fiverr_assets/streamlit_demo.py
   ```

## Demo Checklist
- Upload PDF/CSV.
- Ask a question answered in the document.
- Verify citations appear.
- Ask a question **not** in the docs → refusal.

## Notes
- Document‑only branch: no DB/API/Object/SQL ingestion.
- Hybrid search improves accuracy; disable with `RAG_HYBRID_SEARCH=false` if needed.
