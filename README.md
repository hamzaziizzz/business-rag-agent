# Custom AI Agent / RAG System for Business Data

**Production-Grade Retrieval-Augmented AI for Enterprises**

---

## 1. Project Overview

Modern businesses store critical information across **documents**, but accessing this data remains slow, fragmented, and error-prone.

This project delivers a **Custom AI Agent powered by Retrieval-Augmented Generation (RAG)** that allows organizations to query **their own data** securely and accurately using **natural language** — without hallucinations or data leakage.

> This is **not** a generic chatbot.
> It is a **business-aware AI system** designed for **accuracy, security, and scalability**.

---

## 2. Core Capabilities (Basic Package)

* Document ingestion (PDF/CSV/Markdown/Text)
* Grounded Q&A (answers only from retrieved context)
* Source citations

This branch is **document-only**. All DB/API/object/SQL ingestion paths are removed to keep the system focused and reliable for document RAG.

---

## 3. System Architecture (High-Level)

```
User Query
   ↓
Query Understanding
   ↓
Embedding Generation
   ↓
Vector Search (Milvus)
   ↓
Relevant Context Retrieval
   ↓
LLM Reasoning (Grounded)
   ↓
Final Answer + Sources
```

---

## 4. Technology Stack (Basic Package)

* FastAPI backend
* Milvus vector database
* Embeddings via OpenAI or Gemini (pluggable)

---

## 5. Quick Start

1. Copy `.env.basic` → `.env` and fill in values.
2. Start Milvus services:
   ```bash
   docker compose up -d milvus etcd minio
   ```
3. Run the API:
   ```bash
   /home/hamza/.local/share/virtualenvs/business-rag-agent-C7T_qDqk/bin/python3 \
     -m uvicorn src.app.main:app --host 0.0.0.0 --port 8010
   ```

---

## 6. Milvus Reset (Document-Only)

If you change schema (e.g., enable HNSW + BM25 hybrid), drop and recreate the collection and re-ingest documents.

```
/home/hamza/.local/share/virtualenvs/business-rag-agent-C7T_qDqk/bin/python3 \
  tools/reset_milvus_collection.py --collection business_documents
```
