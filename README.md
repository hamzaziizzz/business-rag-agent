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
* Multi-turn grounded chat (strict RAG)
* Source citations and structured JSON output
* Streamlit demo UI
* Milvus HNSW + BM25 hybrid search

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
* Milvus vector database (HNSW + BM25 hybrid)
* LLM providers: Ollama / OpenAI / Gemini (pluggable)
* Streamlit demo UI

---

## 5. Quick Start

Follow the Basic Package guide:

* `fiverr_assets/basic_package.md`
* `docs/standard_package.md` (Standard package checklist)

---

## 6. Milvus Reset (Document-Only)

If you change schema (e.g., enable HNSW + BM25 hybrid), drop and recreate the collection and re-ingest documents.

```
/home/hamza/.local/share/virtualenvs/business-rag-agent-C7T_qDqk/bin/python3 \
  tools/reset_milvus_collection.py --collection business_documents
```
