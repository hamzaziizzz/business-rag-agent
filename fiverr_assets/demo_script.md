# Demo Script: Grounded Business RAG Assistant

Goal: Show grounded answers, citations, and "I don't know" behavior in 3-4 minutes.

## 1) Intro (20s)
- "This is a production-ready RAG assistant that answers only from your data."
- "No hallucinations. Every response is grounded and cited."

## 2) Ingest Data (45s)
- Upload a policy PDF and a CSV export.
- Mention multi-source ingestion and metadata tracking.

## 3) Grounded Q&A (60s)
- Ask a direct question covered in the documents.
- Show answer and source citations (e.g., "Sources: [1] [2]").
- Open a source chunk to show the exact text.

## 4) Trick Question (40s)
- Ask a question not in the documents.
- Show refusal: "I don't know based on the provided context."

## 5) Structured Response (35s)
- Ask a question that yields bullets (e.g., "Summarize travel policy").
- Show structured output in the API response.

## 6) Wrap (15s)
- "This is API-first, deployable with Docker."
- "Ideal for HR, Legal, SOPs, and internal knowledge."
