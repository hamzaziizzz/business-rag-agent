# AGENTS.md

## Project Overview

This repository implements a **production-grade Custom AI Agent using Retrieval-Augmented Generation (RAG)** for querying business data (documents, databases, APIs) with high accuracy and minimal hallucination.

The system is designed for **enterprise use**, not experimentation or toy demos.

---

## Primary Objectives

AI agents contributing to this repository should optimize for:

1. **Accuracy over creativity**
2. **Deterministic, grounded responses**
3. **Security and data isolation**
4. **Clear separation of concerns**
5. **Production readiness**

---

## High-Level Architecture

The system follows a strict RAG pipeline:

1. User Query
2. Query understanding & routing
3. Embedding generation
4. Vector search (Milvus)
5. Context filtering & ranking
6. LLM reasoning using retrieved context only
7. Response with optional source references

At no point should the LLM answer without retrieved context.

---

## Core Components

### `src/app/`
- FastAPI entrypoint and HTTP routes
- API schemas and request/response validation

### `src/rag/`
- Embedding generation
- Retrieval logic
- RAG chain orchestration
- Prompt templates with guardrails

### `src/vectorstore/`
- Milvus client and collection management
- Metadata-based filtering

### `src/loaders/`
- Document loaders (PDF, Markdown, text)
- Chunking and preprocessing logic

### `src/agents/`
- Agent routing logic
- Tool selection (RAG, SQL, summarization, etc.)

---

## Rules for AI Agents (Important)

### ❗ Do NOT:
- Add direct LLM calls without retrieval
- Introduce hallucination-prone prompts
- Hardcode credentials or secrets
- Mix business logic into API routes
- Bypass vector retrieval for responses

### ✅ DO:
- Enforce context grounding in all answers
- Keep prompts minimal and structured
- Validate inputs and outputs
- Log errors and edge cases clearly
- Prefer explicit configuration over magic values

---

## Prompting Guidelines

- System prompts must clearly state:
  - Answer only from provided context
  - Say “I don’t know” if context is insufficient
- Avoid open-ended or speculative prompts
- Use structured outputs where possible

---

## Security & Privacy

- Client data must never be logged in raw form
- No training or fine-tuning on user data
- Assume multi-tenant usage
- Support on-prem and private deployments

---

## Testing Expectations

AI agents should:
- Add tests for retrieval correctness
- Validate that empty context → refusal
- Ensure API stability
- Avoid breaking backward compatibility

---

## Contribution Philosophy

This repository prioritizes:
- Reliability
- Maintainability
- Clarity

Performance optimizations are welcome **only after correctness is ensured**.

---

## Final Note to AI Agents

This is **not** a chatbot playground.

Treat this codebase as:
> Infrastructure software where wrong answers have real-world consequences.

Proceed accordingly.
