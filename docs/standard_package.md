# Standard Package Checklist (Business-Knowledge-ChatBot)

This checklist documents the **Standard package** scope for the
`Business-Knowledge-ChatBot` branch.

## Requirements

✅ Multiple documents ingestion  
✅ Query rewriting (`RAG_QUERY_REWRITER`)  
✅ Hallucination control (guardrails + LLM gate + strict refusal)  
✅ Structured JSON responses  
✅ Simple chat UI + `/chat` endpoint  

## Recommended Defaults

* `RAG_LLM_FALLBACK=refuse`
* `RAG_CHAT_HISTORY_TURNS=6`
* `RAG_QUERY_REWRITER=on`

## Release Checklist

1. Run tests: `pipenv run pytest`
2. Reset Milvus collection and re-ingest demo docs
3. Verify `/query` and `/chat` with refusal behavior
4. Record the demo video
