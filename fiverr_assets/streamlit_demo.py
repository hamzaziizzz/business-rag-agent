from __future__ import annotations

"""Streamlit demo UI for the document-only RAG FastAPI backend."""

import json
from typing import Any

import httpx
import streamlit as st


DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_REQUEST_TIMEOUT = None


def _headers(api_key: str | None) -> dict[str, str]:
    """Build optional API key headers for backend requests."""
    if not api_key:
        return {}
    return {"X-API-Key": api_key.strip()}


def _merge_headers(api_key: str | None) -> dict[str, str]:
    return _headers(api_key)


def _post_json(
    api_url: str,
    path: str,
    payload: dict[str, Any],
    api_key: str | None,
    timeout: float | None,
):
    """POST JSON payloads to the API."""
    url = api_url.rstrip("/") + path
    with httpx.Client(timeout=timeout) as client:
        return client.post(url, json=payload, headers=_merge_headers(api_key))


def _post_files(
    api_url: str,
    path: str,
    files,
    api_key: str | None,
    timeout: float | None,
):
    """POST multipart file uploads to the API."""
    url = api_url.rstrip("/") + path
    with httpx.Client(timeout=timeout) as client:
        return client.post(url, files=files, headers=_merge_headers(api_key))


def _health_check(api_url: str, api_key: str | None) -> tuple[bool, str]:
    """Return backend health status and a human-readable message."""
    url = api_url.rstrip("/") + "/health"
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(url, headers=_merge_headers(api_key))
        if response.status_code == 200:
            return True, "API is reachable."
        return False, f"API responded with status {response.status_code}."
    except httpx.HTTPError as exc:
        return False, f"API connection failed: {exc}"


def _render_response(response: httpx.Response) -> None:
    """Render an API response with answer, citations, and sources."""
    st.subheader("Response")
    st.code(f"Status: {response.status_code}")
    try:
        payload = response.json()
    except ValueError:
        st.text(response.text)
        return
    st.json(payload)
    answer = payload.get("answer")
    if answer:
        st.markdown("**Answer**")
        st.write(answer)
    answerer = payload.get("answerer")
    if answerer:
        st.markdown("**Answerer**")
        st.write(answerer)
    answerer_reason = payload.get("answerer_reason")
    if answerer_reason:
        st.markdown("**Answerer Reason**")
        st.write(answerer_reason)
    citations = payload.get("citations") or []
    if citations:
        st.markdown("**Citations**")
        st.json(citations)
    structured = payload.get("structured")
    if structured:
        st.markdown("**Structured Output**")
        st.json(structured)
    sources = payload.get("sources") or []
    if sources:
        st.markdown("**Source Chunks**")
        for chunk in sources:
            st.markdown(f"- `{chunk.get('document_id')}` (score={chunk.get('score')})")


st.set_page_config(page_title="Grounded Business RAG Assistant", layout="wide")
st.title("Grounded Business RAG Assistant")
st.caption("Streamlit demo UI for the FastAPI RAG backend.")

with st.sidebar:
    st.header("Connection")
    api_url = st.text_input("API base URL", value=DEFAULT_API_URL)
    api_key = st.text_input("API key (optional)", type="password")
    request_timeout = st.number_input(
        "Request timeout (seconds, 0 = no timeout)",
        min_value=0,
        max_value=3600,
        value=0,
        step=5,
    )
    if st.button("Health Check"):
        ok, message = _health_check(api_url, api_key)
        if ok:
            st.success(message)
        else:
            st.error(message)

tab_chat, tab_query, tab_ingest, tab_upload = st.tabs(
    ["Chat", "Query", "Ingest Text", "Upload Files"]
)

with tab_chat:
    st.subheader("Chat")
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if st.button("Clear Chat"):
        st.session_state.chat_messages = []

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Ask a question grounded in your documents...")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        timeout = None if request_timeout == 0 else float(request_timeout)
        payload = {
            "messages": st.session_state.chat_messages,
            "top_k": 4,
        }
        try:
            response = _post_json(api_url, "/chat", payload, api_key, timeout)
            result = response.json()
            answer = result.get("answer", "")
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)
                if result.get("answerer"):
                    st.caption(f"Answerer: {result.get('answerer')}")
                if result.get("answerer_reason"):
                    st.caption(f"Reason: {result.get('answerer_reason')}")
                citations = result.get("citations") or []
                if citations:
                    st.markdown("**Citations**")
                    st.json(citations)
        except httpx.HTTPError as exc:
            st.error(f"API connection failed: {exc}")

with tab_query:
    st.subheader("Ask a Question")
    query = st.text_area("Query", placeholder="Ask a question about your data...")
    top_k = st.slider("Top K", min_value=1, max_value=20, value=4)
    min_score = st.slider("Min Score Threshold", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
    route = st.selectbox("Route", ["auto", "rag", "summarize"], index=0)
    if st.button("Run Query", type="primary"):
        try:
            payload: dict[str, Any] = {"query": query, "top_k": top_k, "min_score": min_score}
            if route != "auto":
                payload["route"] = route
            timeout = None if request_timeout == 0 else float(request_timeout)
            response = _post_json(api_url, "/query", payload, api_key, timeout)
            _render_response(response)
        except httpx.HTTPError as exc:
            st.error(f"API connection failed: {exc}")

with tab_ingest:
    st.subheader("Manual Text Ingest")
    doc_id = st.text_input("Doc ID", value="doc-1")
    source_name = st.text_input("Source name", value="manual")
    content = st.text_area("Document content")
    if st.button("Ingest Document"):
        try:
            payload = {
                "documents": [
                    {
                        "doc_id": doc_id,
                        "content": content,
                        "metadata": {"source_name": source_name, "source_type": "manual"},
                    }
                ]
            }
            timeout = None if request_timeout == 0 else float(request_timeout)
            response = _post_json(api_url, "/ingest", payload, api_key, timeout)
            _render_response(response)
        except httpx.HTTPError as exc:
            st.error(f"API connection failed: {exc}")

with tab_upload:
    st.subheader("Upload Files")
    uploaded_files = st.file_uploader(
        "Upload PDF/DOCX/CSV/TXT/etc.",
        accept_multiple_files=True,
    )
    if st.button("Upload & Ingest") and uploaded_files:
        try:
            files = [
                ("files", (file.name, file.getvalue(), file.type or "application/octet-stream"))
                for file in uploaded_files
            ]
            timeout = None if request_timeout == 0 else float(request_timeout)
            response = _post_files(api_url, "/ingest/files", files, api_key, timeout)
            _render_response(response)
        except httpx.HTTPError as exc:
            st.error(f"API connection failed: {exc}")

st.divider()
st.caption("Tip: Start the API with `uvicorn src.app.main:app --port 8000`.")
