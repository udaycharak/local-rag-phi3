"""
streamlit_app.py

A Streamlit CLIENT for the RAG API you built in app.py.

Architecturally, this file knows NOTHING about BM25, phi3, or
chunking -- it only knows the API contract (the Pydantic request/
response shapes from app.py). This is the client side of the
client-server split we diagrammed earlier.

Prerequisite: the FastAPI service must already be running:
    uvicorn app:app --reload

Run this with:
    streamlit run streamlit_app.py
"""

import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Local RAG (phi3:mini)", layout="centered")
st.title("Local RAG — phi3:mini via Ollama")
st.caption("Fully local: no data leaves your machine, no API costs.")


# --- Helper functions wrapping the API calls ------------------------
# Each function here maps 1:1 to an endpoint in app.py. If the API
# contract ever changes, this is the only place that needs updating.

def check_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return None


def index_text(text: str, source: str):
    r = requests.post(
        f"{API_BASE}/index/text",
        json={"text": text, "source": source},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def ask_question(question: str):
    r = requests.post(
        f"{API_BASE}/query",
        json={"question": question},
        timeout=60,  # phi3 generation can take a moment on M1
    )
    r.raise_for_status()
    return r.json()


def reset_index():
    r = requests.post(f"{API_BASE}/reset", timeout=10)
    r.raise_for_status()
    return r.json()


# --- Sidebar: connection status + index management ------------------
with st.sidebar:
    st.header("Service status")
    health = check_health()

    if health is None:
        st.error("API not reachable. Is `uvicorn app:app --reload` running?")
        st.stop()  # no point rendering the rest if the API is down
    else:
        st.success(f"Connected — {health['chunks_indexed']} chunk(s) indexed")
        st.caption(f"Model: {health['model']}")

    st.divider()
    st.subheader("Reset index")
    if st.button("Clear all indexed documents", type="secondary"):
        result = reset_index()
        st.success(f"Cleared {result['freed_chunks']} chunk(s)")
        st.rerun()


# --- Main area: two tabs, mirroring the two core endpoints ------------
tab_index, tab_query = st.tabs(["Add document", "Ask a question"])

with tab_index:
    st.subheader("Index new text")
    source_name = st.text_input("Source label", placeholder="e.g. hana_notes, prior_draft_3")
    text_input = st.text_area("Paste text to index", height=200)

    if st.button("Index this text", type="primary"):
        if not text_input.strip():
            st.warning("Paste some text first.")
        elif not source_name.strip():
            st.warning("Give it a source label so you can trace answers back to it.")
        else:
            with st.spinner("Chunking and indexing..."):
                result = index_text(text_input, source_name)
            st.success(
                f"Indexed {result['chunks_created']} chunk(s). "
                f"Total chunks now: {result['total_chunks']}"
            )

with tab_query:
    st.subheader("Ask a question")
    question = st.text_input("Your question", placeholder="What causes...?")

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Type a question first.")
        else:
            with st.spinner("Retrieving context and generating answer (phi3:mini)..."):
                try:
                    result = ask_question(question)
                except requests.exceptions.HTTPError as e:
                    st.error(f"Query failed: {e.response.json().get('detail', str(e))}")
                else:
                    st.markdown("### Answer")
                    st.write(result["answer"])
                    st.caption(f"Sources: {', '.join(result['sources'])}")
