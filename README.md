# Local RAG on phi3:mini

A fully local, vectorless RAG (Retrieval-Augmented Generation) system built for an Apple M1 MacBook Air (8GB RAM). No external API calls, no cost, no data leaves the machine.

Ask questions against your own documents, answered by a small local LLM — grounded strictly in retrieved context, with no vector database and no cloud dependency.

## Why this exists

Most RAG tutorials assume a GPU, a paid API, or at least 16GB+ of RAM. This project was built specifically to work within a real, common hardware constraint: an 8GB Apple Silicon laptop. That constraint drove every architectural decision below — it's the reason there's no vector database, no dense embeddings, and a small quantized model instead of a frontier one.

Along the way, the project surfaced a genuine retrieval bug — some correct answers were being missed entirely due to how source documents were chunked — which was diagnosed, fixed, and verified end to end. See [`FINDINGS.md`](./FINDINGS.md) for the full write-up.

## Architecture

```
Streamlit (client, :8501)
        │  HTTP
        ▼
FastAPI service (:8000)
   ├── BM25 retriever (in-memory, keyword search)
   └── Prompt assembly + LLM call
        │
        ▼
Ollama (background service)
   └── phi3:mini (4-bit quantized)
```

- **Streamlit** — the UI. Knows nothing about retrieval or the model; only calls the API.
- **FastAPI** — owns the actual RAG logic: chunking, retrieval, prompt construction, generation. Exposes a clean HTTP contract (`/index/text`, `/query`, `/reset`, `/health`).
- **Ollama** — runs `phi3:mini` locally.

This client-server split means the UI and the RAG engine can change independently — the same pattern used in typical enterprise integration architectures, where a frontend consumes a backend service without knowing its internals.

No vector database, no dense embeddings — retrieval is pure keyword-based (BM25) over an in-memory chunk list, a deliberate trade-off to fit the RAM budget.

## Stack

- [Ollama](https://ollama.com) — local model runtime (`phi3:mini`)
- [LangChain](https://python.langchain.com) — `BM25Retriever`, `RecursiveCharacterTextSplitter`
- [FastAPI](https://fastapi.tiangolo.com) — API service
- [Streamlit](https://streamlit.io) — client UI

## Setup

```bash
# 1. Install Ollama and pull the model
brew install ollama   # or download from ollama.com
ollama pull phi3:mini

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start the API service (in its own terminal)
uvicorn app:app --reload

# 4. Start the Streamlit client (in a separate terminal)
streamlit run streamlit_app.py
```

Streamlit will open at `http://localhost:8501`. Add a document via the "Add document" tab, then ask questions in "Ask a question."

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service status and current chunk count |
| `/index/text` | POST | Chunk and index raw text (`{"text": "...", "source": "..."}`) |
| `/query` | POST | Retrieve + generate an answer (`{"question": "..."}`) |
| `/reset` | POST | Clear the in-memory index |

Interactive docs available at `http://localhost:8000/docs` once the API is running.

## Project files

- `app.py` — FastAPI service (the actual RAG logic)
- `streamlit_app.py` — Streamlit client
- `ingest.py` — document loading (txt, md, pdf)
- `step1_retrieval_test.py`, `step2_rag_pipeline.py` — standalone scripts from the incremental build process, kept as a record of how this was developed and tested
- `FINDINGS.md` — the debugging narrative: a real chunk-boundary retrieval bug, how it was diagnosed, and how it was fixed with before/after evidence

## Known limitations

- **BM25 vs semantic retrieval**: keyword matching works well for specific queries but weaker on vague ones, and can miss content split across chunk boundaries — see `FINDINGS.md` for a documented case and fix.
- **In-memory only**: the index resets when the API service restarts. No persistent storage.
- **Cold-start latency**: Ollama unloads idle models after a few minutes; the first query after idle time pays a reload cost.

## License

MIT
