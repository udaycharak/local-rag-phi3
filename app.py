"""
app.py

FastAPI wrapper around the RAG pipeline from step2_rag_pipeline.py.

This is the "API contract" layer: it defines WHAT can be asked for
(the endpoints) and WHAT SHAPE the request/response take, without
exposing any of the retrieval/generation internals to the caller.
Any client (Streamlit, curl, another script) only ever needs to know
this contract -- never the BM25/phi3 implementation behind it.

Run with:
    uvicorn app:app --reload

Then test with:
    curl http://localhost:8000/health

    curl -X POST http://localhost:8000/index/text \
      -H "Content-Type: application/json" \
      -d '{"text": "SAP BDC combines Datasphere, Databricks, and AI Core.", "source": "test"}'

    curl -X POST http://localhost:8000/query \
      -H "Content-Type: application/json" \
      -d '{"question": "What does SAP BDC combine?"}'

Or view interactive docs at: http://localhost:8000/docs
(FastAPI generates this automatically from the code below --
 this is your equivalent of an OData $metadata document.)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import ChatOllama

MODEL = "phi3:mini"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 200  # widened from 50 after testing showed answers were
                     # being split across chunk boundaries in Q&A-style
                     # documents; 200 keeps question+answer pairs together
TOP_K = 5  # matches the k=5 fix from testing

RAG_TEMPLATE = """You are a private assistant. Answer the question using ONLY the provided context. If the context does not contain the answer, say you do not know. Do not make things up.

Context:
{context}

Question: {question}

Answer:"""

app = FastAPI(title="Local RAG API", description="Vectorless RAG over phi3:mini via Ollama")

# --- Global in-memory state ------------------------------------------
# Mirrors the spec: no database, everything lives in process memory
# and is lost on restart. This is intentional given the RAM constraint.
_chunks: list[Document] = []
_retriever: BM25Retriever | None = None
_llm = ChatOllama(model=MODEL, temperature=0.0, num_ctx=2048)


# --- Request/response schemas ------------------------------------------
# This is your API contract, in the OData-metadata sense: it defines
# exactly what shape a caller must send and what shape they get back.

class IndexTextRequest(BaseModel):
    text: str
    source: str = "unnamed"  # lets you track which document a chunk came from


class IndexTextResponse(BaseModel):
    status: str
    chunks_created: int
    total_chunks: int


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


class ResetResponse(BaseModel):
    status: str
    freed_chunks: int


# --- Helper functions ------------------------------------------
def _rebuild_retriever():
    """BM25Retriever doesn't support incremental updates, so we
    rebuild it from the full chunk list every time new text is indexed.
    Fine for prototype scale; would need a different approach at
    much larger document counts."""
    global _retriever
    if _chunks:
        _retriever = BM25Retriever.from_documents(_chunks)
        _retriever.k = TOP_K
    else:
        _retriever = None


def _format_context(chunks: list[Document]) -> str:
    parts = []
    for doc in chunks:
        parts.append(f"[Source: {doc.metadata['source']}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


# --- Endpoints ------------------------------------------

@app.get("/health")
def health():
    """Quick check that the service is up and how many chunks are indexed."""
    return {"status": "ok", "chunks_indexed": len(_chunks), "model": MODEL}


@app.post("/index/text", response_model=IndexTextResponse)
def index_text(request: IndexTextRequest):
    """Chunk incoming text and add it to the in-memory index."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    text_chunks = splitter.split_text(request.text)

    if not text_chunks:
        raise HTTPException(status_code=400, detail="No content to index (empty text).")

    start_id = len(_chunks)
    for i, chunk in enumerate(text_chunks):
        _chunks.append(Document(
            page_content=chunk,
            metadata={"chunk_id": start_id + i, "source": request.source},
        ))

    _rebuild_retriever()

    return IndexTextResponse(
        status="success",
        chunks_created=len(text_chunks),
        total_chunks=len(_chunks),
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Retrieve relevant chunks and generate a grounded answer."""
    if _retriever is None:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Call /index/text first."
        )

    retrieved_chunks = _retriever.invoke(request.question)
    context = _format_context(retrieved_chunks)

    prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    messages = prompt.format_messages(context=context, question=request.question)

    response = _llm.invoke(messages)

    return QueryResponse(
        answer=response.content,
        sources=list(set(d.metadata["source"] for d in retrieved_chunks)),
    )


@app.post("/reset", response_model=ResetResponse)
def reset():
    """Flush the in-memory index."""
    global _chunks, _retriever
    freed = len(_chunks)
    _chunks = []
    _retriever = None
    return ResetResponse(status="index cleared", freed_chunks=freed)
