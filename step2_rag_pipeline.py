"""
step2_rag_pipeline.py

Adds the generation layer on top of step1's retrieval:
  retrieve relevant chunks -> assemble a strict context-only prompt
  -> send to phi3:mini via ChatOllama -> return the answer.

This is the actual RAG loop. Run step1_retrieval_test.py first if
you haven't confirmed retrieval quality on your documents yet.

Usage:
    python3 step2_rag_pipeline.py
"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import ChatOllama

from ingest import load_folder

DOCS_FOLDER = "sample_docs"
MODEL = "phi3:mini"

# Strict, anti-hallucination system prompt — from the functional spec.
# Small models like phi3 are MORE prone to confidently making things up
# than larger models, so this instruction matters more here than it
# would with GPT-4.
RAG_TEMPLATE = """You are a private assistant. Answer the question using ONLY the provided context. If the context does not contain the answer, say you do not know. Do not make things up.

Context:
{context}

Question: {question}

Answer:"""


def build_retriever(source_docs, chunk_size=500, chunk_overlap=50, k=3):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    documents = []
    chunk_id = 0
    for filename, text in source_docs:
        chunks = splitter.split_text(text)
        for chunk in chunks:
            documents.append(Document(
                page_content=chunk,
                metadata={"chunk_id": chunk_id, "source": filename},
            ))
            chunk_id += 1

    print(f"Split {len(source_docs)} document(s) into {len(documents)} chunks.")
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k
    return retriever


def format_context(chunks: list[Document]) -> str:
    """Join retrieved chunks into a single context block, tagging
    each with its source so you can trace an answer back to a file."""
    parts = []
    for doc in chunks:
        parts.append(f"[Source: {doc.metadata['source']}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def answer_question(retriever, llm, question: str, verbose: bool = True):
    """
    The core RAG call: retrieve -> build prompt -> generate.
    """
    retrieved_chunks = retriever.invoke(question)
    context = format_context(retrieved_chunks)

    prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    messages = prompt.format_messages(context=context, question=question)

    if verbose:
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"\nRetrieved {len(retrieved_chunks)} chunk(s) from: "
              f"{set(d.metadata['source'] for d in retrieved_chunks)}")
        print("-" * 60)

    response = llm.invoke(messages)

    if verbose:
        print(f"\nANSWER:\n{response.content}")

    return {
        "question": question,
        "answer": response.content,
        "sources": [d.metadata["source"] for d in retrieved_chunks],
        "context_used": context,
    }


if __name__ == "__main__":
    source_docs = load_folder(DOCS_FOLDER)

    if not source_docs:
        print(f"No documents found in '{DOCS_FOLDER}/'. Add files and rerun.")
    else:
        retriever = build_retriever(source_docs)
        llm = ChatOllama(model=MODEL, temperature=0.0, num_ctx=2048)

        # Test with a specific, keyword-rich question — the kind step1
        # showed BM25 handles well. Edit this to match your real docs.
        answer_question(
            retriever,
            llm,
            "What causes SAP HANA memory reclamation issues on large memory systems?"
        )
