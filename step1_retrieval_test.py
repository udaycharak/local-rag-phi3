"""
step1_retrieval_test.py

Standalone test of the vectorless retrieval piece: chunk a document,
build an in-memory BM25 retriever, and confirm it returns sensible
results for a query — BEFORE wiring in phi3 or FastAPI.

Why test this in isolation first: if retrieval quality is bad, no
amount of prompt engineering on the LLM side will fix a RAG pipeline.
Isolating this lets us debug retrieval and generation separately.

Install first (if not already):
    pip install langchain-community rank_bm25
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

from ingest import load_folder


# --- Real document ingestion ------------------------------------------
# Put your own files (.txt, .md, .pdf) in the sample_docs/ folder.
# load_folder() returns [(filename, text), ...] for everything it finds.
DOCS_FOLDER = "sample_docs"


def build_retriever(source_docs: list[tuple[str, str]], chunk_size: int = 500, chunk_overlap: int = 50, k: int = 3):
    """
    Chunk real documents and build an in-memory BM25 retriever.

    source_docs: list of (filename, text) tuples, e.g. from ingest.load_folder()
    chunk_size/chunk_overlap match the spec's sizing (tight, to fit
    phi3:mini's 2048-token context window).
    k = how many chunks to return per query (3, since our test showed
    2 missed a relevant chunk ranked just outside top-2 on a small corpus).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    documents = []
    chunk_id = 0
    for filename, text in source_docs:
        chunks = splitter.split_text(text)
        for chunk in chunks:
            # metadata keeps track of WHICH file each chunk came from —
            # useful once you want to cite sources in the final answer
            documents.append(Document(
                page_content=chunk,
                metadata={"chunk_id": chunk_id, "source": filename},
            ))
            chunk_id += 1

    print(f"Split {len(source_docs)} document(s) into {len(documents)} chunks.\n")
    for doc in documents:
        print(f"--- Chunk {doc.metadata['chunk_id']} (source: {doc.metadata['source']}) ---")
        print(doc.page_content.strip())
        print()

    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k
    return retriever


def test_query(retriever, question: str):
    print(f"\n{'='*60}")
    print(f"QUERY: {question}")
    print("-" * 60)
    results = retriever.invoke(question)
    for i, doc in enumerate(results):
        print(f"[Retrieved #{i+1}, chunk_id={doc.metadata['chunk_id']}, source={doc.metadata['source']}]")
        print(doc.page_content.strip())
        print()
    return results


if __name__ == "__main__":
    source_docs = load_folder(DOCS_FOLDER)

    if not source_docs:
        print(f"\nNo documents found. Add .txt, .md, or .pdf files to the '{DOCS_FOLDER}/' folder and rerun.")
    else:
        retriever = build_retriever(source_docs)

        # Edit these to match whatever's actually in your documents
        test_query(retriever, "What is this document about?")
