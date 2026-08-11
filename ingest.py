"""
ingest.py

Handles reading real files off disk and extracting their text content,
before it gets chunked and fed into the BM25 retriever.

Supports: .txt, .md (trivial — just read as plain text)
          .pdf (needs pypdf to extract text properly)

Install for PDF support:
    pip install pypdf

Usage:
    from ingest import load_document, load_folder

    text = load_document("notes.txt")
    all_docs = load_folder("sample_docs/")   # returns list of (filename, text)
"""

import os
from pathlib import Path


def load_document(filepath: str) -> str:
    """
    Extract plain text from a single file. Dispatches based on
    file extension.
    """
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext in (".txt", ".md"):
        return _load_text_file(path)
    elif ext == ".pdf":
        return _load_pdf_file(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .txt, .md, .pdf")


def _load_text_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_pdf_file(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            "PDF support requires pypdf. Install with: pip install pypdf"
        )

    reader = PdfReader(str(path))
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n\n".join(text_parts)


def load_folder(folder_path: str) -> list[tuple[str, str]]:
    """
    Load every supported file in a folder.

    Returns a list of (filename, extracted_text) tuples so you keep
    track of which chunk came from which source document — useful
    later when you want to cite "this answer came from X.pdf".
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    supported_exts = {".txt", ".md", ".pdf"}
    results = []

    for file in sorted(folder.iterdir()):
        if file.suffix.lower() in supported_exts:
            try:
                text = load_document(str(file))
                results.append((file.name, text))
                print(f"Loaded: {file.name} ({len(text)} chars)")
            except Exception as e:
                print(f"⚠️  Failed to load {file.name}: {e}")

    if not results:
        print(f"No supported files found in {folder_path} (looking for .txt, .md, .pdf)")

    return results


if __name__ == "__main__":
    # quick manual test — point this at your sample_docs folder
    docs = load_folder("sample_docs")
    for filename, text in docs:
        print(f"\n--- {filename} ---")
        print(text[:300], "...")
