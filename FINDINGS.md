# Local RAG on phi3:mini — Findings & Architecture Notes

## Summary

A fully local, vectorless RAG (Retrieval-Augmented Generation) system built for an Apple M1 MacBook Air (8GB RAM). No external API calls, no cost, no data leaves the machine. Built incrementally: standalone retrieval test → standalone generation pipeline → FastAPI service → Streamlit client.

**Stack:** Ollama (phi3:mini, 4-bit quantized) · LangChain (BM25Retriever, RecursiveCharacterTextSplitter) · FastAPI · Streamlit

**Test corpus:** a technical PDF document structured as a numbered Q&A list (question headings followed by explanatory answers, with some answers elaborated further in tables later in the document). The specific source document is not included in this repository — see "A note on test data" below.

---

## Architecture

Client-server separation, chosen deliberately over a single script:

- **Streamlit** (client) — UI only. Knows nothing about BM25, chunking, or phi3. Talks exclusively to the FastAPI contract via HTTP.
- **FastAPI** (service, `localhost:8000`) — owns the actual RAG logic: chunking, in-memory BM25 index, prompt assembly, and the call to the LLM. Exposes `/index/text`, `/query`, `/reset`, `/health`.
- **Ollama** (background service) — runs `phi3:mini` locally, called by FastAPI via `langchain-ollama`.

This mirrors a standard enterprise integration pattern (e.g. a frontend app → OData service → backend system): the client only knows the interface contract, never the implementation, so either side can change independently.

No vector database, no dense embeddings — retrieval is pure keyword-based (BM25) over an in-memory chunk list. This was a deliberate trade-off to stay within the 8GB RAM budget: embedding models and vector stores add meaningful memory overhead that isn't available on this hardware.

---

## Key finding: chunk-boundary retrieval failures

### Symptom
Specific, keyword-rich questions against the test PDF sometimes returned a correct, well-grounded answer — and sometimes returned "I do not know," even though the actual answer clearly existed in the source document.

### Diagnosis process
1. Isolated retrieval from generation — inspected the raw chunks BM25 returned for a failing query, bypassing the LLM entirely.
2. Found that the retrieved chunks contained the *question* (as a heading/cross-reference) but not its *answer* — the answer text began at or just after a chunk boundary.
3. Searched the full chunk list directly (not just the top-k) and confirmed the real answer content existed in the document, in a chunk adjacent to — but not overlapping enough with — the chunks BM25 was retrieving.

### Root cause
The source document is structured as a numbered Q&A list (e.g. a heading like "Which general checks can be performed...") where some answers are short and some are long. With `chunk_size=500` and `chunk_overlap=50`, a chunk boundary would frequently fall between a question heading and the start of its answer. BM25 scores on keyword overlap, so heading-only chunks (which repeat the query's exact words) sometimes outranked the actual answer chunk in the top-k results — meaning **more retrieved chunks (k) did not reliably fix this**, since the answer chunk could sit outside the top-k regardless of how large k was.

### Fix
Increased `chunk_overlap` from 50 to 200 (with `chunk_size` held at 500). This meant adjacent chunks shared far more content, making it much less likely that a question and its answer would be fully separated across a boundary.

### Evidence (before / after, same query, paraphrased — see note below)

**Before (`chunk_overlap=50`):** the model reported that the provided context did not contain the specific checks being asked about, and that it did not know.

**After (`chunk_overlap=200`):** the model returned the correct, specific answer — a checklist of concrete items pulled directly from the now-complete retrieved chunk, with the source citation the document itself used.

Verified through the full stack — API called directly, then re-verified through the Streamlit UI — not just in an isolated test script.

*Note: exact quoted model output and specific document citations have been omitted here since they were generated from proprietary source material. The finding and fix are fully reproducible against any Q&A-structured technical document — see "Reproducing this test" below.*

### Note on the fix's limits
Increasing overlap is a mitigation, not a structural solution. For documents where an answer is separated from its heading by a large distance (e.g. elaborated in a table many paragraphs later), a fixed-size/fixed-overlap chunking strategy can still fail. The more robust fix — not implemented here, but worth naming — is **semantic or structure-aware chunking**: splitting by document heading/section rather than raw character count, so a full Q&A unit stays together as one chunk regardless of its length.

---

## Key finding: hallucination resistance under pressure

Tested whether the system's grounding instruction ("answer ONLY from context, say you don't know otherwise") holds when retrieval returns genuinely irrelevant content — the realistic failure mode, as opposed to a clean "no context at all" test.

- **Fully off-topic query** (e.g. a general-knowledge question unrelated to the indexed document) against a single-topic index: BM25 was forced to return several irrelevant chunks (it cannot return zero results). phi3 correctly declined to answer, despite almost certainly knowing the answer from its own training data.
- **On-topic-but-incomplete context**: given a chunk that only *referenced* an answer elsewhere in the document, without containing the answer itself, phi3 correctly reported it did not know — rather than guessing or fabricating specifics.

This is a stronger result than a simple "ask it something unrelated" test, because both cases involved topically-relevant-looking context that still didn't actually contain the answer — exactly the condition where a weaker instruction-following model might be expected to fabricate.

---

## Other observations

- **PDF extraction noise**: `pypdf` text extraction occasionally introduces artifacts — stray mid-word line breaks, embedded hash-like strings — likely from unusual internal PDF formatting. Not yet a blocker, but worth cleaning up if it starts appearing in generated answers.
- **Cold-start latency**: Ollama unloads idle models from memory after a few minutes. The first query after idle time pays a model-reload cost (several seconds); subsequent queries are fast. This is a real, inherent trade-off of local-first inference vs. an always-warm cloud API.
- **In-memory-only state**: the FastAPI service holds its index purely in process memory — restarting the server (e.g. after a code change with `--reload`) clears it. Acceptable for a prototype; would need persistent storage for a real deployment.

---

## A note on test data

The document used to develop and test this system was a technical support-style PDF obtained through a licensed enterprise support portal, and is not included in this repository out of respect for that platform's terms of use. The architecture, code, and findings above are fully general — they were derived from and apply to *any* document with a numbered Q&A / heading-then-answer structure (FAQs, knowledge base exports, structured technical documentation), not anything specific to the original source.

### Reproducing this test
To reproduce the chunk-boundary failure and fix independently with public data:
1. Use any long-form FAQ or knowledge base document (e.g. a public open-source project's FAQ page, a public Wikipedia "List of..." style article with short sub-entries).
2. Index it with `chunk_overlap=50` and ask a question whose answer sits just after a natural section heading.
3. Compare retrieval and generated output against the same run with `chunk_overlap=200`.

---

## What this demonstrates

- Deliberate architecture trade-offs for a genuine hardware constraint (8GB RAM), not just following a tutorial.
- A real bug found through systematic isolation (retrieval tested separately from generation) rather than guessing at the LLM.
- A hypothesis-driven fix (chunk overlap, not just "increase k") tested in isolation before being applied to production code.
- End-to-end verification across a real client-server stack, not just a script.
