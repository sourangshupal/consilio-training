# Day 3 — RAG & Agents (Streamlit)

Interactive companion to `day3-notebooks/`, covering RAG techniques (vanilla, hybrid +
rerank, evaluation, self-correction, adaptive routing) and agent patterns (LangGraph
ReAct, multi-agent supervisor with human-in-the-loop) as a single multi-page Streamlit
app. Multimodal/ColPali retrieval is intentionally not included (GPU-only, notebook-only).

## Setup

```bash
cd day3-rag-agent-app
uv venv --python 3.11
source .venv/bin/activate
uv sync
```

## Run

```bash
uv run streamlit run app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`).

## Getting started

1. On the **Home** page, click **Load sample corpus** (4 bundled legal documents —
   service agreement, NDA, employment agreement, commercial lease) or upload your own
   PDF/DOCX/Markdown files and click **Process uploaded documents**.
2. Enter an API key for one provider in the sidebar (Groq is free — get one at
   console.groq.com). Retrieval-only pages (Vanilla RAG's retrieval step, BM25 & ColBERT)
   work without a key; every page that generates or judges text needs one.
3. Explore pages in the sidebar — each documents which day3 notebook it's based on.

## Notes

- Retrieval is entirely local (Chroma in-memory per session, sentence-transformers
  embeddings, BM25, a from-scratch ColBERT MaxSim demo) — no API key needed for
  retrieval-only exploration.
- Generation, evaluation (RAGAS), query transforms, and both agent pages route through
  whichever provider is selected in the sidebar (OpenAI/Gemini/Groq) via
  `src/llm_router.py` — pick any one, no need for multiple keys.
- RAGAS evaluation always uses the local embedding model for its embeddings step, so it
  never requires an OpenAI key specifically — only a judge-LLM key for whichever
  provider you picked.
- The document index is ephemeral and in-memory per browser session — nothing is
  persisted to disk, so there's no stale `chroma_data/` to clean up between runs.
