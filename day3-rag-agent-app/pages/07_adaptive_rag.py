"""Adaptive RAG — query classification + strategy routing."""

import sys
sys.path.insert(0, ".")

import streamlit as st
from src.retriever import dense_retrieve, BM25Retriever, hybrid_rrf
from src.reranker import rerank
from src.generator import generate_answer
from src.llm_router import ask, normalize_label
from src.query_transforms import SYSTEM_DECOMPOSE, generate_variants, multi_retrieve
from src.rag_bank import ADAPTIVE_RAG_QUERIES
from src.state import active_api_key

st.title("🧠 Adaptive RAG")
st.caption(":material/menu_book: Notebook: `02_hybrid_retrieval_reranking.ipynb` (extended)")
st.badge("Needs API key", icon=":material/wifi:", color="gray")
st.caption("Classify the query, then route to the best retrieval strategy automatically.")

if not st.session_state.get("is_processed"):
    st.warning("Process documents on the home page first.", icon=":material/warning:")
    st.stop()

provider = st.session_state["provider"]
api_key = active_api_key()
embed_model = st.session_state["embedding_model_name"]
chunks = st.session_state["chunks"]

st.markdown("""
**Adaptive RAG** adds a routing layer that classifies the query type and selects the optimal strategy:

| Query type | Strategy | Why |
|---|---|---|
| Factual lookup | BM25 | Fast keyword match, exact clause lookups |
| Comparison | Hybrid (dense + BM25) | Needs breadth to cover both sides |
| Summary | Dense retrieval | Semantic coverage of a topic |
| Complex reasoning | Multi-query + rerank | Needs the most precise chunks |
""")

st.caption("One example query per classification type:")
type_choice = st.selectbox("Example query", list(ADAPTIVE_RAG_QUERIES.keys()))
custom_query = st.text_input("...or type your own question", placeholder="e.g., Compare the liability clauses in sections 3 and 7")
query = custom_query.strip() or ADAPTIVE_RAG_QUERIES[type_choice]

CLASSIFIER_PROMPT = """Classify this legal query into exactly ONE type:
- factual: asking for a specific clause, definition, or single fact
- comparison: comparing two or more clauses, sections, or concepts
- summary: asking for an overview or summary of a topic
- complex: multi-part question requiring reasoning across multiple clauses

Respond with exactly one word: factual, comparison, summary, or complex."""

if st.button("Classify & route", type="primary", icon=":material/play_arrow:", disabled=not (query and api_key)):
    with st.spinner("Classifying query..."):
        raw_qtype = ask(provider, api_key, query, system=CLASSIFIER_PROMPT, temperature=0.0)
        qtype = normalize_label(raw_qtype, ("factual", "comparison", "summary", "complex"))

    type_labels = {
        "factual": "📌 Factual lookup",
        "comparison": "⚖️ Comparison",
        "summary": "📝 Summary",
        "complex": "🧩 Complex reasoning",
    }
    label = type_labels.get(qtype, f"❓ Unclassified — raw model output: {raw_qtype.strip()[:80]!r}")
    st.subheader(f"Classification: {label}")

    with st.spinner(f"Routing to strategy for '{qtype}' queries..."):

        if qtype == "factual":
            st.markdown("**Strategy:** BM25 (fast keyword match)")
            bm25 = BM25Retriever(chunks)
            results = bm25.query(query, k=4)

        elif qtype == "comparison":
            st.markdown("**Strategy:** Hybrid RAG (dense + BM25 fusion)")
            bm25 = BM25Retriever(chunks)
            dense_res = dense_retrieve(query, embed_model, k=8)
            bm25_res = bm25.query(query, k=8)
            results = hybrid_rrf(dense_res, bm25_res, k=60)[:5]

        elif qtype == "summary":
            st.markdown("**Strategy:** Dense retrieval (semantic coverage)")
            results = dense_retrieve(query, embed_model, k=6)

        elif qtype == "complex":
            st.markdown("**Strategy:** Multi-query + rerank")

            sub_queries = generate_variants(provider, api_key, query, SYSTEM_DECOMPOSE, n=3)
            st.info("Sub-queries: " + " | ".join(sub_queries))

            candidates = multi_retrieve(sub_queries, embed_model, k_per_variant=3, top_n=10)
            candidate_chunks = [c for c, _ in candidates]
            results = rerank(query, candidate_chunks, top_k=5)
        else:
            st.warning("Could not classify the query, falling back to dense retrieval.")
            results = dense_retrieve(query, embed_model, k=4)

    result_chunks = [r[0] for r in results]

    with st.spinner("Generating answer..."):
        answer = generate_answer(query, result_chunks, provider, api_key)

    st.subheader(":material/download: Retrieved chunks")
    for i, (chunk, score) in enumerate(results):
        with st.expander(f"Rank {i + 1} — score: {score:.3f}"):
            st.text(chunk[:500])

    st.subheader(":material/chat: Generated answer")
    st.markdown(answer)
elif not api_key:
    st.info(f"Enter your {provider} API key in the sidebar to run this page.", icon=":material/key:")
