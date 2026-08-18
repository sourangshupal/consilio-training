"""Hybrid RAG — dense + sparse fusion + cross-encoder reranking."""

import sys
sys.path.insert(0, ".")

from pathlib import Path

import streamlit as st
from src.retriever import dense_retrieve, BM25Retriever, hybrid_rrf
from src.reranker import rerank
from src.generator import generate_answer
from src.rag_bank import HYBRID_RERANK_QUERIES
from src.state import active_api_key

ASSETS = Path(__file__).parent.parent / "assets"

st.title("🔄 Hybrid RAG & Reranking")
st.caption(":material/menu_book: Notebook: `02_hybrid_retrieval_reranking.ipynb`")
st.badge("Needs API key for generation", icon=":material/wifi:", color="gray")

st.image(str(ASSETS / "04-retrieval-techniques-slide09-hybrid-rag.png"), caption="Hybrid RAG: dense + sparse fusion", width="stretch")

if not st.session_state.get("is_processed"):
    st.warning("Process documents on the home page first.", icon=":material/warning:")
    st.stop()

provider = st.session_state["provider"]
api_key = active_api_key()
chunks = st.session_state["chunks"]

col1, col2, col3 = st.columns(3)
with col1:
    dense_k = st.slider("Dense k", 3, 15, 10, key="dense_k")
with col2:
    bm25_k = st.slider("BM25 k", 3, 15, 10, key="bm25_k")
with col3:
    rrf_k = st.slider("RRF constant", 10, 120, 60, key="rrf_k")

rerank_k = st.slider("Rerank top-N", 3, 10, 5, key="rerank_k")

example_query = st.selectbox("Example query", HYBRID_RERANK_QUERIES)
query = st.text_input("...or type your own query:", placeholder="e.g., What are the indemnification obligations?") or example_query

if st.button("Run hybrid RAG", type="primary", icon=":material/play_arrow:", disabled=not api_key) and query:
    bm25_retriever = BM25Retriever(chunks)

    with st.spinner("Retrieving..."):
        dense_res = dense_retrieve(query, st.session_state["embedding_model_name"], k=dense_k)
        bm25_res = bm25_retriever.query(query, k=bm25_k)

        fused = hybrid_rrf(dense_res, bm25_res, k=rrf_k)
        # Rerank a candidate pool larger than rerank_k so the cross-encoder can
        # actually promote chunks RRF ranked lower — truncating to rerank_k
        # *before* reranking would defeat the point of reranking.
        candidate_pool = fused[: max(rerank_k * 3, dense_k, bm25_k)]

    with st.spinner("Reranking..."):
        reranked = rerank(query, [c for c, _ in candidate_pool], top_k=rerank_k)

    with st.spinner("Generating answer..."):
        answer = generate_answer(query, [r[0] for r in reranked], provider, api_key)

    st.subheader(":material/query_stats: Retrieval results comparison")

    tabs = st.tabs(["Dense-only", "BM25-only", "Hybrid (RRF)", "After reranking"])
    for tab, results, label in zip(
        tabs,
        [dense_res, bm25_res, candidate_pool, reranked],
        ["Dense", "BM25", "Hybrid (RRF)", "Reranked"],
    ):
        with tab:
            for i, (chunk, score) in enumerate(results):
                with st.expander(f"Rank {i + 1} — score: {score:.3f}"):
                    st.text(chunk)

    st.subheader(":material/chat: Generated answer")
    st.markdown(answer)
elif not api_key:
    st.info(f"Enter your {provider} API key in the sidebar to generate an answer.", icon=":material/key:")
