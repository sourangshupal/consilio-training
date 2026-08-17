"""BM25 and ColBERT retrieval comparison."""

import sys
sys.path.insert(0, ".")

from pathlib import Path

import streamlit as st
from src.retriever import BM25Retriever
from src.rag_bank import BM25_VS_DENSE_QUERIES

ASSETS = Path(__file__).parent.parent / "assets"

st.title("🧮 BM25 & ColBERT Retrieval")
st.caption(":material/menu_book: Notebook: `02_hybrid_retrieval_reranking.ipynb`")
st.badge("Runs fully offline", icon=":material/wifi_off:", color="green")

st.image(str(ASSETS / "03-retrieval-techniques-slide07-colbert-late-interaction.png"), caption="ColBERT late-interaction (MaxSim)", width="stretch")

if not st.session_state.get("is_processed"):
    st.warning("Process documents on the home page first.", icon=":material/warning:")
    st.stop()

chunks = st.session_state["chunks"]
k = st.slider("Number of results", 1, 10, 5)

st.caption("Example queries — pick one to see where dense/sparse retrieval agree or disagree:")
example_choice = st.selectbox("Example query", list(BM25_VS_DENSE_QUERIES.keys()))
example_query = BM25_VS_DENSE_QUERIES[example_choice]
st.code(example_query, language=None)

tab1, tab2, tab3 = st.tabs(["BM25 (sparse)", "ColBERT (multi-vector)", "Side-by-side comparison"])

with tab1:
    st.subheader(":material/tag: BM25 — sparse keyword retrieval")
    st.markdown(
        "BM25 ranks documents by term frequency and inverse document frequency. "
        "No embeddings needed — it's pure lexical matching."
    )

    bm25_query = st.text_input("BM25 query:", value=example_query, key="bm25_q")

    if st.button("Run BM25", key="bm25_btn", icon=":material/play_arrow:") and bm25_query:
        bm25_retriever = BM25Retriever(chunks)
        results = bm25_retriever.query(bm25_query, k=k)

        st.subheader("BM25 results")
        for i, (chunk, score) in enumerate(results):
            with st.expander(f"Rank {i + 1} — score: {score:.3f}"):
                st.text(chunk)

with tab2:
    st.subheader(":material/token: ColBERT — late-interaction multi-vector retrieval")
    st.markdown(
        "ColBERT stores one vector *per token*, not per document. "
        "At query time, it computes MaxSim: each query token finds its best-matching document token. "
        "This captures fine-grained semantic matches that single-vector embeddings miss."
    )

    colbert_query = st.text_input("ColBERT query:", value=example_query, key="colbert_q")

    if st.button("Run ColBERT", key="colbert_btn", icon=":material/play_arrow:") and colbert_query:
        from src.retriever import ColBERTRetriever

        with st.spinner("Encoding with per-token embeddings (ColBERT MaxSim)..."):
            colbert_retriever = ColBERTRetriever(chunks, st.session_state["embedding_model_name"])
            results = colbert_retriever.query(colbert_query, k=k)

        st.subheader("ColBERT results")
        for i, (chunk, score) in enumerate(results):
            with st.expander(f"Rank {i + 1} — score: {score:.3f}"):
                st.text(chunk)

with tab3:
    st.subheader(":material/compare: BM25 vs. ColBERT — side-by-side")
    st.markdown("Run the same query through both retrievers and compare the rankings side by side.")

    compare_query = st.text_input("Query:", value=example_query, key="compare_q")

    if st.button("Compare", key="compare_btn", icon=":material/play_arrow:") and compare_query:
        bm25_retriever = BM25Retriever(chunks)
        bm25_results = bm25_retriever.query(compare_query, k=k)

        from src.retriever import ColBERTRetriever
        with st.spinner("Encoding with per-token embeddings..."):
            colbert_retriever = ColBERTRetriever(chunks, st.session_state["embedding_model_name"])
            colbert_results = colbert_retriever.query(compare_query, k=k)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**BM25 results**")
            for i, (chunk, score) in enumerate(bm25_results):
                st.markdown(f"**#{i + 1}** — score: {score:.3f}")
                st.text(chunk[:300] + "...")

        with c2:
            st.markdown("**ColBERT results**")
            for i, (chunk, score) in enumerate(colbert_results):
                st.markdown(f"**#{i + 1}** — score: {score:.3f}")
                st.text(chunk[:300] + "...")
