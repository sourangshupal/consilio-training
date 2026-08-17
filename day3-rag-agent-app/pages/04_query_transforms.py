"""Query Transformations — Query Rewriting, Multi-Query, and HyDE."""

import sys
sys.path.insert(0, ".")

import streamlit as st
from src.retriever import dense_retrieve
from src.generator import generate_answer
from src.query_transforms import (
    SYSTEM_MULTI_QUERY,
    generate_variants,
    hyde_hypothesis,
    multi_retrieve,
    rewrite_query,
)
from src.rag_bank import QUERY_TRANSFORM_QUERIES
from src.state import active_api_key

st.title("🪄 Query Transformations")
st.caption(":material/menu_book: Notebook: `01_vanilla_rag_legal_documents.ipynb` (extended)")
st.badge("Needs API key", icon=":material/wifi:", color="gray")
st.markdown("Improve retrieval by transforming the user's question before searching.")

if not st.session_state.get("is_processed"):
    st.warning("Process documents on the home page first.", icon=":material/warning:")
    st.stop()

provider = st.session_state["provider"]
api_key = active_api_key()
embed_model = st.session_state["embedding_model_name"]

example_choice = st.selectbox("Example question", QUERY_TRANSFORM_QUERIES)
custom_query = st.text_input("...or type your own question")
query = custom_query.strip() or example_choice

tab1, tab2, tab3 = st.tabs(["Query rewriting", "Multi-query", "HyDE"])

if not api_key:
    st.info(f"Enter your {provider} API key in the sidebar to run these transforms.", icon=":material/key:")

# --- Query Rewriting ---
with tab1:
    st.markdown(
        "**Query rewriting** uses an LLM to rephrase a vague or poorly-worded question "
        "into a more precise query that retrieves better chunks."
    )

    if st.button("Rewrite & search", key="rw_btn", icon=":material/play_arrow:", disabled=not (query and api_key)):
        with st.spinner("Rewriting query..."):
            rewritten = rewrite_query(provider, api_key, query)

        st.info(f"**Original:** {query}")
        st.success(f"**Rewritten:** {rewritten}")

        results = dense_retrieve(rewritten, embed_model, k=4)
        answer = generate_answer(query, [r[0] for r in results], provider, api_key)

        st.subheader(":material/chat: Answer")
        st.markdown(answer)

# --- Multi-Query ---
with tab2:
    st.markdown(
        "**Multi-query** generates several variant questions, retrieves chunks for each, "
        "then deduplicates the results for broader coverage."
    )

    if st.button("Generate variants & search", key="mq_btn", icon=":material/play_arrow:", disabled=not (query and api_key)):
        with st.spinner("Generating query variants..."):
            variants = generate_variants(provider, api_key, query, SYSTEM_MULTI_QUERY, n=3)

        st.info(f"**Original:** {query}")
        for i, v in enumerate(variants):
            st.success(f"**Variant {i + 1}:** {v}")

        unique_chunks = multi_retrieve(variants, embed_model, k_per_variant=3, top_n=5)
        answer = generate_answer(query, [c[0] for c in unique_chunks], provider, api_key)

        st.subheader(f":material/chat: Answer (from {len(unique_chunks)} unique chunks)")
        st.markdown(answer)

# --- HyDE ---
with tab3:
    st.markdown(
        "**HyDE** (Hypothetical Document Embeddings) asks the LLM to generate a hypothetical "
        "answer first, then embeds *that* answer to find chunks that are semantically similar. "
        "This works well when the query and documents use different vocabulary."
    )

    if st.button("Generate hypothesis & search", key="hyde_btn", icon=":material/play_arrow:", disabled=not (query and api_key)):
        with st.spinner("Generating hypothetical answer..."):
            hypothesis = hyde_hypothesis(provider, api_key, query)

        st.info(f"**Original query:** {query}")
        st.success(f"**Hypothetical answer:** {hypothesis}")

        results = dense_retrieve(hypothesis, embed_model, k=4)
        answer = generate_answer(query, [r[0] for r in results], provider, api_key)

        st.subheader(":material/chat: Answer")
        st.markdown(answer)
