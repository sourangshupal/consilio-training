"""Vanilla RAG — basic dense retrieval + generation."""

import sys
sys.path.insert(0, ".")

from pathlib import Path

import streamlit as st
from src.retriever import dense_retrieve
from src.generator import generate_answer, build_prompt_text
from src.rag_bank import VANILLA_RAG_QUERIES
from src.state import active_api_key

ASSETS = Path(__file__).parent.parent / "assets"

st.title("🔍 Vanilla RAG")
st.caption(":material/menu_book: Notebook: `01_vanilla_rag_legal_documents.ipynb`")
st.badge("Needs API key for generation", icon=":material/wifi:", color="gray")

st.image(str(ASSETS / "01-rag-fundamentals-slide05-vanilla-rag-architecture.png"), caption="Vanilla RAG architecture", width="stretch")

if not st.session_state.get("is_processed"):
    st.warning("Process documents on the home page first.", icon=":material/warning:")
    st.stop()

provider = st.session_state["provider"]
api_key = active_api_key()

st.header(":material/search: Retrieve & generate", divider="gray")
choice = st.selectbox("Example question", VANILLA_RAG_QUERIES)
custom = st.text_input("...or type your own question")
query = custom.strip() or choice

k = st.slider("Number of chunks to retrieve", 1, 10, 4)

if st.button("Search", type="primary", icon=":material/play_arrow:", disabled=not api_key) and query:
    with st.spinner("Retrieving and generating..."):
        results = dense_retrieve(query, st.session_state["embedding_model_name"], k=k)
        answer = generate_answer(query, [r[0] for r in results], provider, api_key)
        prompt_text = build_prompt_text(query, [r[0] for r in results])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(":material/download: Retrieved chunks")
        for i, (chunk, score) in enumerate(results):
            with st.expander(f"Chunk {i + 1} — score: {score:.3f}"):
                st.text(chunk)

    with col2:
        st.subheader(":material/chat: Generated answer")
        st.markdown(answer)

    with st.expander(":material/code: Full prompt sent to LLM"):
        st.code(prompt_text, language="text")
elif not api_key:
    st.info(f"Enter your {provider} API key in the sidebar to generate an answer.", icon=":material/key:")
