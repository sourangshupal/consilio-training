"""Day 3 — RAG & Agents. Entry point: navigation, sidebar, and the home page
(document upload/processing + sample corpus loader)."""

import sys
import streamlit as st

sys.path.insert(0, ".")
from src.state import init_session
from src.loader import load_document
from src.chunker import chunk_documents
from src.embedder import embed_chunks
from src.retriever import build_chromadb
from src.rag_bank import sample_documents

st.set_page_config(
    page_title="Day 3 — RAG & Agents",
    page_icon="📚",
    layout="wide",
)

init_session()


def _process_documents(documents: list[dict]) -> None:
    with st.status("Processing documents...", expanded=True) as status:
        st.session_state["documents"] = documents
        for doc in documents:
            st.write(f"✓ Loaded: *{doc['filename']}* ({len(doc['text']):,} chars, {doc['type']})")

        st.write("Chunking...")
        chunks = chunk_documents(
            documents,
            st.session_state["chunk_strategy"],
            st.session_state["chunk_size"],
            st.session_state["chunk_overlap"],
        )
        st.session_state["chunks"] = chunks
        st.write(f"✓ Created {len(chunks)} chunks (strategy: {st.session_state['chunk_strategy']})")

        st.write("Embedding chunks...")
        embeddings = embed_chunks(chunks, st.session_state["embedding_model_name"])
        st.write(f"✓ Embedded {len(chunks)} chunks into {embeddings.shape[1]}-d vectors")

        st.write("Building vector index...")
        build_chromadb(chunks, embeddings)
        st.write("✓ Chroma collection ready (in-memory, this session only)")

        st.session_state["is_processed"] = True
        status.update(label="Processing complete!", state="complete")


def home():
    st.title("📚 Day 3 — RAG & Agents")
    st.caption("Consilio 4-Day AI Training — interactive companion to the day3 notebooks")
    st.space("small")

    modules = [
        (":material/search:", "Vanilla RAG", "Baseline dense retrieval and grounded generation.", "green"),
        (":material/compare_arrows:", "BM25 & ColBERT", "Sparse keyword search vs. late-interaction retrieval.", "green"),
        (":material/merge:", "Hybrid & rerank", "RRF fusion + cross-encoder reranking.", "green"),
        (":material/auto_fix_high:", "Query transforms", "Rewriting, multi-query, and HyDE.", "gray"),
        (":material/fact_check:", "RAG evaluation", "RAGAS faithfulness, relevancy, precision, recall.", "gray"),
        (":material/self_improvement:", "Self-RAG & CRAG", "Self-reflection and corrective retrieval grading.", "gray"),
        (":material/route:", "Adaptive RAG", "Query classification and strategy routing.", "gray"),
        (":material/smart_toy:", "LangGraph ReAct agent", "A minimal tool-using agent with a clause-lookup tool.", "gray"),
        (":material/groups:", "Multi-agent supervisor", "Research + risk-analysis workers with human-in-the-loop.", "gray"),
    ]
    cols = st.columns(3)
    for i, (icon, name, desc, badge_color) in enumerate(modules):
        with cols[i % 3].container(border=True, height="stretch"):
            st.markdown(f"{icon} **{name}**")
            st.caption(desc)
            st.badge("Offline" if badge_color == "green" else "Needs API key", color=badge_color)

    st.space("medium")
    st.header(":material/description: Document corpus", divider="gray")

    c1, c2 = st.columns([3, 1])
    with c1:
        st.caption("Load the bundled sample legal corpus, or upload your own PDF/DOCX/Markdown files.")
    with c2:
        if st.button("Load sample corpus", icon=":material/library_add:", type="primary"):
            _process_documents(sample_documents())

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or Markdown files",
        type=["pdf", "docx", "md"],
        accept_multiple_files=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state["chunk_strategy"] = st.radio(
            "Chunking strategy",
            options=["Recursive", "Fixed-size"],
            help="Recursive splits on natural boundaries (paragraphs, sentences). Fixed-size splits by character count.",
        )
    with col2:
        st.session_state["chunk_size"] = st.slider("Chunk size (chars)", min_value=200, max_value=1500, value=500, step=50)
    with col3:
        st.session_state["chunk_overlap"] = st.slider("Chunk overlap (chars)", min_value=0, max_value=300, value=75, step=25)

    if st.button("Process uploaded documents", icon=":material/build:", disabled=not uploaded_files):
        documents = []
        for uf in uploaded_files:
            doc = load_document(uf)
            if doc:
                documents.append(doc)
            else:
                st.warning(f"Unsupported file type: {uf.name}")
        _process_documents(documents)

    if st.session_state["is_processed"]:
        st.success(
            f"**{len(st.session_state['documents'])} documents** loaded, "
            f"**{len(st.session_state['chunks'])} chunks** indexed. "
            "Pick a page from the sidebar to explore RAG & agent techniques.",
            icon=":material/check_circle:",
        )
        st.subheader(":material/preview: Chunk preview (first 3)")
        for i, chunk in enumerate(st.session_state["chunks"][:3]):
            with st.expander(f"Chunk {i + 1} ({len(chunk)} chars)", expanded=i == 0):
                st.text(chunk)
    else:
        st.info("Load the sample corpus or upload + process your own documents to get started.", icon=":material/rocket_launch:")


pages = [
    st.Page(home, title="Home", icon=":material/home:", default=True, url_path="home"),
    st.Page("pages/01_vanilla_rag.py", title="Vanilla RAG", icon="🔍", url_path="vanilla-rag"),
    st.Page("pages/02_bm25_colbert.py", title="BM25 & ColBERT", icon="🧮", url_path="bm25-colbert"),
    st.Page("pages/03_hybrid_rerank.py", title="Hybrid & rerank", icon="🔄", url_path="hybrid-rerank"),
    st.Page("pages/04_query_transforms.py", title="Query transforms", icon="🪄", url_path="query-transforms"),
    st.Page("pages/05_rag_evaluation.py", title="RAG evaluation", icon="📊", url_path="rag-evaluation"),
    st.Page("pages/06_self_crag.py", title="Self-RAG & CRAG", icon="🪞", url_path="self-crag"),
    st.Page("pages/07_adaptive_rag.py", title="Adaptive RAG", icon="🧠", url_path="adaptive-rag"),
    st.Page("pages/08_langgraph_react_agent.py", title="LangGraph ReAct agent", icon="🤖", url_path="react-agent"),
    st.Page("pages/09_multi_agent_supervisor.py", title="Multi-agent supervisor", icon="👥", url_path="supervisor"),
]

with st.sidebar:
    st.markdown("### 📚 Day 3 — RAG & Agents")
    st.caption("Consilio 4-Day AI Training")

    with st.container(border=True):
        st.markdown(":material/key: **LLM provider**")
        st.selectbox("Provider", ["Groq", "OpenAI", "Gemini"], key="provider", label_visibility="collapsed")
        st.text_input("Groq API key", type="password", key="groq_api_key", icon=":material/lock:", help="Get a free key at console.groq.com")
        st.text_input("OpenAI API key", type="password", key="openai_api_key", icon=":material/lock:")
        st.text_input("Gemini API key", type="password", key="gemini_api_key", icon=":material/lock:")

        active_key = {
            "OpenAI": st.session_state["openai_api_key"],
            "Gemini": st.session_state["gemini_api_key"],
            "Groq": st.session_state["groq_api_key"],
        }[st.session_state["provider"]]
        if active_key:
            st.badge(f"{st.session_state['provider']} key set", icon=":material/check_circle:", color="green")
        else:
            st.badge("No key set", icon=":material/info:", color="gray")

    st.session_state["embedding_model_name"] = st.selectbox(
        "Embedding model",
        options=["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
        index=0,
        help="all-MiniLM-L6-v2 is fast and free (384-d). all-mpnet-base-v2 is more accurate (768-d).",
    )

    st.space("small")
    st.markdown(":green-badge[:material/wifi_off: Offline] :gray-badge[:material/wifi: Needs API key]")
    st.caption("Retrieval-only pages (01-03) run offline; generation, evaluation, and agent pages need an LLM key.")

nav = st.navigation(pages)
nav.run()
