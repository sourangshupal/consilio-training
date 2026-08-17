"""Session state initialization helpers for the Day 3 RAG & Agents app."""

import streamlit as st

DEFAULTS = {
    "provider": "Groq",
    "openai_api_key": "",
    "gemini_api_key": "",
    "groq_api_key": "",
    "embedding_model_name": "all-MiniLM-L6-v2",
    "documents": [],
    "chunks": [],
    "chunk_strategy": "Recursive",
    "chunk_size": 500,
    "chunk_overlap": 75,
    "is_processed": False,
}


def init_session():
    for key, default in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def active_api_key() -> str:
    provider = st.session_state.get("provider", "Groq")
    return {
        "OpenAI": st.session_state.get("openai_api_key", ""),
        "Gemini": st.session_state.get("gemini_api_key", ""),
        "Groq": st.session_state.get("groq_api_key", ""),
    }.get(provider, "")
