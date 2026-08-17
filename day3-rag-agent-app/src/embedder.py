"""Embedding model wrapper with Streamlit caching."""

import numpy as np
import streamlit as st


@st.cache_resource
def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_chunks(chunks: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    model = get_embedding_model(model_name)
    return model.encode(chunks, show_progress_bar=True)
