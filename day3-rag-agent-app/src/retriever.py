"""Retrievers: Dense (ChromaDB), BM25, ColBERT (from-scratch late-interaction), and hybrid RRF."""

import numpy as np
import streamlit as st


def _get_chroma_client():
    """Ephemeral in-memory Chroma client, one per browser session (stored in
    session_state, not cache_resource, so each user gets an isolated store
    instead of sharing one global client across sessions)."""
    import chromadb

    if "chroma_client" not in st.session_state:
        st.session_state["chroma_client"] = chromadb.Client()
    return st.session_state["chroma_client"]


def build_chromadb(chunks: list[str], embeddings: np.ndarray) -> None:
    client = _get_chroma_client()
    try:
        client.delete_collection("documents")
    except Exception:
        pass
    collection = client.create_collection("documents")

    ids = [str(i) for i in range(len(chunks))]
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=chunks,
    )
    st.session_state["chroma_ready"] = True


def dense_retrieve(query: str, embedding_model_name: str, k: int = 5) -> list[tuple[str, float]]:
    from src.embedder import get_embedding_model

    model = get_embedding_model(embedding_model_name)
    query_embedding = model.encode([query])[0]

    client = _get_chroma_client()
    collection = client.get_collection("documents")
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=k)

    chunks = results["documents"][0]
    distances = results.get("distances", [[0] * len(chunks)])[0]
    scores = [1 / (1 + d) for d in distances] if distances else [1.0] * len(chunks)
    return list(zip(chunks, scores))


class BM25Retriever:
    def __init__(self, chunks: list[str]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.tokenized = [c.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized)

    def query(self, text: str, k: int = 5) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(text.lower().split())
        top_idx = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]


class ColBERTRetriever:
    """Demonstrates ColBERT's late-interaction (MaxSim) scoring from scratch.

    Instead of embedding whole documents into a single vector, ColBERT stores
    one vector PER TOKEN. At query time, each query token finds its best-matching
    document token via cosine similarity, then these per-token max scores are summed.
    This captures fine-grained token-level matches that single-vector models miss.

    Uses the underlying BERT model from sentence-transformers for per-token embeddings,
    correctly demonstrating the architecture — even though real ColBERT uses a
    fine-tuned ColBERT checkpoint.
    """

    def __init__(self, chunks: list[str], model_name: str = "all-MiniLM-L6-v2"):
        from src.embedder import get_embedding_model
        import torch

        self.chunks = chunks
        model = get_embedding_model(model_name)
        self.transformer = model[0]
        self.auto_model = self.transformer.auto_model.to("cpu")
        self.device = torch.device("cpu")

    def _get_token_embeddings(self, text: str) -> np.ndarray:
        import torch

        tokens = self.transformer.tokenize([text])
        tokens = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in tokens.items()}
        tokens.pop("modality", None)

        with torch.no_grad():
            out = self.auto_model(**tokens)

        token_embeds = out.last_hidden_state[0].cpu().numpy()
        attention_mask = tokens["attention_mask"].cpu().numpy()[0]
        return token_embeds[attention_mask == 1]

    def _maxsim(self, query_embeds: np.ndarray, doc_embeds: np.ndarray) -> float:
        from sklearn.metrics.pairwise import cosine_similarity

        sim = cosine_similarity(query_embeds, doc_embeds)
        return float(np.sum(np.max(sim, axis=1)))

    def query(self, text: str, k: int = 5) -> list[tuple[str, float]]:
        query_tokens = self._get_token_embeddings(text)
        scores = []

        for doc_text in self.chunks:
            doc_tokens = self._get_token_embeddings(doc_text)
            score = self._maxsim(query_tokens, doc_tokens)
            scores.append(score)

        top_idx = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]


def hybrid_rrf(
    dense_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion. Returns (chunk, rrf_score) pairs, highest first,
    so callers can display the fused score instead of a placeholder."""
    chunk_to_score: dict[str, float] = {}

    for results in (dense_results, bm25_results):
        for rank, (chunk, _) in enumerate(results):
            chunk_to_score[chunk] = chunk_to_score.get(chunk, 0.0) + 1 / (k + rank + 1)

    return sorted(chunk_to_score.items(), key=lambda x: x[1], reverse=True)
