"""Offline smoke test for the Day 3 app — the closest thing this app has to a
test suite. Runs the whole no-API-key path (load -> chunk -> embed -> index ->
dense/BM25/ColBERT/RRF/rerank), the RAGAS import + metric selection, and the
LLM-verdict parsers. No network, no API key.

    .venv/bin/python smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.chunker import chunk_documents
from src.embedder import embed_chunks
from src.evaluator import references_for
from src.llm_router import normalize_label
from src.rag_bank import EVAL_QUESTIONS, EVAL_REFERENCES, sample_documents
from src.reranker import rerank
from src.retriever import (
    BM25Retriever,
    ColBERTRetriever,
    build_chromadb,
    dense_retrieve,
    hybrid_rrf,
)

QUERY = "What are the indemnification obligations?"


def test_retrieval_pipeline() -> None:
    docs = sample_documents()
    chunks = chunk_documents(docs, "Recursive", 500, 75)
    assert len(chunks) > 10, chunks

    embeddings = embed_chunks(chunks)
    assert embeddings.shape[0] == len(chunks)
    build_chromadb(chunks, embeddings)

    dense = dense_retrieve(QUERY, "all-MiniLM-L6-v2", k=4)
    assert len(dense) == 4 and all(s > 0 for _, s in dense)

    bm25 = BM25Retriever(chunks).query("indemnification", k=4)
    assert len(bm25) == 4

    fused = hybrid_rrf(dense, bm25, k=60)
    assert all(isinstance(c, str) and s > 0 for c, s in fused), "RRF must return real scores"
    assert [s for _, s in fused] == sorted((s for _, s in fused), reverse=True)

    reranked = rerank(QUERY, [c for c, _ in fused][:8], top_k=3)
    assert len(reranked) == 3

    colbert = ColBERTRetriever(chunks[:8]).query("indemnification", k=3)
    assert len(colbert) == 3


def test_ragas_imports_and_metric_selection() -> None:
    from src.evaluator import REFERENCE_METRICS, _patch_langchain_community_vertexai

    _patch_langchain_community_vertexai()
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    for metric in (context_precision, context_recall):
        assert metric.name in REFERENCE_METRICS
        assert "reference" in metric.required_columns["SINGLE_TURN"]
    for metric in (faithfulness, answer_relevancy):
        assert "reference" not in metric.required_columns["SINGLE_TURN"]

    assert references_for(EVAL_QUESTIONS) == [EVAL_REFERENCES[q] for q in EVAL_QUESTIONS]
    assert references_for(EVAL_QUESTIONS + ["a question with no ground truth"]) is None


def test_verdict_parsing() -> None:
    grades = ("Incorrect", "Correct", "Ambiguous")
    assert normalize_label("Correct", grades) == "Correct"
    assert normalize_label("**Correct.**", grades) == "Correct"
    assert normalize_label("Grade: Incorrect", grades) == "Incorrect"
    assert normalize_label("no verdict here", grades) == ""

    types_ = ("factual", "comparison", "summary", "complex")
    assert normalize_label("Complex.", types_) == "complex"
    assert normalize_label("The answer is: summary", types_) == "summary"


if __name__ == "__main__":
    test_retrieval_pipeline()
    test_ragas_imports_and_metric_selection()
    test_verdict_parsing()
    print("smoke test OK")
