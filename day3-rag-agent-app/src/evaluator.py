"""RAGAS evaluation pipeline for measuring RAG quality.

The judge LLM is built from whichever provider is selected in the sidebar
(OpenAI/Gemini/Groq) via src/llm_router.get_langchain_chat_model(). RAGAS's
embeddings step is intentionally decoupled from the LLM provider — it always
uses the local sentence-transformers embedding model already loaded for
retrieval, so evaluation never requires an OpenAI key just to run.
"""

import sys
import types

import pandas as pd

from src.rag_bank import EVAL_QUESTIONS, EVAL_REFERENCES

# Metrics that grade retrieval against a ground-truth answer. RAGAS raises
# ValueError if they run without a `reference` column, so they are only
# included when every question has one.
REFERENCE_METRICS = ("context_precision", "context_recall")


def _patch_langchain_community_vertexai() -> None:
    """ragas 0.4.x imports `langchain_community.chat_models.vertexai`, which
    langchain-community >= 0.4 deleted. The symbol is only used in an
    isinstance table for Vertex models this app never builds, so register a
    stub module rather than pinning the whole LangChain stack back to 0.3."""
    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    try:
        __import__(name)
    except ImportError:
        stub = types.ModuleType(name)

        class ChatVertexAI:  # noqa: D401 - placeholder, never instantiated
            """Placeholder for the removed langchain-community ChatVertexAI."""

        stub.ChatVertexAI = ChatVertexAI
        sys.modules[name] = stub


def run_ragas_evaluation(
    questions: list[str],
    answers: list[str],
    contexts_list: list[list[str]],
    provider: str,
    api_key: str,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    references: list[str] | None = None,
) -> pd.DataFrame:
    _patch_langchain_community_vertexai()

    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from langchain_huggingface import HuggingFaceEmbeddings

    from src.llm_router import get_langchain_chat_model

    eval_llm = LangchainLLMWrapper(get_langchain_chat_model(provider, api_key, temperature=0.0))
    eval_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=f"sentence-transformers/{embedding_model_name}")
    )

    columns = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts_list,
    }
    metrics = [faithfulness, answer_relevancy]
    if references and all(r for r in references):
        columns["reference"] = references
        metrics += [context_precision, context_recall]

    result = evaluate(
        dataset=Dataset.from_dict(columns),
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
        # Default is False, which turns every judge failure into a silent NaN
        # column — the caller sees an all-NaN table with no clue why.
        raise_exceptions=True,
    )

    return result.to_pandas()


def load_test_questions() -> list[str]:
    return list(EVAL_QUESTIONS)


def references_for(questions: list[str]) -> list[str] | None:
    """Ground-truth answers aligned to `questions`, or None if the user edited
    the question list so that any question has no reference."""
    refs = [EVAL_REFERENCES.get(q, "") for q in questions]
    return refs if all(refs) else None


def check_judge_llm(provider: str, api_key: str) -> str | None:
    """One cheap round-trip to the judge model before the eval loop spends
    real calls on generation. Returns an error message, or None if it works.

    The judge goes through LangChain (get_langchain_chat_model), not the plain
    SDK path used for generation, so it can fail on its own — a bad model id,
    a key without access, or a provider that rejects the request shape.
    """
    _patch_langchain_community_vertexai()

    from src.llm_router import get_langchain_chat_model

    try:
        response = get_langchain_chat_model(provider, api_key, temperature=0.0).invoke(
            "Reply with the single word: OK"
        )
    except Exception as exc:  # noqa: BLE001 - surface any provider/SDK error verbatim
        return f"{type(exc).__name__}: {exc}"

    if not str(getattr(response, "content", "")).strip():
        return "Judge model returned an empty response."
    return None
