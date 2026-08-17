"""RAGAS evaluation pipeline for measuring RAG quality.

The judge LLM is built from whichever provider is selected in the sidebar
(OpenAI/Gemini/Groq) via src/llm_router.get_langchain_chat_model(). RAGAS's
embeddings step is intentionally decoupled from the LLM provider — it always
uses the local sentence-transformers embedding model already loaded for
retrieval, so evaluation never requires an OpenAI key just to run.
"""

import pandas as pd
import streamlit as st

from src.rag_bank import EVAL_QUESTIONS


def run_ragas_evaluation(
    questions: list[str],
    answers: list[str],
    contexts_list: list[list[str]],
    provider: str,
    api_key: str,
    embedding_model_name: str = "all-MiniLM-L6-v2",
) -> pd.DataFrame:
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
    eval_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=f"sentence-transformers/{embedding_model_name}"))

    dataset = Dataset.from_dict(
        {
            "user_input": questions,
            "response": answers,
            "retrieved_contexts": contexts_list,
        }
    )

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=eval_llm,
        embeddings=eval_embeddings,
    )

    return result.to_pandas()


@st.cache_resource
def load_test_questions() -> list[str]:
    return EVAL_QUESTIONS
