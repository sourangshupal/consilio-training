"""Grounded RAG answer generation, routed through the selected LLM provider."""

from src.llm_router import ask

SYSTEM_PROMPT = """You are a legal AI assistant. Answer the user's question using ONLY the provided context below.
If the context does not contain enough information to answer the question, say so clearly.
Do not make up facts. Cite specific parts of the context in your answer."""


def build_prompt_text(query: str, context_chunks: list[str]) -> str:
    context_text = "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{c}" for i, c in enumerate(context_chunks)
    )
    return f"""Context:
{context_text}

Question: {query}

Answer:"""


def generate_answer(query: str, context_chunks: list[str], provider: str, api_key: str) -> str:
    prompt = build_prompt_text(query, context_chunks)
    return ask(provider, api_key, prompt, system=SYSTEM_PROMPT, temperature=0.0)
