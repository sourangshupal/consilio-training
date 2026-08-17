"""Query transformation helpers shared by the Query Transforms and Adaptive
RAG pages — query rewriting, multi-query/decomposition variant generation,
HyDE, and the multi-query retrieve+dedupe step. Kept provider-agnostic via
src/llm_router.ask() so any of OpenAI/Gemini/Groq works.
"""

from src.llm_router import ask
from src.retriever import dense_retrieve

SYSTEM_REWRITE = (
    "Rewrite the user's question into a precise, self-contained search query. "
    "Expand abbreviations, add keywords, and make it specific. Return ONLY the rewritten query."
)

SYSTEM_MULTI_QUERY = (
    "Generate 3 different reformulations of the user's question. "
    "Each should capture a different angle or use different terminology. "
    "Return one per line, no numbering."
)

SYSTEM_DECOMPOSE = (
    "Break this complex question into 2-3 simpler sub-questions. "
    "Return one per line, no numbering."
)

SYSTEM_HYDE = (
    "Write a short hypothetical answer to the user's question. "
    "Use the style and terminology of a legal document. Keep it 3-5 sentences."
)


def rewrite_query(provider: str, api_key: str, query: str) -> str:
    return ask(provider, api_key, query, system=SYSTEM_REWRITE, temperature=0.0)


def hyde_hypothesis(provider: str, api_key: str, query: str) -> str:
    return ask(provider, api_key, query, system=SYSTEM_HYDE, temperature=0.3)


def generate_variants(provider: str, api_key: str, query: str, system_prompt: str, n: int = 3) -> list[str]:
    response = ask(provider, api_key, query, system=system_prompt, temperature=0.3)
    variants = [v.strip() for v in response.split("\n") if v.strip()]
    return variants[:n]


def multi_retrieve(variants: list[str], embed_model: str, k_per_variant: int = 3, top_n: int = 5) -> list[tuple[str, float]]:
    """Retrieves for each query variant, dedupes chunks keeping the highest score."""
    all_chunks: dict[str, float] = {}
    for variant in variants:
        for chunk, score in dense_retrieve(variant, embed_model, k=k_per_variant):
            all_chunks[chunk] = max(all_chunks.get(chunk, 0.0), score)
    return sorted(all_chunks.items(), key=lambda x: x[1], reverse=True)[:top_n]
