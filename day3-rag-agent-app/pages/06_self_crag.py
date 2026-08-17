"""Self-RAG and Corrective RAG — self-reflection and retrieval grading patterns."""

import sys
sys.path.insert(0, ".")

from pathlib import Path

import streamlit as st
from src.retriever import dense_retrieve
from src.generator import generate_answer
from src.llm_router import ask
from src.rag_bank import CRAG_QUERIES
from src.state import active_api_key

ASSETS = Path(__file__).parent.parent / "assets"

st.title("🪞 Self-RAG & Corrective RAG")
st.caption(":material/menu_book: Notebook: `03_ragas_eval_and_crag_evaluator.ipynb`")
st.badge("Needs API key", icon=":material/wifi:", color="gray")

st.image(str(ASSETS / "07-advanced-rag-slide12-crag-mechanics.png"), caption="CRAG: grade, then route", width="stretch")

st.markdown("RAG with built-in quality checks: self-reflection and retrieval grading.")

if not st.session_state.get("is_processed"):
    st.warning("Process documents on the home page first.", icon=":material/warning:")
    st.stop()

provider = st.session_state["provider"]
api_key = active_api_key()
embed_model = st.session_state["embedding_model_name"]

example_choice = st.selectbox("Example query", list(CRAG_QUERIES.keys()))
custom_query = st.text_input("...or type your own question", placeholder="e.g., What is the governing law?")
query = custom_query.strip() or CRAG_QUERIES[example_choice]

tab1, tab2 = st.tabs(["Self-RAG", "Corrective RAG (CRAG)"])

# --- Self-RAG ---
with tab1:
    st.caption("This is a prompted simulation of Self-RAG's reflection steps, not a trained reflection-token model.")
    st.markdown(
        "**Self-RAG** generates an answer with explicit self-reflection tokens. "
        "The LLM first checks whether each retrieved chunk is relevant, then whether "
        "the generated answer is fully supported by the context."
    )

    if st.button("Run Self-RAG", key="self_btn", icon=":material/play_arrow:", disabled=not (query and api_key)):
        results = dense_retrieve(query, embed_model, k=4)
        context_text = "\n\n---\n\n".join(
            f"[Chunk {i + 1}]\n{r[0]}" for i, r in enumerate(results)
        )

        self_rag_prompt = f"""You are a legal AI that uses Self-RAG — you explicitly reflect on your retrieval and generation.

Step 1 — Relevance Check: For each chunk below, state whether it is RELEVANT or IRRELEVANT to the question.

Context:
{context_text}

Step 2 — Generate an answer using ONLY relevant chunks.

Step 3 — Factuality Check: Verify whether every statement in your answer is directly supported by the relevant chunks. Flag any unsupported statement.

Question: {query}

Respond in this exact format:
RELEVANCE CHECK:
[Chunk 1]: RELEVANT/IRRELEVANT — brief reason
[Chunk 2]: RELEVANT/IRRELEVANT — brief reason
...

ANSWER:
[Your answer here]

FACTUALITY CHECK:
[Your verification here]
"""

        with st.spinner("Running Self-RAG..."):
            output = ask(provider, api_key, self_rag_prompt, temperature=0.0)

        st.subheader(":material/psychology: Self-RAG output")
        st.markdown(output)

# --- Corrective RAG ---
with tab2:
    st.markdown(
        "**Corrective RAG (CRAG)** grades each retrieved chunk, then routes based on quality:\n"
        "- **Any Correct** → use correct + ambiguous chunks\n"
        "- **Only Ambiguous** → use with caution\n"
        "- **All Incorrect** → fallback response, no fabricated answer"
    )

    if st.button("Run CRAG", key="crag_btn", icon=":material/play_arrow:", disabled=not (query and api_key)):
        results = dense_retrieve(query, embed_model, k=4)
        chunks = [r[0] for r in results]

        grades = []
        with st.spinner("Grading chunks..."):
            for chunk in chunks:
                grade = ask(
                    provider, api_key,
                    f"Query: {query}\n\nRetrieved text:\n{chunk[:1000]}",
                    system="Grade whether this retrieved text is relevant to answering the query. "
                           "Respond with exactly one word: Correct, Ambiguous, or Incorrect.",
                    temperature=0.0,
                ).strip()
                grades.append(grade)

        st.subheader(":material/checklist: Chunk grading")
        for i, (chunk, grade) in enumerate(zip(chunks, grades)):
            icon = {"Correct": "✅", "Ambiguous": "⚠️", "Incorrect": "❌"}.get(grade, "❓")
            with st.expander(f"{icon} Chunk {i + 1}: {grade}"):
                st.text(chunk[:500])

        correct_chunks = [c for c, g in zip(chunks, grades) if g == "Correct"]
        ambiguous_chunks = [c for c, g in zip(chunks, grades) if g == "Ambiguous"]

        st.subheader(":material/alt_route: Routing decision")

        if correct_chunks:
            st.success("Correct chunks found — using correct + ambiguous chunks for generation.", icon=":material/check_circle:")
            usable = correct_chunks + ambiguous_chunks
        elif ambiguous_chunks:
            st.warning("Only ambiguous chunks — using with caution.", icon=":material/warning:")
            usable = ambiguous_chunks
        else:
            st.error(
                "All chunks graded Incorrect — this corpus doesn't contain a good answer. "
                "Falling back to an honest 'insufficient information' response instead of a "
                "generic web search (this demo's fallback source), rather than fabricating an answer.",
                icon=":material/block:",
            )
            usable = []

        if usable:
            with st.spinner("Generating answer..."):
                answer = generate_answer(query, usable, provider, api_key)
            st.subheader(":material/chat: Generated answer")
            st.markdown(answer)
        else:
            st.subheader(":material/chat: Generated answer")
            st.markdown(
                f"I don't have enough information in the provided documents to answer "
                f"\"{query}\" — none of the retrieved chunks were graded relevant. "
                "In a production CRAG system, this would trigger a web search or "
                "knowledge-base fallback instead of stopping here."
            )
