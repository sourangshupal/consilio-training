"""RAGAS Evaluation — measure RAG quality with faithfulness, relevancy, and precision."""

import sys
sys.path.insert(0, ".")

from pathlib import Path

import streamlit as st
from src.retriever import dense_retrieve
from src.generator import generate_answer
from src.evaluator import check_judge_llm, references_for, run_ragas_evaluation, load_test_questions
from src.state import active_api_key

ASSETS = Path(__file__).parent.parent / "assets"

st.title("📊 RAGAS Evaluation")
st.caption(":material/menu_book: Notebook: `03_ragas_eval_and_crag_evaluator.ipynb`")
st.badge("Needs API key for judging", icon=":material/wifi:", color="gray")

st.image(str(ASSETS / "06-advanced-rag-slide06-ragas-faithfulness.png"), caption="RAGAS: LLM-as-judge metrics", width="stretch")

if not st.session_state.get("is_processed"):
    st.warning("Process documents on the home page first.", icon=":material/warning:")
    st.stop()

provider = st.session_state["provider"]
api_key = active_api_key()
embed_model = st.session_state["embedding_model_name"]

st.markdown("""
RAGAS uses an **LLM-as-judge** to evaluate your RAG pipeline across four metrics:

- **Faithfulness** — Is the answer supported by the retrieved context?
- **Answer relevancy** — Does the answer actually address the question?
- **Context precision** — Are relevant chunks ranked higher than irrelevant ones?
- **Context recall** — Did we retrieve all the necessary information?

*Targets: Faithfulness ≥ 0.9, Relevancy ≥ 0.85, Precision ≥ 0.8*

The judge LLM is whichever provider is selected in the sidebar — evaluation embeddings
always use the local sentence-transformers model, so this never needs an OpenAI key
specifically.
""")

st.subheader(":material/quiz: Test questions")
test_questions = load_test_questions()

manual_questions = st.text_area(
    "Edit test questions (one per line), or use the defaults:",
    value="\n".join(test_questions),
    height=160,
)
questions = [q.strip() for q in manual_questions.split("\n") if q.strip()]

k = st.slider("Chunks to retrieve per question", 1, 8, 3)

references = references_for(questions)
if references is None:
    st.info(
        "Context precision and context recall are reference-based — they need a "
        "ground-truth answer per question. Custom questions have none, so this run "
        "will report faithfulness and answer relevancy only. Restore the default "
        "questions to get all four metrics.",
        icon=":material/info:",
    )

if st.button("Run RAGAS evaluation", type="primary", icon=":material/play_arrow:", disabled=not api_key):
    # Preflight the judge before spending a generation call per question — the
    # judge runs through LangChain, so it can fail even when generation works.
    with st.spinner(f"Checking the {provider} judge model..."):
        judge_error = check_judge_llm(provider, api_key)
    if judge_error:
        st.error(
            f"The {provider} judge model is not reachable, so every metric would come "
            f"back empty. Fix this first:\n\n`{judge_error}`",
            icon=":material/error:",
        )
        if "model_not_found" in judge_error or "does not exist" in judge_error:
            from src.llm_router import DEFAULT_MODELS, list_models

            st.caption(
                f"`DEFAULT_MODELS['{provider}']` is currently "
                f"`{DEFAULT_MODELS[provider]}`. Model ids your key can use:"
            )
            try:
                st.code("\n".join(list_models(provider, api_key)), language="text")
            except Exception as exc:  # noqa: BLE001
                st.caption(f"Could not fetch the model list: {exc}")
        st.stop()

    answers = []
    contexts_list = []

    progress = st.progress(0)
    status_text = st.empty()

    for i, q in enumerate(questions):
        status_text.text(f"Evaluating: {q}")
        results = dense_retrieve(q, embed_model, k=k)
        retrieved_chunks = [r[0] for r in results]
        answer = generate_answer(q, retrieved_chunks, provider, api_key)
        answers.append(answer)
        contexts_list.append(retrieved_chunks)
        progress.progress((i + 1) / len(questions))

    status_text.text(f"Computing RAGAS metrics (judge: {provider})...")

    try:
        df = run_ragas_evaluation(
            questions, answers, contexts_list, provider, api_key, embed_model,
            references=references,
        )
        status_text.empty()

        st.subheader(":material/table_chart: Evaluation results")
        st.dataframe(df, width="stretch")

        metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        available_cols = [c for c in metric_cols if c in df.columns]

        if available_cols:
            st.subheader(":material/bar_chart: Score distribution")
            chart_data = df[available_cols].mean().to_frame("Average score")
            st.bar_chart(chart_data)

            st.subheader(":material/flag: Target comparison")
            targets = {
                "faithfulness": 0.9,
                "answer_relevancy": 0.85,
                "context_precision": 0.8,
                "context_recall": 0.8,
            }
            for col in available_cols:
                avg = df[col].mean()
                target = targets.get(col, 0.8)
                if avg >= target:
                    st.success(f"**{col}**: {avg:.3f} — meets target (≥{target})", icon=":material/check_circle:")
                else:
                    st.warning(f"**{col}**: {avg:.3f} — below target (≥{target})", icon=":material/warning:")
    except Exception as e:
        st.error(f"RAGAS evaluation failed: {e}", icon=":material/error:")
elif not api_key:
    st.info(f"Enter your {provider} API key in the sidebar to run evaluation.", icon=":material/key:")
