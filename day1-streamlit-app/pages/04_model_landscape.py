from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from src.clause_bank import CLASSIFY_TARGET_CLAUSES, CLAUSE_TAXONOMY, QWEN_PROMPT_PRESETS
from src.models import load_legalbert_model, load_qwen_pipeline
from src.state import init_session

init_session()

ASSETS = Path(__file__).parent.parent / "assets"

st.title("🗺️ Model Landscape — Legal Angle")
st.caption(":material/menu_book: Notebook: `05_model_landscape_legal_tour.ipynb`")
st.badge("Runs fully offline", icon=":material/wifi_off:", color="green")

st.image(str(ASSETS / "10-modern-model-landscape-slide14-choosing-a-model.png"), caption="Choosing a model", width="stretch")


def embed_text(tokenizer, model, text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    summed = (outputs.last_hidden_state * mask).sum(1)
    counted = mask.sum(1).clamp(min=1e-9)
    mean_pooled = summed / counted
    normed = mean_pooled / mean_pooled.norm(dim=1, keepdim=True)
    return normed[0].numpy()


st.header(":material/psychology: Part 1 — Legal-domain encoder: zero-shot clause classification", divider="gray")
st.caption("LegalBERT sentence embeddings, cosine similarity against a taxonomy of clause-type descriptions.")

legal_tok, legal_model = load_legalbert_model()

clause_choice = st.selectbox("Clause to classify", list(CLASSIFY_TARGET_CLAUSES.keys()))
custom_clause = st.text_area("...or paste your own clause (overrides the dropdown)")
target_text = custom_clause.strip() or CLASSIFY_TARGET_CLAUSES[clause_choice]
st.code(target_text, language=None)

if st.button("Classify clause", type="primary", icon=":material/label:"):
    with st.spinner("Embedding and scoring..."):
        target_emb = embed_text(legal_tok, legal_model, target_text)
        scores = {}
        for label, desc in CLAUSE_TAXONOMY.items():
            label_emb = embed_text(legal_tok, legal_model, desc)
            scores[label] = float(np.dot(target_emb, label_emb))

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    st.success(f"Predicted: **{ranked[0][0]}** (score {ranked[0][1]:.3f})")

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [l for l, _ in ranked]
    values = [s for _, s in ranked]
    ax.barh(labels[::-1], values[::-1], color="#55a868")
    ax.set_xlabel("Cosine similarity")
    st.pyplot(fig)
    plt.close(fig)

st.header(":material/smart_toy: Part 2 — Small language model: Qwen2.5-0.5B-Instruct", divider="gray")
st.caption(":material/download: ~1GB one-time download, cached after first load. Slower on CPU but fully workable.")

preset = st.selectbox("Preset prompt", list(QWEN_PROMPT_PRESETS.keys()))
custom_prompt = st.text_area("...or write your own prompt")
final_prompt = custom_prompt.strip() or QWEN_PROMPT_PRESETS[preset]

if st.button("Generate with Qwen 0.5B", icon=":material/auto_awesome:"):
    with st.spinner("Loading model (first run only) and generating..."):
        generator = load_qwen_pipeline()
        messages = [{"role": "user", "content": final_prompt}]
        output = generator(messages, max_new_tokens=150, do_sample=True, temperature=0.7)
    generated = output[0]["generated_text"]
    reply = generated[-1]["content"] if isinstance(generated, list) else generated
    st.write(reply)

st.header(":material/visibility: Part 3 — Vision-language models (conceptual)", divider="gray")
st.caption("Not executed locally — VLMs like Gemini 3 Pro, GPT-5 vision, and Qwen3-VL handle scanned contracts and redlines in production.")
st.markdown(
    """
```mermaid
flowchart LR
    A[Scanned contract / redline PDF] --> B[Vision-language model]
    B --> C[Layout + text understanding]
    C --> D[Structured extraction: parties, dates, clauses]
    D --> E[Downstream review / comparison tooling]
```
"""
)
