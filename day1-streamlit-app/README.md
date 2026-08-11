# Day 1 — AI Foundations (Streamlit)

Interactive companion to `day1-notebooks/`, covering Transformers 101, Tokenization,
KV Cache, Model Landscape, and Prompt Engineering as a single multi-page Streamlit app.
(Finetuning is intentionally not included in this app.)

## Setup

```bash
cd day1-streamlit-app
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

## Run

```bash
uv run streamlit run app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`).

## Notes

- All tabs except **Prompt Engineering** run fully offline on local models
  (DistilBERT, GPT-2, LegalBERT, DistilGPT2, Qwen2.5-0.5B). Models download once
  on first use and are cached for the rest of the session (`@st.cache_resource`).
- **Model Landscape**'s Qwen2.5-0.5B-Instruct is the heaviest download (~1GB) —
  consider pre-loading it before class by visiting that tab once ahead of time.
- **Prompt Engineering** needs an API key (OpenAI, Gemini, or Groq) entered in the
  sidebar to run live sections; without a key, static explanations still render.
- No GPU required. Flash Attention comparison (KV Cache tab) only runs on CUDA and
  shows an explainer diagram otherwise.
