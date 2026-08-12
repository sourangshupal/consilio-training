<div align="center">

# ⚖️ Consilio — 4-Day AI Training

**Hands-on curriculum for teaching how modern AI actually works —
from attention mechanisms to agentic RAG to EU AI Act compliance.**

[![Days](https://img.shields.io/badge/days-4-6366F1?style=flat-square)](#-curriculum)
[![Notebooks](https://img.shields.io/badge/notebooks-20-8B5CF6?style=flat-square)](#-curriculum)
[![Runtime](https://img.shields.io/badge/runtime-Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](#-running-the-notebooks)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](#-day-1-streamlit-app)
[![Streamlit](https://img.shields.io/badge/app-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](#-day-1-streamlit-app)
[![Domain](https://img.shields.io/badge/domain-legal%20%2F%20CUAD%20%2F%20LEDGAR-0EA5E9?style=flat-square)](#-legal-domain-grounding)
[![License](https://img.shields.io/badge/license-proprietary-lightgrey?style=flat-square)](#-license--usage)

</div>

---

## 📖 About

This repo is the full working set behind **Consilio's 4-Day AI Training** — an 8-hour-a-day
program that takes a **law firm audience** from "what is a transformer" to "how do I build
and evaluate an agentic RAG pipeline on our contracts, safely and compliantly."

It is training content, not a software product — there's no build/lint/test pipeline to run.
Everything here is meant to be **opened, run, and taught from**: 20 runnable notebooks, one
interactive Streamlit app, and a full custom diagram library, all flavored with real legal
datasets instead of generic NLP examples.

---

## 🗂️ Repository structure

```
counsilio-training/
├── 📓 day1-notebooks/        AI Foundations                    — 7 notebooks
├── 📓 day2-notebooks/        Tools & LLM Finetuning             — 3 notebooks
├── 📓 day3-notebooks/        RAG & Agents                       — 6 notebooks
├── 📓 day4-notebooks/        AI Security & Legal Compliance     — 4 notebooks + 2 worksheets
├── 🖥️  day1-streamlit-app/    interactive multi-page app for Day 1
├── 🎨 diagrams/              custom HTML/PNG diagrams, one set per day
└── 🙈 .gitignore
```

---

## 📅 Curriculum

<table>
<tr><th>Day</th><th>Theme</th><th>#</th><th>Highlights</th></tr>

<tr valign="top">
<td><b>1</b></td>
<td>🧠 AI Foundations</td>
<td align="center">7</td>
<td>

Transformer attention visualization · BPE tokenization on legal text · DistilGPT2
finetuning on contract language · KV-cache speed benchmarking · modern model landscape
tour · legal prompt-engineering playground

</td>
</tr>

<tr valign="top">
<td><b>2</b></td>
<td>🛠️ Tools & LLM Finetuning</td>
<td align="center">3</td>
<td>

LoRA vs. QLoRA on legal text · quantization & inference-serving benchmarks (vLLM /
GGUF / llama.cpp) · Unsloth vs. TRL speed comparison.
<br><i>Cursor & Claude Code are taught live in-IDE — intentionally no notebook.</i>

</td>
</tr>

<tr valign="top">
<td><b>3</b></td>
<td>🔍 RAG & Agents</td>
<td align="center">6</td>
<td>

Vanilla RAG · hybrid retrieval + reranking · RAGAS eval & self-correcting (CRAG)
retrieval · ColPali OCR-free multimodal RAG · minimal LangGraph ReAct agent ·
multi-agent supervisor pattern

</td>
</tr>

<tr valign="top">
<td><b>4</b></td>
<td>🛡️ AI Security & Legal Compliance</td>
<td align="center">4 + 2</td>
<td>

PII / prompt-injection / toxicity guardrails · LiteLLM routing & fallback · RAGAS +
Langfuse continuous eval · fairness metrics mini-lab · **plus** EU AI Act risk
classification and GDPR-for-engineers worksheets (Markdown, legal/policy — no code)

</td>
</tr>
</table>

---

## ▶️ Running the notebooks

Every notebook is **Colab-first** — click, run top to bottom, done:

- 📦 First code cell installs every dependency (`%pip install -q ...`) — nothing needs
  to be preinstalled.
- 🖥️ A callout under the title states whether a **GPU runtime** is required.
- 🔑 API keys are pulled from **Colab Secrets** (falls back to environment variables) —
  never hard-coded.
- ⚡ GPU-heavy notebooks (finetuning, quantization, ColPali) are scoped to the free-tier
  **T4 GPU**; where live quantization is impractical on free Colab, a pre-quantized
  checkpoint is loaded instead.

To run locally instead of Colab, any standard Jupyter environment with the packages
from each notebook's first cell will work.

---

## 🔐 Secrets & API keys

Nothing is hard-coded anywhere in this repo — every key is read at runtime from
**Colab Secrets** (`google.colab.userdata`) or a **local environment variable** as a
fallback. Below is the full set used across the notebooks and the Streamlit app,
and exactly which notebooks need what.

| Secret | Required by | Get it from |
|---|---|---|
| `OPENAI_API_KEY` | Day 1 §06 Prompt Playground · Day 1 Streamlit app · Day 4 §02 LiteLLM · Day 4 §03 Langfuse eval | [platform.openai.com](https://platform.openai.com/api-keys) |
| `GEMINI_API_KEY` | Day 1 §06 Prompt Playground · Day 1 Streamlit app · Day 4 §02 LiteLLM | [aistudio.google.com](https://aistudio.google.com/apikey) |
| `GROQ_API_KEY` | Day 1 §06 Prompt Playground · Day 1 Streamlit app · Day 4 §02 LiteLLM | [console.groq.com](https://console.groq.com/keys) |
| `HF_TOKEN` | Gated/rate-limited Hugging Face model downloads (Day 2/3 finetuning & retrieval notebooks) | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `LANGFUSE_PUBLIC_KEY` | Day 4 §03 RAGAS + Langfuse continuous eval | [cloud.langfuse.com](https://cloud.langfuse.com) |
| `LANGFUSE_SECRET_KEY` | Day 4 §03 RAGAS + Langfuse continuous eval | [cloud.langfuse.com](https://cloud.langfuse.com) |

> **None of these are required to open or read any notebook.** Every notebook that
> needs one degrades gracefully to static/offline sections without it — only the
> live-API cells are gated.

**In Colab:** open the 🔑 key icon in the left sidebar → *Add new secret* → paste the
name/value above exactly → toggle *Notebook access* on. The notebook's key-loader cell
tries `userdata.get(...)` first automatically.

**Locally:** copy the template below into `.env` in the relevant app/notebook folder
(never commit it — `.gitignore` already excludes `.env*`):

```dotenv
# .env.example — copy to .env and fill in what you need, leave the rest blank
OPENAI_API_KEY=
GEMINI_API_KEY=
GROQ_API_KEY=
HF_TOKEN=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
```

---

## 🖥️ Day 1 Streamlit app

An interactive companion to `day1-notebooks/` — Transformers 101, Tokenization, KV
Cache, Model Landscape, and Prompt Engineering as one multi-page app. *(Finetuning is
intentionally left notebook-only.)*

```bash
cd day1-streamlit-app
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
uv run streamlit run app.py
```

Open the URL Streamlit prints — default `http://localhost:8501`.

| Tab | Runs offline? | Notes |
|---|:---:|---|
| Transformers 101 | ✅ | local DistilBERT |
| Tokenization | ✅ | local GPT-2 / LegalBERT |
| KV Cache | ✅ | local DistilGPT2 · Flash Attention compare needs CUDA, else static diagram |
| Model Landscape | ✅ | Qwen2.5-0.5B-Instruct, ~1GB first download — pre-load before class |
| Prompt Engineering | ⚠️ | needs an OpenAI / Gemini / Groq key in the sidebar for live sections; static explanations render without one |

All models cache after first load (`@st.cache_resource`) — no GPU required anywhere
in the app. See [`day1-streamlit-app/README.md`](day1-streamlit-app/README.md) for
full details.

---

## 🎨 Diagrams

`diagrams/day{1-4}/` holds every custom diagram built for the decks — one
self-contained `.html` source (inline SVG, no external deps) and one exported `.png`
per diagram, named:

```
NN-<deck-slug>-slideNN-<topic>.{html,png}
```

These cover the visuals that are more precise than AI slide-generation alone —
architecture diagrams, math derivations (LoRA, attention), decision trees, and
step-by-step mechanics (KV cache, PagedAttention, CRAG).

---

## 🔀 Multi-provider LLM pattern

Anything that calls a hosted LLM — Day 1's Prompt Engineering Playground, the Day 1
Streamlit app, Day 4's LiteLLM notebook — supports **OpenAI, Gemini, and Groq**
interchangeably through one `PROVIDER` variable and a unified `ask()` / router
function. Never locked to a single vendor; Day 4's LiteLLM notebook is the same
pattern generalized into a `Router` config.

---

## 🏛️ Legal-domain grounding

Legal-domain flavor is a **hard requirement** of this content, not decoration — every
notebook pulls from named, real legal datasets/models instead of generic NLP examples:

- **CUAD** — Contract Understanding Atticus Dataset
- **LexGLUE / LEDGAR** — legal clause classification
- **Pile of Law** (`atticus_contracts` subset)
- **`nlpaueb/legal-bert-base-uncased`**

Thin CSV topics (e.g. Tokenization, Tooling & Frameworks) are still expanded to the
same depth bar as richer topics rather than left shallow.

---


