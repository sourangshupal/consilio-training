from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
import torch

from src.kv_cache import sweep_prompt_lengths, timed_generate
from src.models import load_distilgpt2
from src.state import init_session

init_session()

ASSETS = Path(__file__).parent.parent / "assets"

st.title("⚡ KV Cache Speed Benchmark")
st.caption(":material/menu_book: Notebook: `04_kv_cache_speed_benchmark.ipynb`")
st.badge("Runs fully offline", icon=":material/wifi_off:", color="green")

col1, col2 = st.columns(2)
with col1:
    st.image(str(ASSETS / "08-finetuning-kvcache-slide11-kv-cache.png"), caption="KV cache mechanism", width="stretch")
with col2:
    st.image(str(ASSETS / "09-finetuning-kvcache-slide15-flash-attention.png"), caption="Flash attention", width="stretch")

DEFAULT_PROMPT = (
    "This Master Services Agreement (\"Agreement\") is entered into by and "
    "between the Client and the Service Provider. The Service Provider "
    "shall perform the services described in Exhibit A in a professional "
    "and workmanlike manner, consistent with generally accepted industry "
    "standards, and shall deliver all work product free of material "
    "defects. Payment terms are net thirty days from receipt of a valid "
    "invoice."
)

tokenizer, model = load_distilgpt2()

st.header(":material/compare: Single comparison: with cache vs. without", divider="gray")
prompt = st.text_area("Prompt", value=DEFAULT_PROMPT, height=120)
max_new_tokens = st.slider("New tokens to generate", 10, 100, 60)

if st.button("Run benchmark", type="primary", icon=":material/play_arrow:"):
    with st.spinner("Generating with cache..."):
        cached_time, cached_text = timed_generate(tokenizer, model, prompt, max_new_tokens, use_cache=True)
    with st.spinner("Generating without cache..."):
        uncached_time, _ = timed_generate(tokenizer, model, prompt, max_new_tokens, use_cache=False)

    m1, m2, m3 = st.columns(3)
    m1.metric("With cache", f"{cached_time:.2f}s")
    m2.metric("Without cache", f"{uncached_time:.2f}s")
    m3.metric("Speedup", f"{uncached_time / cached_time:.2f}x" if cached_time > 0 else "n/a")
    with st.expander("Generated text (with cache)"):
        st.write(cached_text)

st.header(":material/show_chart: Sweep: latency vs. prompt length", divider="gray")
st.caption("Repeats the prompt 1x-6x to grow context length, timing generation with/without cache at each length.")
if st.button("Run sweep", icon=":material/play_arrow:"):
    with st.spinner("Running sweep (this takes a little while)..."):
        results = sweep_prompt_lengths(tokenizer, model, prompt, repeats=list(range(1, 7)), max_new_tokens=30)

    st.dataframe(results, width="stretch", hide_index=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    lengths = [r["prompt_tokens"] for r in results]
    ax.plot(lengths, [r["cached_seconds"] for r in results], marker="o", label="With cache")
    ax.plot(lengths, [r["uncached_seconds"] for r in results], marker="o", label="Without cache")
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Seconds")
    ax.set_title("Generation latency vs. prompt length")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

st.header(":material/flash_on: Flash attention (GPU-only)", divider="gray")
if torch.cuda.is_available():
    st.success("CUDA detected — flash attention comparison would run here (see notebook 04, Section 7).", icon=":material/check_circle:")
else:
    st.warning("No CUDA GPU detected on this machine — flash attention only accelerates GPU inference.", icon=":material/warning:")
    st.markdown(
        """
```mermaid
flowchart LR
    A[Query, Key, Value in HBM] --> B[Tile Q/K/V into SRAM-sized blocks]
    B --> C[Compute attention per tile in fast SRAM]
    C --> D[Online softmax rescaling across tiles]
    D --> E[Write only final output back to HBM]
    E --> F[Avoids materializing full N x N attention matrix]
```
"""
    )
    st.caption("Flash attention avoids materializing the full attention matrix in HBM, tiling the computation instead — the speedup shows up on GPU, not CPU.")
