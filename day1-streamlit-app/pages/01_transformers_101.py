from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from src.attention import (
    avg_last_layer_attention_to_token,
    causal_mask_demo,
    get_attentions,
    layer_head_matrix,
    positional_encoding,
)
from src.clause_bank import ATTENTION_SENTENCES
from src.models import load_distilbert
from src.state import init_session

init_session()

ASSETS = Path(__file__).parent.parent / "assets"

st.title("🧠 Transformers 101 — Attention on Contract Language")
st.caption(":material/menu_book: Notebook: `01_transformers101_attention_visualization.ipynb`")
st.badge("Runs fully offline", icon=":material/wifi_off:", color="green")

st.header(":material/hub: How a transformer reads a clause")
diagram_cols = st.columns(5)
diagrams = [
    ("01-transformers101-slide03-encoder-decoder.png", "Encoder-decoder"),
    ("02-transformers101-slide06-qkv-attention.png", "Q, K, V attention"),
    ("03-transformers101-slide11-multihead-attention.png", "Multi-head attention"),
    ("04-transformers101-slide14-masked-attention.png", "Masked (causal) attention"),
    ("05-transformers101-slide16-positional-encoding.png", "Positional encoding"),
]
for col, (fname, caption) in zip(diagram_cols, diagrams):
    with col:
        st.image(str(ASSETS / fname), caption=caption, width="stretch")

st.header(":material/grid_view: Live attention heatmap", divider="gray")

tokenizer, model = load_distilbert()

col1, col2 = st.columns([2, 1])
with col1:
    choice = st.selectbox("Legal clause", list(ATTENTION_SENTENCES.keys()))
    custom = st.text_input("...or type your own sentence (overrides the dropdown)")
sentence = custom.strip() or ATTENTION_SENTENCES[choice]
st.code(sentence, language=None)

tokens, attentions = get_attentions(tokenizer, model, sentence)
num_layers = len(attentions)
num_heads = attentions[0].shape[1]

with col2:
    layer = st.slider("Layer", 0, num_layers - 1, min(5, num_layers - 1))
    head = st.slider("Head", 0, num_heads - 1, 0)

matrix = layer_head_matrix(attentions, layer, head)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(matrix, cmap="viridis")
ax.set_xticks(range(len(tokens)))
ax.set_yticks(range(len(tokens)))
ax.set_xticklabels(tokens, rotation=90, fontsize=8)
ax.set_yticklabels(tokens, fontsize=8)
ax.set_title(f"Layer {layer}, Head {head}")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
st.pyplot(fig)
plt.close(fig)

st.subheader(":material/target: Where does a token attend? (averaged over all heads, last layer)")
target = st.selectbox("Target token", tokens, index=min(3, len(tokens) - 1))
ranked = avg_last_layer_attention_to_token(tokens, attentions, target)
if ranked:
    top = ranked[:8]
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.barh([t for t, _ in top][::-1], [s for _, s in top][::-1], color="#4c72b0")
    ax2.set_xlabel("Attention weight")
    ax2.set_title(f"Tokens attending most to '{target}'")
    st.pyplot(fig2)
    plt.close(fig2)

st.subheader(":material/apps: Multi-head comparison (4 heads, last layer)")
grid_heads = [h for h in (0, 3, 6, 9) if h < num_heads] or list(range(min(4, num_heads)))
fig3, axes = plt.subplots(2, 2, figsize=(9, 8))
for ax, h in zip(axes.flat, grid_heads):
    m = layer_head_matrix(attentions, num_layers - 1, h)
    ax.imshow(m, cmap="viridis")
    ax.set_title(f"Head {h}", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
st.pyplot(fig3)
plt.close(fig3)

st.header(":material/visibility_off: Toy causal-masking demo", divider="gray")
st.caption("Hand-rolled softmax on a 4-token toy sequence, showing why a decoder can't look ahead.")
toy_tokens = ["Party", "shall", "indemnify", "Client"]
raw, masked = causal_mask_demo(toy_tokens)
fig4, (ax_raw, ax_masked) = plt.subplots(1, 2, figsize=(9, 4))
for ax, mat, title in [(ax_raw, raw, "Raw scores"), (ax_masked, masked, "Causal-masked softmax")]:
    ax.imshow(mat, cmap="viridis")
    ax.set_xticks(range(len(toy_tokens)))
    ax.set_yticks(range(len(toy_tokens)))
    ax.set_xticklabels(toy_tokens, rotation=45, fontsize=8)
    ax.set_yticklabels(toy_tokens, fontsize=8)
    ax.set_title(title)
st.pyplot(fig4)
plt.close(fig4)

st.header(":material/waves: Positional encoding", divider="gray")
seq_len = st.slider("Sequence length", 10, 100, 50)
dim = st.select_slider("Embedding dimensions", options=[16, 32, 64, 128], value=64)
pe = positional_encoding(seq_len, dim)
fig5, ax5 = plt.subplots(figsize=(9, 4))
im5 = ax5.imshow(pe.T, cmap="RdBu", aspect="auto")
ax5.set_xlabel("Position")
ax5.set_ylabel("Dimension")
fig5.colorbar(im5, ax=ax5, fraction=0.02, pad=0.02)
st.pyplot(fig5)
plt.close(fig5)
