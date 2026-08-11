from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from src.bpe import run_bpe_merges
from src.clause_bank import BPE_TOY_CORPUS, COMPARISON_CLAUSES, SHOOTOUT_TERMS
from src.models import load_gpt2_tokenizer, load_legalbert_tokenizer
from src.state import init_session

init_session()

ASSETS = Path(__file__).parent.parent / "assets"

st.title("🔤 Tokenization — BPE From Scratch + Legal Tokenizer Shootout")
st.caption(":material/menu_book: Notebook: `02_tokenization_bpe_legal_shootout.ipynb`")
st.badge("Runs fully offline", icon=":material/wifi_off:", color="green")

col1, col2 = st.columns(2)
with col1:
    st.image(str(ASSETS / "06-tokenization-slide03-taxonomy.png"), caption="Tokenization taxonomy", width="stretch")
with col2:
    st.image(str(ASSETS / "07-tokenization-slide09-bpe-merge-trace.png"), caption="BPE merge trace", width="stretch")

st.header(":material/join_inner: Part A — BPE from scratch", divider="gray")
st.caption("Toy legal-word corpus: " + ", ".join(f"{w} ({f})" for w, f in BPE_TOY_CORPUS.items()))
num_merges = st.slider("Number of merges", 1, 15, 6)
steps = run_bpe_merges(BPE_TOY_CORPUS, num_merges)

for s in steps:
    a, b = s["pair"]
    st.markdown(f"**Step {s['step']}:** merge `{a}` + `{b}` → `{s['merged_symbol']}`  (count: {s['count']})")

if steps:
    with st.expander("Final vocabulary after merges"):
        st.json(steps[-1]["vocab_snapshot"])

st.header(":material/compare_arrows: Part B — General-purpose vs. legal tokenizer shootout", divider="gray")

gpt2_tok = load_gpt2_tokenizer()
legal_tok = load_legalbert_tokenizer()

st.caption(f"GPT-2 vocab size: {gpt2_tok.vocab_size:,} | LegalBERT vocab size: {legal_tok.vocab_size:,}")

rows = []
for term in SHOOTOUT_TERMS:
    gpt2_ids = gpt2_tok.tokenize(term)
    legal_ids = legal_tok.tokenize(term)
    rows.append({
        "Term": term,
        "GPT-2 tokens": len(gpt2_ids),
        "GPT-2 split": " | ".join(gpt2_ids),
        "LegalBERT tokens": len(legal_ids),
        "LegalBERT split": " | ".join(legal_ids),
    })
st.dataframe(rows, width="stretch", hide_index=True)

st.subheader(":material/description: Full-clause comparison")
clause_choice = st.selectbox("Clause", list(COMPARISON_CLAUSES.keys()))
clause_text = COMPARISON_CLAUSES[clause_choice]
st.code(clause_text, language=None)

gpt2_clause = gpt2_tok.tokenize(clause_text)
legal_clause = legal_tok.tokenize(clause_text)
n_words = len(clause_text.split())

fig, ax = plt.subplots(figsize=(5, 3))
names = ["GPT-2", "LegalBERT"]
ratios = [len(gpt2_clause) / n_words, len(legal_clause) / n_words]
ax.bar(names, ratios, color=["#4c72b0", "#55a868"])
ax.set_ylabel("Tokens per word")
ax.set_title("Tokenization efficiency")
st.pyplot(fig)
plt.close(fig)

st.subheader(":material/edit_note: Try your own clause")
custom_clause = st.text_area("Paste a clause", value="")
if custom_clause.strip():
    c1, c2 = st.columns(2)
    with c1:
        toks = gpt2_tok.tokenize(custom_clause)
        st.markdown(f"**GPT-2** — {len(toks)} tokens")
        st.code(" | ".join(toks), language=None)
    with c2:
        toks = legal_tok.tokenize(custom_clause)
        st.markdown(f"**LegalBERT** — {len(toks)} tokens")
        st.code(" | ".join(toks), language=None)
