import streamlit as st

from src.state import init_session

st.set_page_config(
    page_title="Day 1 — AI Foundations",
    page_icon="⚖️",
    layout="wide",
)

init_session()


def home():
    st.title("⚖️ Day 1 — AI Foundations")
    st.caption("Consilio 4-Day AI Training — interactive companion to the day1 notebooks")
    st.space("small")

    modules = [
        (":material/hub:", "Transformers 101", "Self-attention heatmaps on real contract clauses.", "Offline", "green"),
        (":material/split_scene:", "Tokenization", "BPE from scratch + general vs. legal tokenizer shootout.", "Offline", "green"),
        (":material/bolt:", "KV cache", "Measured generation speedup from caching, live on your machine.", "Offline", "green"),
        (":material/travel_explore:", "Model landscape", "Legal-domain encoder, small language model, and VLM discussion.", "Offline", "green"),
        (":material/forum:", "Prompt engineering", "Every major prompting technique, applied to legal fact patterns.", "Needs API key", "gray"),
    ]

    cols = st.columns(3)
    for i, (icon, name, desc, badge_label, badge_color) in enumerate(modules):
        with cols[i % 3].container(border=True, height="stretch"):
            st.markdown(f"{icon} **{name}**")
            st.caption(desc)
            st.badge(badge_label, color=badge_color)

    st.space("medium")
    st.info(
        "Pick a page from the sidebar to get started. Models download once on first "
        "use and stay cached for the rest of the session.",
        icon=":material/rocket_launch:",
    )


pages = [
    st.Page(home, title="Home", icon=":material/home:", default=True, url_path="home"),
    st.Page("pages/01_transformers_101.py", title="Transformers 101", icon="🧠", url_path="transformers-101"),
    st.Page("pages/02_tokenization.py", title="Tokenization", icon="🔤", url_path="tokenization"),
    st.Page("pages/03_kv_cache.py", title="KV cache", icon="⚡", url_path="kv-cache"),
    st.Page("pages/04_model_landscape.py", title="Model landscape", icon="🗺️", url_path="model-landscape"),
    st.Page("pages/05_prompt_engineering.py", title="Prompt engineering", icon="✍️", url_path="prompt-engineering"),
]

with st.sidebar:
    st.markdown("### ⚖️ Day 1 — AI Foundations")
    st.caption("Consilio 4-Day AI Training")

    with st.container(border=True):
        st.markdown(":material/key: **LLM provider**")
        st.caption("Only needed for the live sections of the Prompt Engineering tab.")
        st.selectbox("Provider", ["OpenAI", "Gemini", "Groq"], key="provider", label_visibility="collapsed")
        st.text_input("OpenAI API key", type="password", key="openai_api_key", icon=":material/lock:")
        st.text_input("Gemini API key", type="password", key="gemini_api_key", icon=":material/lock:")
        st.text_input("Groq API key", type="password", key="groq_api_key", icon=":material/lock:")

        active_key = {
            "OpenAI": st.session_state["openai_api_key"],
            "Gemini": st.session_state["gemini_api_key"],
            "Groq": st.session_state["groq_api_key"],
        }[st.session_state["provider"]]
        if active_key:
            st.badge(f"{st.session_state['provider']} key set", icon=":material/check_circle:", color="green")
        else:
            st.badge("No key set", icon=":material/info:", color="gray")

    st.space("small")
    st.markdown(":green-badge[:material/wifi_off: Offline] :gray-badge[:material/wifi: Needs API key]")
    st.caption("Every tab except Prompt Engineering runs fully offline on local models.")

nav = st.navigation(pages)
nav.run()
