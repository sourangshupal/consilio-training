import streamlit as st

DEFAULTS = {
    "provider": "OpenAI",
    "openai_api_key": "",
    "gemini_api_key": "",
    "groq_api_key": "",
}


def init_session() -> None:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def active_api_key() -> str:
    provider = st.session_state.get("provider", "OpenAI")
    return {
        "OpenAI": st.session_state.get("openai_api_key", ""),
        "Gemini": st.session_state.get("gemini_api_key", ""),
        "Groq": st.session_state.get("groq_api_key", ""),
    }.get(provider, "")
