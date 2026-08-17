"""A minimal LangGraph ReAct agent with a clause-lookup tool."""

import sys
sys.path.insert(0, ".")

from pathlib import Path

import streamlit as st
from src.agents import build_react_app, react_trace, run_react
from src.rag_bank import CLAUSE_DATABASE
from src.state import active_api_key

ASSETS = Path(__file__).parent.parent / "assets"

st.title("🤖 LangGraph ReAct Agent")
st.caption(":material/menu_book: Notebook: `05_minimal_langgraph_react_agent.ipynb`")
st.badge("Needs API key", icon=":material/wifi:", color="gray")

col1, col2 = st.columns(2)
with col1:
    st.image(str(ASSETS / "10-agent-fundamentals-slide06-langgraph-core-concepts.png"), caption="LangGraph core concepts", width="stretch")
with col2:
    st.image(str(ASSETS / "12-agent-fundamentals-slide15-minimal-react-agent.png"), caption="Minimal ReAct agent loop", width="stretch")

st.markdown(
    "This agent has one tool — `lookup_clause` — over a small clause database. "
    "It decides on its own whether to call the tool or answer directly, and the "
    "loop keeps going until the model responds without a tool call."
)

provider = st.session_state["provider"]
api_key = active_api_key()

with st.expander(":material/database: Clause database (what the tool can look up)"):
    for topic, text in CLAUSE_DATABASE.items():
        st.markdown(f"**{topic}** — {text}")

st.header(":material/forum: Ask the agent", divider="gray")

example_questions = [
    "What does our contract say about termination?",
    "What is our liability cap under the agreement?",
    "What does the contract say about late payment penalties?",  # not in the DB — graceful "not found"
]
example_choice = st.selectbox("Example question", example_questions)
custom_question = st.text_input("...or type your own question")
question = custom_question.strip() or example_choice

if st.button("Run agent", type="primary", icon=":material/play_arrow:", disabled=not (question and api_key)):
    with st.spinner("Running agent..."):
        app = build_react_app(provider, api_key)
        result = run_react(app, question)
        trace = react_trace(result)

    st.subheader(":material/route: ReAct trace")
    icons = {"thought": "💭", "action": "🔧", "observation": "👁️", "final": "✅"}
    labels = {"thought": "Thought", "action": "Action", "observation": "Observation", "final": "Final Answer"}
    for step in trace:
        st.markdown(f"{icons[step['type']]} **{labels[step['type']]}:** {step['text']}")
elif not api_key:
    st.info(f"Enter your {provider} API key in the sidebar to run the agent.", icon=":material/key:")
