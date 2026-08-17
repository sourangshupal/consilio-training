"""Multi-agent supervisor pattern: research + risk-analysis workers, with an
optional human-in-the-loop checkpoint before synthesis."""

import sys
sys.path.insert(0, ".")

import uuid
from pathlib import Path

import streamlit as st
from src.agents import build_supervisor_graph, initial_supervisor_state
from src.rag_bank import SUPERVISOR_TASKS
from src.state import active_api_key

ASSETS = Path(__file__).parent.parent / "assets"

st.title("👥 Multi-Agent Supervisor")
st.caption(":material/menu_book: Notebook: `06_multi_agent_supervisor_pattern.ipynb`")
st.badge("Needs API key", icon=":material/wifi:", color="gray")

col1, col2 = st.columns(2)
with col1:
    st.image(str(ASSETS / "13-advanced-agents-slide05-orchestrator-worker.png"), caption="Orchestrator-worker pattern", width="stretch")
with col2:
    st.image(str(ASSETS / "11-agent-fundamentals-slide10-edges-and-routing.png"), caption="Edges and routing", width="stretch")

st.markdown(
    "A supervisor routes a task through two specialist workers — **clause research** "
    "and **risk analysis** — then a synthesis step combines both findings. This demo "
    "also shows a human-in-the-loop checkpoint that pauses execution right before "
    "synthesis so a reviewer can inspect both workers' output first."
)

provider = st.session_state["provider"]
api_key = active_api_key()

example_choice = st.selectbox("Example task", SUPERVISOR_TASKS)
custom_task = st.text_area("...or describe your own task")
task = custom_task.strip() or example_choice

st.header(":material/play_circle: Run the supervisor", divider="gray")

hitl = st.toggle("Pause before synthesis (human-in-the-loop)", value=True)

if st.button("Start run", type="primary", icon=":material/play_arrow:", disabled=not (task and api_key)):
    graph = build_supervisor_graph(provider, api_key)
    initial_state = initial_supervisor_state(task)

    if hitl:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
        app = graph.compile(checkpointer=checkpointer, interrupt_before=["synthesize"])
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        with st.spinner("Running research + risk analysis workers..."):
            interrupted_state = app.invoke(initial_state, config=config)

        st.session_state["supervisor_app"] = app
        st.session_state["supervisor_config"] = config
        st.session_state["supervisor_interrupted_state"] = interrupted_state
        st.session_state["supervisor_final"] = None
    else:
        app = graph.compile()
        with st.spinner("Running full pipeline..."):
            result = app.invoke(initial_state)
        st.session_state["supervisor_app"] = None
        st.session_state["supervisor_interrupted_state"] = result
        st.session_state["supervisor_final"] = result

if st.session_state.get("supervisor_interrupted_state"):
    state = st.session_state["supervisor_interrupted_state"]

    st.subheader(":material/gavel: Clause research worker output")
    st.markdown(state["clause_findings"])

    st.subheader(":material/warning: Risk analysis worker output")
    st.markdown(state["risk_findings"])

    if st.session_state.get("supervisor_app") is not None and not st.session_state.get("supervisor_final"):
        st.info("Execution paused before synthesis — review both outputs above, then resume.", icon=":material/pause_circle:")
        if st.button("Resume synthesis", icon=":material/play_arrow:"):
            app = st.session_state["supervisor_app"]
            config = st.session_state["supervisor_config"]
            with st.spinner("Resuming and synthesizing..."):
                final_result = app.invoke(None, config=config)
            st.session_state["supervisor_final"] = final_result

    final = st.session_state.get("supervisor_final")
    if final:
        st.subheader(":material/summarize: Synthesized final output")
        st.markdown(final["final_output"])
elif not api_key:
    st.info(f"Enter your {provider} API key in the sidebar to run the supervisor.", icon=":material/key:")
