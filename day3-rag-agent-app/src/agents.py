"""LangGraph agent graphs: a minimal ReAct agent with a clause-lookup tool,
and a multi-agent supervisor pattern (research -> risk analysis -> synthesis)
with an optional human-in-the-loop checkpoint. Ported from day3 notebooks
05 and 06, generalized to work with any of the three LLM providers via
src/llm_router.get_langchain_chat_model().
"""

from typing import Annotated, Literal, TypedDict

from src.llm_router import get_langchain_chat_model
from src.rag_bank import CLAUSE_DATABASE


def _lookup_clause(topic: str) -> str:
    topic_key = topic.lower().strip()
    for key, clause_text in CLAUSE_DATABASE.items():
        if key in topic_key or topic_key in key:
            return clause_text
    return f"No clause found for topic: {topic!r}. Available topics: {list(CLAUSE_DATABASE.keys())}"


def _build_lookup_tool():
    from langchain_core.tools import tool

    @tool
    def lookup_clause(topic: str) -> str:
        """Look up a contract clause by topic (e.g. 'indemnification', 'termination', 'liability', 'governing law')."""
        return _lookup_clause(topic)

    return lookup_clause


def build_react_app(provider: str, api_key: str):
    """Compiles the agent<->tools loop from notebook 05: agent_node calls the
    LLM (bound to lookup_clause), should_continue routes to tools or END."""
    from langgraph.graph import END, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode

    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    tool_fn = _build_lookup_tool()
    tools = [tool_fn]
    llm = get_langchain_chat_model(provider, api_key, temperature=0.0)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        return "tools" if getattr(last_message, "tool_calls", None) else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


def run_react(app, question: str):
    from langchain_core.messages import HumanMessage

    return app.invoke({"messages": [HumanMessage(content=question)]})


def react_trace(result) -> list[dict]:
    """Renders the LangGraph message trace as Thought/Action/Observation/
    Final-Answer steps, matching notebook 05's run_and_print_react_trace()."""
    steps = []
    for msg in result["messages"][1:]:  # skip the initial human question
        cls_name = msg.__class__.__name__
        if cls_name == "AIMessage" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                topic = tc["args"].get("topic", "?")
                steps.append({"type": "thought", "text": f"I should look up the '{topic}' clause."})
                steps.append({"type": "action", "text": f"{tc['name']}({tc['args']})"})
        elif cls_name == "ToolMessage":
            steps.append({"type": "observation", "text": msg.content})
        elif cls_name == "AIMessage" and msg.content:
            steps.append({"type": "final", "text": msg.content})
    return steps


class SupervisorState(TypedDict):
    task: str
    clause_findings: str
    risk_findings: str
    next_worker: str
    final_output: str


def initial_supervisor_state(task: str) -> SupervisorState:
    return {"task": task, "clause_findings": "", "risk_findings": "", "next_worker": "", "final_output": ""}


def build_supervisor_graph(provider: str, api_key: str):
    """Returns an uncompiled StateGraph so the caller can compile it plain
    (for a normal full run) or with a checkpointer + interrupt_before for the
    human-in-the-loop demo (notebook 06)."""
    from langgraph.graph import END, StateGraph
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_langchain_chat_model(provider, api_key, temperature=0.0)

    def clause_research_worker(state: SupervisorState) -> SupervisorState:
        response = llm.invoke([
            SystemMessage(content="You are a legal research specialist. Identify which standard contract clauses are relevant to the task, briefly."),
            HumanMessage(content=state["task"]),
        ])
        return {**state, "clause_findings": response.content}

    def risk_analysis_worker(state: SupervisorState) -> SupervisorState:
        response = llm.invoke([
            SystemMessage(content="You are a contract risk analyst. Identify the top risk factors relevant to the task, briefly."),
            HumanMessage(content=state["task"]),
        ])
        return {**state, "risk_findings": response.content}

    def supervisor_node(state: SupervisorState) -> SupervisorState:
        if not state.get("clause_findings"):
            return {**state, "next_worker": "clause_research"}
        if not state.get("risk_findings"):
            return {**state, "next_worker": "risk_analysis"}
        return {**state, "next_worker": "done"}

    def route_from_supervisor(state: SupervisorState) -> Literal["clause_research", "risk_analysis", "synthesize"]:
        if state["next_worker"] == "clause_research":
            return "clause_research"
        elif state["next_worker"] == "risk_analysis":
            return "risk_analysis"
        return "synthesize"

    def synthesize_node(state: SupervisorState) -> SupervisorState:
        response = llm.invoke([
            SystemMessage(content="Combine the following findings into one concise summary for a legal associate."),
            HumanMessage(content=f"Clause findings:\n{state['clause_findings']}\n\nRisk findings:\n{state['risk_findings']}"),
        ])
        return {**state, "final_output": response.content}

    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("clause_research", clause_research_worker)
    graph.add_node("risk_analysis", risk_analysis_worker)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {"clause_research": "clause_research", "risk_analysis": "risk_analysis", "synthesize": "synthesize"},
    )
    graph.add_edge("clause_research", "supervisor")
    graph.add_edge("risk_analysis", "supervisor")
    graph.add_edge("synthesize", END)

    return graph
