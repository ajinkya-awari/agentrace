"""LangGraph medical QA agent definition with ToolNode callback propagation."""

from typing import Annotated, Any, TypedDict

try:
    from langchain_core.tools import tool
except ImportError:  # Static-only local fallback; notebook installs the real dependency.
    def tool(function):  # type: ignore[no-redef]
        return function


try:
    from langchain_core.messages import AnyMessage
except ImportError:
    AnyMessage = Any  # type: ignore[misc,assignment]

try:
    from langgraph.graph import add_messages
except ImportError:  # Static-only fallback; notebook installs LangGraph.
    def add_messages(messages):  # type: ignore[no-redef]
        return messages


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


TRACER_SMOKE_SYSTEM_PROMPT = (
    "You are executing a deterministic tracer-smoke check. "
    "Return a valid JSON object only, exactly {\"answer\": \"A\"} with one label from A, B, C, or D."
)
TRACER_SMOKE_USER_PROMPT = (
    "Tracer-smoke response contract: return valid JSON only in the form "
    "{\"answer\": \"A\"}, selecting one label from A, B, C, or D."
)


@tool
def medical_lookup(query: str) -> str:
    """Return a static reference notice without fetching clinical data."""
    return "Reference lookup unavailable in MVP; answer from provided MCQ text only."


def build_medical_agent(llm: Any):
    """Build the graph lazily so importing the module never creates a client or graph."""
    from langchain_core.messages import SystemMessage
    from langgraph.graph import END, START, StateGraph, add_messages
    from langgraph.prebuilt import ToolNode, tools_condition

    class GraphState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    tools = [medical_lookup]
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: GraphState):
        system = SystemMessage(
            content=(
                "You are a cautious medical QA assistant. Use tools only when useful. "
                "For MCQs, reply with only A, B, C, or D."
            )
        )
        response = llm_with_tools.invoke([system] + state["messages"])
        return {"messages": [response]}

    graph = StateGraph(GraphState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def build_tracer_smoke_agent(llm: Any):
    """Build a smoke-only graph with two inert ToolNode calls and one model call."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.tools import tool
    from langgraph.graph import END, START, StateGraph, add_messages
    from langgraph.prebuilt import ToolNode

    @tool
    def synthetic_tracer_smoke_tool(marker: str) -> str:
        """Return a fixed marker without using a network, dataset, or clinical content."""
        if marker not in {"first", "second"}:
            raise ValueError("unexpected tracer smoke marker")
        return "synthetic tracer smoke tool completed"

    class SmokeState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    def seed_tool_calls(state: SmokeState):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "synthetic_tracer_smoke_tool", "args": {"marker": "first"}, "id": "smoke-tool-1"},
                        {"name": "synthetic_tracer_smoke_tool", "args": {"marker": "second"}, "id": "smoke-tool-2"},
                    ],
                )
            ]
        }

    def call_model(state: SmokeState):
        system = SystemMessage(content=TRACER_SMOKE_SYSTEM_PROMPT)
        contract_message = HumanMessage(content=TRACER_SMOKE_USER_PROMPT)
        return {"messages": [llm.invoke([system] + state["messages"] + [contract_message])]}

    graph = StateGraph(SmokeState)
    graph.add_node("seed_tool_calls", seed_tool_calls)
    graph.add_node("tools", ToolNode([synthetic_tracer_smoke_tool]))
    graph.add_node("agent", call_model)
    graph.add_edge(START, "seed_tool_calls")
    graph.add_edge("seed_tool_calls", "tools")
    graph.add_edge("tools", "agent")
    graph.add_edge("agent", END)
    return graph.compile()
