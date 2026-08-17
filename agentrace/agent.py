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
