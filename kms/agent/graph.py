"""LangGraph agent graph for conversational RAG."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from kms.agent.state import AgentState
from kms.config import KMSConfig
from kms.llm.openrouter import get_chat_model
from kms.tool.context import create_context_tool
from kms.tool.explore import create_explore_tool
from kms.tool.search import create_search_tool


def build_graph(config: KMSConfig):
    """Build and compile the conversational RAG agent graph.

    Args:
        config: KMS configuration.

    Returns:
        A compiled LangGraph that accepts AgentState and manages
        the agent ↔ tools loop with memory checkpointing.
    """
    model = get_chat_model(config)
    tools = [
        create_explore_tool(config),
        create_search_tool(config),
        create_context_tool(config),
    ]
    model_with_tools = model.bind_tools(tools)

    def agent_node(state: AgentState):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=MemorySaver())
