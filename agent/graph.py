"""LangGraph agent graph for conversational RAG."""

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.config import AgentConfig

from agent.adapters.langchain import create_rag_tools
from agent.history_rag import create_history_tool
from agent.llm.openrouter import get_chat_model
from agent.history import prepare_messages_for_agent
from agent.state import AgentState


def build_graph(config: AgentConfig, extra_tools: list | None = None):
    """Build and compile the conversational RAG agent graph.

    Args:
        config: Agent configuration.
        extra_tools: Optional additional LangChain-compatible tools (e.g. MCP
            tools loaded at startup) appended after the local agent tools.

    Returns:
        A compiled LangGraph that accepts AgentState and manages
        the bounded agent ↔ tools loop for a single turn.
    """
    model = get_chat_model(config)
    tools = create_rag_tools(config)
    tools.append(create_history_tool(config))
    if extra_tools:
        tools = tools + list(extra_tools)
    model_with_tools = model.bind_tools(tools)

    def agent_node(state: AgentState):
        prompt_messages = prepare_messages_for_agent(
            state["messages"],
            max_messages=config.agent_max_messages,
            max_tool_interactions=config.agent_max_tool_interactions,
        )
        return {"messages": [model_with_tools.invoke(prompt_messages)]}

    def _tool_error_to_message(exc: Exception) -> str:
        return f"Tool error: {type(exc).__name__}: {exc}"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=_tool_error_to_message))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()
