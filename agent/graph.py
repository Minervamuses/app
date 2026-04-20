"""LangGraph agent graph for conversational RAG."""

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.config import AgentConfig

from agent.adapters.langchain import create_context_tool, create_explore_tool, create_search_tool
from agent.llm.openrouter import get_chat_model
from agent.history import prepare_messages_for_agent
from agent.state import AgentState


def build_graph(config: AgentConfig, extra_tools: list | None = None):
    """Build and compile the conversational RAG agent graph.

    Args:
        config: KMS configuration.
        extra_tools: Optional additional LangChain-compatible tools (e.g. MCP
            tools loaded at startup) appended after the three local KB tools.

    Returns:
        A compiled LangGraph that accepts AgentState and manages
        the bounded agent ↔ tools loop for a single turn.
    """
    model = get_chat_model(config)
    tools = [
        create_explore_tool(config),
        create_search_tool(config),
        create_context_tool(config),
    ]
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

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()
