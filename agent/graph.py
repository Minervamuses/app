"""LangGraph agent graph for conversational RAG."""

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition

from agent.config import AgentConfig

from agent.adapters.langchain import create_rag_tools
from agent.history_rag import create_history_tool
from agent.llm.openrouter import get_chat_model
from agent.history import prepare_messages_for_agent
from agent.policy_tool_node import PolicyToolNode
from agent.state import AgentState
from agent.tools import create_bash_tool, create_read_file_tool


def _skill_runtime_state(runtime) -> dict:
    if runtime is None:
        return {}
    return {
        "active_skill": runtime.name,
        "skill_root": str(runtime.root),
        "skill_instructions": runtime.instructions,
        "loaded_references": dict(runtime.pinned_references),
        "task_mode": runtime.task_mode,
        "allowed_tools": sorted(runtime.allowed_tools),
        "denied_tools": sorted(runtime.denied_tools),
        "validation_errors": [],
        "validation_attempts": 0,
    }


def build_graph(
    config: AgentConfig,
    extra_tools: list | None = None,
    history_store=None,
    skill_runtime_getter=None,
):
    """Build and compile the conversational RAG agent graph.

    Args:
        config: Agent configuration.
        extra_tools: Optional additional LangChain-compatible tools (e.g. MCP
            tools loaded at startup) appended after the local agent tools.
        history_store: Optional store injected into the recall_history tool.
        skill_runtime_getter: Optional callable returning the active SkillRuntime.

    Returns:
        A compiled LangGraph that accepts AgentState and manages
        the bounded agent ↔ tools loop for a single turn.
    """
    model = get_chat_model(config)
    tools = create_rag_tools(config)
    tools.append(create_history_tool(config, store=history_store))
    tools.append(create_read_file_tool(config))
    tools.append(create_bash_tool(config))
    if extra_tools:
        tools = tools + list(extra_tools)
    tools_by_name = {getattr(tool, "name", str(tool)): tool for tool in tools}
    tool_order = [getattr(tool, "name", str(tool)) for tool in tools]
    bound_model_cache = {
        (None, None, (), ()): model.bind_tools(tools),
    }

    def _select_tools(state: AgentState) -> list:
        allowed = set(state.get("allowed_tools") or [])
        denied = set(state.get("denied_tools") or [])
        if not allowed and not denied:
            return tools
        selected_names = [
            name
            for name in tool_order
            if (not allowed or name in allowed) and name not in denied
        ]
        return [tools_by_name[name] for name in selected_names]

    def _model_for_state(state: AgentState):
        allowed = tuple(sorted(state.get("allowed_tools") or []))
        denied = tuple(sorted(state.get("denied_tools") or []))
        key = (
            state.get("active_skill"),
            state.get("task_mode"),
            allowed,
            denied,
        )
        if key not in bound_model_cache:
            bound_model_cache[key] = model.bind_tools(_select_tools(state))
        return bound_model_cache[key]

    def agent_node(state: AgentState):
        prompt_messages = prepare_messages_for_agent(
            state["messages"],
            max_messages=config.agent_max_messages,
            max_tool_interactions=config.agent_max_tool_interactions,
        )
        return {"messages": [_model_for_state(state).invoke(prompt_messages)]}

    def _tool_error_to_message(exc: Exception) -> str:
        return f"Tool error: {type(exc).__name__}: {exc}"

    def skill_loader_node(state: AgentState):
        if state.get("skill_instructions"):
            return {}
        if skill_runtime_getter is None:
            return {}
        return _skill_runtime_state(skill_runtime_getter())

    graph = StateGraph(AgentState)
    graph.add_node("skill_loader", skill_loader_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", PolicyToolNode(tools, handle_tool_errors=_tool_error_to_message))

    graph.add_edge(START, "skill_loader")
    graph.add_edge("skill_loader", "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()
