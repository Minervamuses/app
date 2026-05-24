"""Tests for graph skill state loading."""

from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from agent.config import AgentConfig
from agent.graph import build_graph


class _DummyModel:
    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return AIMessage(content="ok")


@tool("rag_explore")
def _rag_explore() -> str:
    """Explore."""
    return "explore"


@tool("rag_search")
def _rag_search(query: str) -> str:
    """Search."""
    return query


@tool("rag_get_context")
def _rag_get_context(pid: str, chunk_id: int) -> str:
    """Context."""
    return f"{pid}:{chunk_id}"


@tool("recall_history")
def _recall_history(query: str) -> str:
    """Recall."""
    return query


def _patch_graph_tools(monkeypatch):
    monkeypatch.setattr("agent.graph.get_chat_model", lambda _cfg: _DummyModel())
    monkeypatch.setattr(
        "agent.graph.create_rag_tools",
        lambda _cfg: [_rag_explore, _rag_search, _rag_get_context],
    )
    monkeypatch.setattr(
        "agent.graph.create_history_tool",
        lambda _cfg, store=None: _recall_history,
    )


def test_skill_loader_no_skill_is_noop(monkeypatch, tmp_path):
    _patch_graph_tools(monkeypatch)
    cfg = AgentConfig(persist_dir=str(tmp_path))
    graph = build_graph(cfg, skill_runtime_getter=lambda: None)

    result = graph.invoke({"messages": [HumanMessage(content="hi")]})

    assert "active_skill" not in result
    assert result["messages"][-1].content == "ok"


def test_skill_loader_populates_state_from_runtime(monkeypatch, tmp_path):
    _patch_graph_tools(monkeypatch)
    runtime = SimpleNamespace(
        name="paper-writing",
        root=Path(tmp_path / "skills" / "paper-writing"),
        instructions="# Skill",
        pinned_references={"references/guide.md": "guide"},
        task_mode="revision",
        allowed_tools=frozenset({"read_file"}),
        denied_tools=frozenset({"bash"}),
        tool_policy_active=True,
    )
    cfg = AgentConfig(persist_dir=str(tmp_path))
    graph = build_graph(cfg, skill_runtime_getter=lambda: runtime)

    result = graph.invoke({"messages": [HumanMessage(content="hi")]})

    assert result["active_skill"] == "paper-writing"
    assert result["skill_root"] == str(runtime.root)
    assert result["skill_instructions"] == "# Skill"
    assert result["loaded_references"] == {"references/guide.md": "guide"}
    assert result["allowed_tools"] == ["read_file"]
    assert result["denied_tools"] == ["bash"]
    assert result["tool_policy_active"] is True


def test_agent_node_binds_filtered_tools_for_active_skill(monkeypatch, tmp_path):
    bind_calls: list[list[str]] = []

    class RecordingModel:
        def bind_tools(self, tools):
            bind_calls.append([tool.name for tool in tools])

            class Bound:
                def invoke(self, _messages):
                    return AIMessage(content="ok")

            return Bound()

    monkeypatch.setattr("agent.graph.get_chat_model", lambda _cfg: RecordingModel())
    monkeypatch.setattr(
        "agent.graph.create_rag_tools",
        lambda _cfg: [_rag_explore, _rag_search, _rag_get_context],
    )
    monkeypatch.setattr(
        "agent.graph.create_history_tool",
        lambda _cfg, store=None: _recall_history,
    )
    cfg = AgentConfig(persist_dir=str(tmp_path))
    graph = build_graph(cfg)
    state = {
        "messages": [HumanMessage(content="hi")],
        "active_skill": "paper-writing",
        "task_mode": "revision",
        "allowed_tools": ["read_file"],
        "denied_tools": ["bash"],
        "tool_policy_active": True,
    }

    graph.invoke(state)
    graph.invoke(state)

    assert bind_calls[0] == [
        "rag_explore",
        "rag_search",
        "rag_get_context",
        "recall_history",
        "read_file",
        "bash",
    ]
    assert bind_calls[1] == ["read_file"]
    assert len(bind_calls) == 2


def test_agent_node_binds_all_except_denied_for_disallow_only_policy(
    monkeypatch,
    tmp_path,
):
    bind_calls: list[list[str]] = []

    class RecordingModel:
        def bind_tools(self, tools):
            bind_calls.append([tool.name for tool in tools])

            class Bound:
                def invoke(self, _messages):
                    return AIMessage(content="ok")

            return Bound()

    monkeypatch.setattr("agent.graph.get_chat_model", lambda _cfg: RecordingModel())
    monkeypatch.setattr(
        "agent.graph.create_rag_tools",
        lambda _cfg: [_rag_explore, _rag_search, _rag_get_context],
    )
    monkeypatch.setattr(
        "agent.graph.create_history_tool",
        lambda _cfg, store=None: _recall_history,
    )
    cfg = AgentConfig(persist_dir=str(tmp_path))
    graph = build_graph(cfg)

    graph.invoke({
        "messages": [HumanMessage(content="hi")],
        "active_skill": "paper-writing",
        "task_mode": "revision",
        "allowed_tools": [],
        "denied_tools": ["bash"],
        "tool_policy_active": True,
    })

    assert bind_calls[1] == [
        "rag_explore",
        "rag_search",
        "rag_get_context",
        "recall_history",
        "read_file",
    ]


def test_agent_node_binds_no_tools_for_active_empty_policy(monkeypatch, tmp_path):
    bind_calls: list[list[str]] = []

    class RecordingModel:
        def bind_tools(self, tools):
            bind_calls.append([tool.name for tool in tools])

            class Bound:
                def invoke(self, _messages):
                    return AIMessage(content="ok")

            return Bound()

    monkeypatch.setattr("agent.graph.get_chat_model", lambda _cfg: RecordingModel())
    monkeypatch.setattr(
        "agent.graph.create_rag_tools",
        lambda _cfg: [_rag_explore, _rag_search, _rag_get_context],
    )
    monkeypatch.setattr(
        "agent.graph.create_history_tool",
        lambda _cfg, store=None: _recall_history,
    )
    cfg = AgentConfig(persist_dir=str(tmp_path))
    graph = build_graph(cfg)

    graph.invoke({
        "messages": [HumanMessage(content="hi")],
        "active_skill": "paper-writing",
        "task_mode": "revision",
        "allowed_tools": [],
        "denied_tools": [],
        "tool_policy_active": True,
    })

    assert bind_calls[1] == []


def test_agent_node_forces_answer_after_tool_budget(monkeypatch, tmp_path):
    class BudgetModel:
        def __init__(self):
            self.bound_calls: list[list] = []
            self.raw_calls: list[list] = []

        def bind_tools(self, _tools):
            model = self

            class Bound:
                def invoke(self, messages):
                    model.bound_calls.append(messages)
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "rag_search",
                                "args": {"query": "x"},
                                "id": "call-1",
                            }
                        ],
                    )

            return Bound()

        def invoke(self, messages):
            self.raw_calls.append(messages)
            return AIMessage(content="final answer")

    model = BudgetModel()
    monkeypatch.setattr("agent.graph.get_chat_model", lambda _cfg: model)
    monkeypatch.setattr(
        "agent.graph.create_rag_tools",
        lambda _cfg: [_rag_explore, _rag_search, _rag_get_context],
    )
    monkeypatch.setattr(
        "agent.graph.create_history_tool",
        lambda _cfg, store=None: _recall_history,
    )
    cfg = AgentConfig(persist_dir=str(tmp_path), agent_max_tool_interactions=1)
    graph = build_graph(cfg)

    result = graph.invoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"recursion_limit": 8},
    )

    assert result["messages"][-1].content == "final answer"
    assert len(model.bound_calls) == 1
    assert len(model.raw_calls) == 1
    assert any(
        "Tool budget exhausted" in message.content
        for message in model.raw_calls[0]
    )
