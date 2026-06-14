"""Regression tests for academic-paper-writing tool policy."""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from agent.config import AgentConfig
from agent.graph import build_graph
from agent.skills.runtime import load_skill_runtime


APP_ROOT = Path(__file__).resolve().parents[1]


@tool("rag_explore")
def _rag_explore() -> str:
    """Explore the indexed knowledge base."""
    return "explore"


@tool("rag_search")
def _rag_search(query: str) -> str:
    """Search the indexed knowledge base."""
    return query


@tool("rag_get_context")
def _rag_get_context(pid: str, chunk_id: int) -> str:
    """Expand one search hit."""
    return f"{pid}:{chunk_id}"


@tool("recall_history")
def _recall_history(query: str) -> str:
    """Search persisted chat history."""
    return query


@tool("read_file")
def _read_file(path: str) -> str:
    """Read a text file."""
    return path


@tool("bash")
def _bash(command: str) -> str:
    """Run a shell command."""
    return command


def _cfg(tmp_path) -> AgentConfig:
    return AgentConfig(
        persist_dir=str(tmp_path / "persist"),
        skills_dir=str(APP_ROOT / "skills"),
    )


def _all_local_tools() -> list:
    return [
        _rag_explore,
        _rag_search,
        _rag_get_context,
        _recall_history,
        _read_file,
        _bash,
    ]


def test_academic_skill_resolves_history_capability_and_denies_shell(tmp_path):
    runtime = load_skill_runtime(
        "academic-paper-writing",
        config=_cfg(tmp_path),
        all_tools=_all_local_tools(),
    )

    assert runtime.tool_policy_active is True
    assert runtime.capability_resolution.requested_required == frozenset({
        "file.read",
        "rag.search",
        "history.search",
    })
    assert runtime.capability_resolution.unresolved_required == frozenset()
    assert {
        "read_file",
        "rag_explore",
        "rag_search",
        "rag_get_context",
        "recall_history",
    }.issubset(runtime.allowed_tools)
    assert "bash" in runtime.denied_tools
    assert "bash" not in runtime.allowed_tools


def test_academic_skill_fails_fast_when_history_capability_cannot_resolve(tmp_path):
    with pytest.raises(ValueError, match="history.search"):
        load_skill_runtime(
            "academic-paper-writing",
            config=_cfg(tmp_path),
            all_tools=[
                _rag_explore,
                _rag_search,
                _rag_get_context,
                _read_file,
                _bash,
            ],
        )


def test_academic_skill_writer_binding_includes_recall_history_schema(
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

    runtime = load_skill_runtime(
        "academic-paper-writing",
        config=_cfg(tmp_path),
        all_tools=_all_local_tools(),
    )

    monkeypatch.setattr("agent.graph.get_chat_model", lambda _cfg: RecordingModel())
    monkeypatch.setattr(
        "agent.tools.inventory.create_rag_tools",
        lambda _cfg: [_rag_explore, _rag_search, _rag_get_context],
    )
    monkeypatch.setattr(
        "agent.tools.inventory.create_history_tool",
        lambda _cfg, store=None: _recall_history,
    )
    monkeypatch.setattr("agent.tools.inventory.create_read_file_tool", lambda _cfg: _read_file)
    monkeypatch.setattr("agent.tools.inventory.create_bash_tool", lambda _cfg: _bash)

    graph = build_graph(_cfg(tmp_path), skill_runtime_getter=lambda: runtime)
    graph.invoke({
        "messages": [
            HumanMessage(
                content=(
                    "我一月上半的成果如果要寫成論文，abstract 重點是什麼？"
                    "我不記得了，你自行看一下。"
                )
            )
        ],
    })

    active_skill_bindings = bind_calls[1:]
    assert active_skill_bindings
    assert active_skill_bindings[-1] == [
        "rag_explore",
        "rag_search",
        "rag_get_context",
        "recall_history",
        "read_file",
    ]
    assert "bash" not in active_skill_bindings[-1]
