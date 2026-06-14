"""Tests for the single-source base tool inventory."""

from pathlib import Path

from langchain_core.tools import tool

from agent.config import AgentConfig
from agent.tools import inventory as tool_inventory


APP_ROOT = Path(__file__).resolve().parents[1]


@tool("web_fetch")
def _web_fetch(url: str) -> str:
    """Fake MCP tool."""
    return url


@tool("rag_search")
def _shadow_rag_search(query: str) -> str:
    """Extra tool that collides with a local base tool name."""
    return query


class _DummyHistoryStore:
    def search(self, query, k=5, role=None):
        return []


def test_base_tool_names_fixed_local_order():
    assert tool_inventory.base_tool_names() == [
        "rag_explore",
        "rag_search",
        "rag_get_context",
        "recall_history",
        "read_file",
        "bash",
    ]


def test_base_tool_names_appends_extra_tools_in_order():
    assert tool_inventory.base_tool_names([_web_fetch]) == [
        "rag_explore",
        "rag_search",
        "rag_get_context",
        "recall_history",
        "read_file",
        "bash",
        "web_fetch",
    ]


def test_base_tool_names_drops_extra_tool_colliding_with_local():
    # A same-named extra tool is ignored: local base tool wins.
    assert tool_inventory.base_tool_names([_shadow_rag_search]) == (
        tool_inventory.base_tool_names()
    )


def test_build_base_tools_names_match_base_tool_names(tmp_path):
    cfg = AgentConfig(persist_dir=str(tmp_path))
    tools = tool_inventory.build_base_tools(
        cfg,
        history_store=_DummyHistoryStore(),
        extra_tools=[_web_fetch],
    )
    built_names = [tool.name for tool in tools]

    assert built_names == tool_inventory.base_tool_names([_web_fetch])


def test_build_base_tools_does_not_double_bind_colliding_extra(tmp_path):
    cfg = AgentConfig(persist_dir=str(tmp_path))
    tools = tool_inventory.build_base_tools(
        cfg,
        history_store=_DummyHistoryStore(),
        extra_tools=[_shadow_rag_search],
    )
    built_names = [tool.name for tool in tools]

    assert built_names.count("rag_search") == 1
    assert built_names == tool_inventory.base_tool_names()


def test_render_base_tool_prompt_has_no_import_or_runtime_side_effects(monkeypatch):
    def _explode(*_args, **_kwargs):
        raise AssertionError("render_base_tool_prompt must not build tools")

    monkeypatch.setattr(tool_inventory, "create_history_tool", _explode)
    monkeypatch.setattr(tool_inventory, "create_rag_tools", _explode)
    monkeypatch.setattr(tool_inventory, "create_read_file_tool", _explode)
    monkeypatch.setattr(tool_inventory, "create_bash_tool", _explode)

    prompt = tool_inventory.render_base_tool_prompt()
    names = tool_inventory.base_tool_names()

    assert "**rag_explore**" in prompt
    assert "Tool selection policy:" in prompt
    assert "Workflow:" in prompt
    assert names[0] == "rag_explore"


def test_render_base_tool_prompt_covers_every_base_tool():
    prompt = tool_inventory.render_base_tool_prompt()
    for name in tool_inventory.base_tool_names():
        assert f"**{name}**" in prompt


def test_system_prompt_embeds_rendered_base_tool_prompt():
    from agent.session import SYSTEM_PROMPT

    assert tool_inventory.render_base_tool_prompt() in SYSTEM_PROMPT


def test_session_source_does_not_duplicate_base_tool_inventory():
    source = (APP_ROOT / "agent" / "session.py").read_text(encoding="utf-8")

    # The inventory/routing/workflow literals must live only in inventory.py.
    assert "Tool selection policy:" not in source
    assert "Workflow:" not in source
    assert "Discover what's in the indexed knowledge base" not in source
    assert "**rag_explore**" not in source

    # A single semantic mention (e.g. the plan-mode hint) is still allowed.
    assert "recall_history" in source


def test_tool_inventory_evaluator_helper_delegates_to_inventory():
    from agent.evaluation.base import tool_inventory as eval_tool_inventory

    assert eval_tool_inventory() == tool_inventory.base_tool_names()
    assert eval_tool_inventory([_web_fetch]) == tool_inventory.base_tool_names(
        [_web_fetch]
    )


def test_behavior_tool_names_include_web_behavior_universe():
    names = tool_inventory.behavior_tool_names()

    for web_name in tool_inventory.WEB_BEHAVIOR_TOOL_NAMES:
        assert web_name in names
    # Behavior universe = local base behavior tools + frozen web names.
    assert set(names) == set(tool_inventory.base_tool_names()) | set(
        tool_inventory.WEB_BEHAVIOR_TOOL_NAMES
    )


def test_tool_routing_public_surface_tracks_inventory():
    from agent.evaluation.metrics.tool_routing import (
        ALL_BEHAVIOR_TOOL_NAMES,
        HISTORY_FORBIDDEN,
        HISTORY_TOOL_NAMES,
        LOCAL_TOOL_NAMES,
        RAG_FORBIDDEN,
        RAG_TOOL_NAMES,
        TOOL_FAMILIES,
        WEB_FORBIDDEN,
        WEB_TOOL_NAMES,
    )

    assert set(ALL_BEHAVIOR_TOOL_NAMES) == set(tool_inventory.behavior_tool_names())
    assert WEB_TOOL_NAMES == tool_inventory.WEB_BEHAVIOR_TOOL_NAMES
    assert set(RAG_FORBIDDEN) == set(HISTORY_TOOL_NAMES) | set(WEB_TOOL_NAMES) | set(
        LOCAL_TOOL_NAMES
    )
    assert set(WEB_FORBIDDEN) == set(RAG_TOOL_NAMES) | set(HISTORY_TOOL_NAMES) | set(
        LOCAL_TOOL_NAMES
    )
    assert set(HISTORY_FORBIDDEN) == set(RAG_TOOL_NAMES) | set(WEB_TOOL_NAMES) | set(
        LOCAL_TOOL_NAMES
    )
    assert TOOL_FAMILIES["web"] == set(WEB_TOOL_NAMES)
    assert "read_file" in RAG_FORBIDDEN
    assert "bash" in RAG_FORBIDDEN
