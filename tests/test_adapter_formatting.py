"""Ensure LangChain adapters follow rag's tool contract."""

import json

from langchain_core.tools import StructuredTool


def test_create_rag_tools_defaults_to_chat_safe_tools(tmp_path):
    """Chat should not bind rag_list_chunks by default."""
    from agent.adapters.langchain.rag_tools import create_rag_tools
    from agent.config import AgentConfig

    tools = create_rag_tools(AgentConfig(persist_dir=str(tmp_path)))

    assert all(isinstance(item, StructuredTool) for item in tools)
    assert [tool.name for tool in tools] == [
        "rag_explore",
        "rag_search",
        "rag_get_context",
    ]


def test_create_rag_tools_can_include_list_chunks(tmp_path):
    """rag_list_chunks is available as an explicit opt-in."""
    from agent.adapters.langchain.rag_tools import create_rag_tools
    from agent.config import AgentConfig

    tools = create_rag_tools(
        AgentConfig(persist_dir=str(tmp_path)),
        include_list_chunks=True,
    )

    assert [tool.name for tool in tools] == [
        "rag_explore",
        "rag_search",
        "rag_get_context",
        "rag_list_chunks",
    ]


def test_rag_tool_invokes_dispatch_and_renders_json(monkeypatch, tmp_path):
    """Tool output should be stable JSON returned from rag.dispatch."""
    from agent.adapters.langchain import rag_tools
    from agent.config import AgentConfig

    seen: dict = {}

    def fake_dispatch(name, args, *, config=None):
        seen["name"] = name
        seen["args"] = args
        seen["config"] = config
        return [{"pid": "p1", "chunk_id": 0, "text": "hello"}]

    monkeypatch.setattr(rag_tools, "dispatch", fake_dispatch)

    config = AgentConfig(persist_dir=str(tmp_path))
    tools = rag_tools.create_rag_tools(config)
    search = next(tool for tool in tools if tool.name == "rag_search")
    out = search.invoke({"query": "x", "k": 1})

    assert seen == {
        "name": "rag_search",
        "args": {"query": "x", "k": 1},
        "config": config,
    }
    assert json.loads(out) == [{"pid": "p1", "chunk_id": 0, "text": "hello"}]
