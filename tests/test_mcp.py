"""Tests for MCP loader behavior and MCP-aware session startup."""

import asyncio

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from agent import mcp as mcp_module


def _clear_mcp_env(monkeypatch):
    for key in [
        "AGENT_ENABLE_MCP_WEB_SEARCH",
        "AGENT_MCP_WEB_SEARCH_COMMAND",
        "AGENT_MCP_WEB_SEARCH_ARGS",
        "AGENT_ENABLE_MCP_GITHUB",
        "AGENT_MCP_GITHUB_COMMAND",
        "AGENT_MCP_GITHUB_ARGS",
        "AGENT_MCP_GITHUB_TOOLSETS",
        "GITHUB_PERSONAL_ACCESS_TOKEN",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_resolve_mcp_specs_empty_when_nothing_enabled(monkeypatch):
    _clear_mcp_env(monkeypatch)
    assert mcp_module.resolve_mcp_specs() == []


def test_resolve_mcp_specs_skips_enabled_but_missing_command(monkeypatch, caplog):
    _clear_mcp_env(monkeypatch)
    monkeypatch.setenv("AGENT_ENABLE_MCP_WEB_SEARCH", "1")
    specs = mcp_module.resolve_mcp_specs()
    assert specs == []


def test_resolve_mcp_specs_web_search(monkeypatch):
    _clear_mcp_env(monkeypatch)
    monkeypatch.setenv("AGENT_ENABLE_MCP_WEB_SEARCH", "true")
    monkeypatch.setenv("AGENT_MCP_WEB_SEARCH_COMMAND", "npx")
    monkeypatch.setenv("AGENT_MCP_WEB_SEARCH_ARGS", "-y web-search-mcp@latest")
    specs = mcp_module.resolve_mcp_specs()
    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "web_search"
    assert spec.command == "npx"
    assert spec.args == ["-y", "web-search-mcp@latest"]
    assert spec.env == {}


def test_resolve_mcp_specs_github_carries_token_and_toolsets(monkeypatch):
    _clear_mcp_env(monkeypatch)
    monkeypatch.setenv("AGENT_ENABLE_MCP_GITHUB", "yes")
    monkeypatch.setenv("AGENT_MCP_GITHUB_COMMAND", "/opt/github-mcp-server")
    monkeypatch.setenv("AGENT_MCP_GITHUB_ARGS", "stdio")
    monkeypatch.setenv("AGENT_MCP_GITHUB_TOOLSETS", "repos,pull_requests")
    monkeypatch.setenv("GITHUB_PERSONAL_ACCESS_TOKEN", "ghp_test")
    specs = mcp_module.resolve_mcp_specs()
    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "github"
    assert spec.command == "/opt/github-mcp-server"
    assert spec.args == ["stdio"]
    assert spec.env == {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_test",
        "GITHUB_TOOLSETS": "repos,pull_requests",
    }


def test_resolve_mcp_specs_both_servers(monkeypatch):
    _clear_mcp_env(monkeypatch)
    monkeypatch.setenv("AGENT_ENABLE_MCP_WEB_SEARCH", "1")
    monkeypatch.setenv("AGENT_MCP_WEB_SEARCH_COMMAND", "npx")
    monkeypatch.setenv("AGENT_ENABLE_MCP_GITHUB", "1")
    monkeypatch.setenv("AGENT_MCP_GITHUB_COMMAND", "gh-mcp")
    monkeypatch.setenv("GITHUB_PERSONAL_ACCESS_TOKEN", "ghp_test")
    specs = mcp_module.resolve_mcp_specs()
    names = {spec.name for spec in specs}
    assert names == {"web_search", "github"}


def test_load_mcp_tools_empty_without_specs():
    tools = asyncio.run(mcp_module.load_mcp_tools(specs=[]))
    assert tools == []


def test_load_mcp_tools_skips_failing_server(monkeypatch):
    @tool("web_fetch")
    def fake_web(url: str) -> str:
        """Fake web search tool."""
        return url

    async def ok_get_tools(self, *, server_name=None):
        return [fake_web]

    async def bad_get_tools(self, *, server_name=None):
        raise RuntimeError("mcp server crashed")

    class FakeClient:
        def __init__(self, connections=None):
            self.connections = connections
            server = next(iter(connections.keys()))
            if server == "github":
                self.get_tools = bad_get_tools.__get__(self, FakeClient)
            else:
                self.get_tools = ok_get_tools.__get__(self, FakeClient)

    import langchain_mcp_adapters.client as client_module
    monkeypatch.setattr(client_module, "MultiServerMCPClient", FakeClient)

    specs = [
        mcp_module.MCPServerSpec(name="web_search", command="x", args=[], env={}),
        mcp_module.MCPServerSpec(name="github", command="y", args=[], env={}),
    ]
    tools = asyncio.run(mcp_module.load_mcp_tools(specs=specs))
    assert len(tools) == 1
    assert tools[0].name == "web_fetch"


def test_session_create_without_mcp(monkeypatch, tmp_path):
    from agent.session import ChatSession
    from agent.config import AgentConfig

    class DummyModel:
        def bind_tools(self, _tools):
            return self

        def invoke(self, _messages):
            return AIMessage(content="ok")

    @tool("rag_explore")
    def fake_explore() -> str:
        """e"""
        return "explore"

    @tool("rag_search")
    def fake_search(query: str) -> str:
        """s"""
        return query

    @tool("rag_get_context")
    def fake_context(pid: str, chunk_id: int) -> str:
        """c"""
        return f"{pid}:{chunk_id}"

    monkeypatch.setattr("agent.graph.get_chat_model", lambda _c: DummyModel())
    monkeypatch.setattr(
        "agent.graph.create_rag_tools",
        lambda _c: [fake_explore, fake_search, fake_context],
    )

    cfg = AgentConfig(persist_dir=str(tmp_path))
    session = asyncio.run(ChatSession.create(cfg, load_mcp=False))
    assert session is not None
    assert session.recent_turns == []


def test_session_create_loads_mcp_tools(monkeypatch, tmp_path):
    from agent.session import ChatSession
    from agent.config import AgentConfig

    @tool("web_fetch")
    def fake_web(url: str) -> str:
        """fake mcp tool"""
        return url

    @tool("rag_explore")
    def fake_explore() -> str:
        """e"""
        return "explore"

    @tool("rag_search")
    def fake_search(query: str) -> str:
        """s"""
        return query

    @tool("rag_get_context")
    def fake_context(pid: str, chunk_id: int) -> str:
        """c"""
        return f"{pid}:{chunk_id}"

    seen_tools: dict = {}

    def capture_bind(tools):
        seen_tools["bound"] = tools

        class M:
            def invoke(self, _m):
                return AIMessage(content="ok")

        return M()

    class DummyModel:
        def bind_tools(self, tools):
            return capture_bind(tools)

    @tool("recall_history")
    def fake_recall(query: str) -> str:
        """h"""
        return query

    monkeypatch.setattr("agent.graph.get_chat_model", lambda _c: DummyModel())
    monkeypatch.setattr(
        "agent.graph.create_rag_tools",
        lambda _c: [fake_explore, fake_search, fake_context],
    )
    monkeypatch.setattr("agent.graph.create_history_tool", lambda _c, store=None: fake_recall)

    async def fake_load():
        return [fake_web]

    monkeypatch.setattr("agent.mcp.load_mcp_tools", fake_load)

    cfg = AgentConfig(persist_dir=str(tmp_path))
    session = asyncio.run(ChatSession.create(cfg, load_mcp=True))
    bound_names = [t.name for t in seen_tools["bound"]]
    assert bound_names == [
        "rag_explore",
        "rag_search",
        "rag_get_context",
        "recall_history",
        "web_fetch",
    ]
    assert session is not None


def test_session_create_survives_mcp_failure(monkeypatch, tmp_path):
    from agent.session import ChatSession
    from agent.config import AgentConfig

    @tool("rag_explore")
    def fake_explore() -> str:
        """e"""
        return "explore"

    @tool("rag_search")
    def fake_search(query: str) -> str:
        """s"""
        return query

    @tool("rag_get_context")
    def fake_context(pid: str, chunk_id: int) -> str:
        """c"""
        return f"{pid}:{chunk_id}"

    @tool("recall_history")
    def fake_recall(query: str) -> str:
        """h"""
        return query

    seen: dict = {}

    class DummyModel:
        def bind_tools(self, tools):
            seen["bound"] = tools

            class M:
                def invoke(self, _m):
                    return AIMessage(content="ok")

            return M()

    monkeypatch.setattr("agent.graph.get_chat_model", lambda _c: DummyModel())
    monkeypatch.setattr(
        "agent.graph.create_rag_tools",
        lambda _c: [fake_explore, fake_search, fake_context],
    )
    monkeypatch.setattr("agent.graph.create_history_tool", lambda _c, store=None: fake_recall)

    async def failing_load():
        raise RuntimeError("mcp unavailable")

    monkeypatch.setattr("agent.mcp.load_mcp_tools", failing_load)

    cfg = AgentConfig(persist_dir=str(tmp_path))
    session = asyncio.run(ChatSession.create(cfg, load_mcp=True))
    assert session is not None
    # Only local agent tools bound (rag + recall_history); MCP load failed.
    assert [t.name for t in seen["bound"]] == [
        "rag_explore",
        "rag_search",
        "rag_get_context",
        "recall_history",
    ]
