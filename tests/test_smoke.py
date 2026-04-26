"""Smoke tests that should pass before and after the decoupling refactor."""

from langchain_core.messages import AIMessage


def test_imports():
    """Core modules should import without circular or structural failures."""
    import rag
    import agent
    from agent.config import AgentConfig

    assert rag is not None
    assert agent is not None
    assert AgentConfig is not None
    assert {tool["name"] for tool in rag.TOOL_SCHEMAS} == {
        "rag_search",
        "rag_explore",
        "rag_list_chunks",
        "rag_get_context",
    }

    import agent.graph
    import agent.evaluation.behavior


def test_graph_builds_without_error(monkeypatch, tmp_path):
    """The graph should compile with real rag tools and a lightweight model."""
    from agent.graph import build_graph
    from agent.config import AgentConfig

    class DummyModel:
        def bind_tools(self, _tools):
            return self

        def invoke(self, _messages):
            return AIMessage(content="ok")

    monkeypatch.setattr("agent.graph.get_chat_model", lambda _config: DummyModel())

    cfg = AgentConfig(persist_dir=str(tmp_path))
    graph = build_graph(cfg)

    assert graph is not None


def test_graph_passes_history_store_to_recall_tool(monkeypatch, tmp_path):
    """Injected history stores should back the recall_history tool."""
    from langchain_core.tools import tool

    from agent.graph import build_graph
    from agent.config import AgentConfig

    class DummyModel:
        def bind_tools(self, _tools):
            return self

        def invoke(self, _messages):
            return AIMessage(content="ok")

    @tool("rag_explore")
    def fake_explore() -> str:
        """fake explore"""
        return "explore"

    @tool("recall_history")
    def fake_recall(query: str) -> str:
        """fake recall"""
        return query

    seen: dict = {}
    fake_store = object()

    monkeypatch.setattr("agent.graph.get_chat_model", lambda _config: DummyModel())
    monkeypatch.setattr("agent.graph.create_rag_tools", lambda _config: [fake_explore])

    def capture_history_tool(_config, store=None):
        seen["store"] = store
        return fake_recall

    monkeypatch.setattr("agent.graph.create_history_tool", capture_history_tool)

    cfg = AgentConfig(persist_dir=str(tmp_path))
    graph = build_graph(cfg, history_store=fake_store)

    assert graph is not None
    assert seen["store"] is fake_store
