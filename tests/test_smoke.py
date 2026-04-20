"""Smoke tests that should pass before and after the decoupling refactor."""

from langchain_core.messages import AIMessage
from langchain_core.tools import tool


def test_imports():
    """Core modules should import without circular or structural failures."""
    import rag
    import agent
    from agent.config import AgentConfig

    assert rag is not None
    assert agent is not None
    assert AgentConfig is not None

    import rag.retriever.vector
    import rag.store.chroma_store
    import agent.graph
    import agent.evaluation.behavior


def test_graph_builds_without_error(monkeypatch, tmp_path):
    """The graph should compile with lightweight test doubles."""
    from agent.graph import build_graph
    from agent.config import AgentConfig

    @tool("explore")
    def fake_explore() -> str:
        """Dummy explore tool for smoke tests."""
        return "explore"

    @tool("search")
    def fake_search(query: str) -> str:
        """Dummy search tool for smoke tests."""
        return query

    @tool("get_context")
    def fake_context(pid: str, chunk_id: int) -> str:
        """Dummy context tool for smoke tests."""
        return f"{pid}:{chunk_id}"

    class DummyModel:
        def bind_tools(self, _tools):
            return self

        def invoke(self, _messages):
            return AIMessage(content="ok")

    monkeypatch.setattr("agent.graph.get_chat_model", lambda _config: DummyModel())
    monkeypatch.setattr("agent.graph.create_explore_tool", lambda _config: fake_explore)
    monkeypatch.setattr("agent.graph.create_search_tool", lambda _config: fake_search)
    monkeypatch.setattr("agent.graph.create_context_tool", lambda _config: fake_context)

    cfg = AgentConfig(persist_dir=str(tmp_path))
    graph = build_graph(cfg)

    assert graph is not None
