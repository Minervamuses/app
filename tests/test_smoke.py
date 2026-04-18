"""Smoke tests that should pass before and after the decoupling refactor."""

from langchain_core.messages import AIMessage
from langchain_core.tools import tool


def test_imports():
    """Core modules should import without circular or structural failures."""
    import kms
    import kms_agent
    from kms.config import KMSConfig

    assert kms is not None
    assert kms_agent is not None
    assert KMSConfig is not None

    import kms.retriever.vector
    import kms.store.chroma_store
    import kms.agent.graph
    import kms_agent.agent.graph
    import kms_agent.evaluation.behavior


def test_graph_builds_without_error(monkeypatch, tmp_path):
    """The graph should compile with lightweight test doubles."""
    from kms.agent.graph import build_graph
    from kms.config import KMSConfig

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

    monkeypatch.setattr("kms.agent.graph.get_chat_model", lambda _config: DummyModel())
    monkeypatch.setattr("kms.agent.graph.create_explore_tool", lambda _config: fake_explore)
    monkeypatch.setattr("kms.agent.graph.create_search_tool", lambda _config: fake_search)
    monkeypatch.setattr("kms.agent.graph.create_context_tool", lambda _config: fake_context)

    cfg = KMSConfig(persist_dir=str(tmp_path))
    graph = build_graph(cfg)

    assert graph is not None
