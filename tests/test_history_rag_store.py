"""Tests for agent.history_rag.store.

Mocks rag.ChromaStore + VectorRetriever so we never hit Ollama or
ChromaDB. The point is to verify the wiring (metadata schema, role
filter, cache behavior), not the underlying vector store.
"""

import pytest
from langchain_core.documents import Document


@pytest.fixture
def fake_chroma(monkeypatch):
    """Replace ChromaStore + VectorRetriever and reset the module cache."""
    captured: dict = {"add_calls": [], "retrieve_calls": []}

    class FakeChromaStore:
        def __init__(self, collection_name, config):
            captured["collection_name"] = collection_name
            captured["persist_dir"] = config.persist_dir

        def add(self, documents):
            captured["add_calls"].append(documents)

    class FakeVectorRetriever:
        def __init__(self, store):
            self._store = store

        def retrieve(self, query, k, pid_filter=None, where=None):
            captured["retrieve_calls"].append({"query": query, "k": k, "where": where})
            return []

    monkeypatch.setattr("agent.history_rag.store.ChromaStore", FakeChromaStore)
    monkeypatch.setattr("agent.history_rag.store.VectorRetriever", FakeVectorRetriever)
    from agent.history_rag import store as store_mod
    monkeypatch.setattr(store_mod, "_chat_store_cache", {})

    return captured


def test_add_turn_emits_two_documents_with_role_metadata(tmp_path, fake_chroma):
    from agent.config import AgentConfig
    from agent.memory import TurnRecord
    from agent.history_rag.store import ChatHistoryStore

    store = ChatHistoryStore(AgentConfig(persist_dir=str(tmp_path)))
    store.add_turn(
        TurnRecord(user_input="hi", assistant_output="hello"),
        session_id="sess-1",
        turn_id=3,
        timestamp="2026-04-25T12:00:00",
    )

    docs = fake_chroma["add_calls"][0]
    assert len(docs) == 2

    by_role = {d.metadata["role"]: d for d in docs}
    assert set(by_role) == {"user", "assistant"}
    assert by_role["user"].page_content == "hi"
    assert by_role["assistant"].page_content == "hello"

    expected_meta = {
        "role": "user",
        "turn_id": 3,
        "session_id": "sess-1",
        "timestamp": "2026-04-25T12:00:00",
    }
    assert by_role["user"].metadata == expected_meta
    assert by_role["assistant"].metadata == {**expected_meta, "role": "assistant"}


def test_add_turn_skips_empty_strings(tmp_path, fake_chroma):
    from agent.config import AgentConfig
    from agent.memory import TurnRecord
    from agent.history_rag.store import ChatHistoryStore

    store = ChatHistoryStore(AgentConfig(persist_dir=str(tmp_path)))
    store.add_turn(
        TurnRecord(user_input="hi", assistant_output=""),
        session_id="s",
        turn_id=1,
        timestamp="t",
    )

    docs = fake_chroma["add_calls"][0]
    assert len(docs) == 1
    assert docs[0].metadata["role"] == "user"


def test_add_turn_with_both_empty_does_not_call_store(tmp_path, fake_chroma):
    from agent.config import AgentConfig
    from agent.memory import TurnRecord
    from agent.history_rag.store import ChatHistoryStore

    store = ChatHistoryStore(AgentConfig(persist_dir=str(tmp_path)))
    store.add_turn(
        TurnRecord(user_input="", assistant_output=""),
        session_id="s",
        turn_id=1,
        timestamp="t",
    )
    assert fake_chroma["add_calls"] == []


def test_search_passes_role_where_clause(tmp_path, fake_chroma):
    from agent.config import AgentConfig
    from agent.history_rag.store import ChatHistoryStore

    store = ChatHistoryStore(AgentConfig(persist_dir=str(tmp_path)))
    store.search("what did I ask?", k=4, role="user")

    call = fake_chroma["retrieve_calls"][0]
    assert call == {"query": "what did I ask?", "k": 4, "where": {"role": {"$eq": "user"}}}


def test_search_without_role_passes_no_where(tmp_path, fake_chroma):
    from agent.config import AgentConfig
    from agent.history_rag.store import ChatHistoryStore

    store = ChatHistoryStore(AgentConfig(persist_dir=str(tmp_path)))
    store.search("anything", k=5)

    call = fake_chroma["retrieve_calls"][0]
    assert call == {"query": "anything", "k": 5, "where": None}


def test_collection_name_and_persist_dir(tmp_path, fake_chroma):
    from agent.config import AgentConfig
    from agent.history_rag.store import CHAT_HISTORY_COLLECTION, ChatHistoryStore

    ChatHistoryStore(AgentConfig(persist_dir=str(tmp_path)))

    assert fake_chroma["collection_name"] == CHAT_HISTORY_COLLECTION
    assert fake_chroma["persist_dir"].endswith("/chat_history")
    assert fake_chroma["persist_dir"].startswith(str(tmp_path))


def test_get_chat_history_store_caches_per_persist_dir(tmp_path, fake_chroma):
    from agent.config import AgentConfig
    from agent.history_rag.store import get_chat_history_store

    cfg_a = AgentConfig(persist_dir=str(tmp_path / "a"))
    cfg_b = AgentConfig(persist_dir=str(tmp_path / "b"))

    a1 = get_chat_history_store(cfg_a)
    a2 = get_chat_history_store(cfg_a)
    b1 = get_chat_history_store(cfg_b)

    assert a1 is a2
    assert a1 is not b1


def test_search_returns_documents_from_retriever(tmp_path, monkeypatch):
    """Sanity: results from the retriever are returned verbatim."""
    sentinel = [Document(page_content="x", metadata={"role": "user"})]

    class FakeChromaStore:
        def __init__(self, *a, **kw):
            pass

        def add(self, docs):
            pass

    class FakeVectorRetriever:
        def __init__(self, store):
            pass

        def retrieve(self, query, k, pid_filter=None, where=None):
            return sentinel

    monkeypatch.setattr("agent.history_rag.store.ChromaStore", FakeChromaStore)
    monkeypatch.setattr("agent.history_rag.store.VectorRetriever", FakeVectorRetriever)
    from agent.history_rag import store as store_mod
    monkeypatch.setattr(store_mod, "_chat_store_cache", {})

    from agent.config import AgentConfig
    from agent.history_rag.store import ChatHistoryStore

    store = ChatHistoryStore(AgentConfig(persist_dir=str(tmp_path)))
    assert store.search("q") is sentinel
