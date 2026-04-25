"""Tests for ChatSession's recent-turn persistence into the long-term store.

Mocks both the LangGraph graph and the ChatHistoryStore so these tests
don't need an LLM, Ollama, or ChromaDB. The point is to verify the
eviction policy:

- on overflow, the oldest turn is sent to history_store.add_turn
- the recent_turns window stays bounded at the configured size
- store failures are logged and the turn stays put
- once recent_turns exceeds window * 3, oldest is dropped unrecorded
- on shutdown, remaining prompt-visible turns are flushed to history_store
"""

import asyncio
import logging

import pytest
from langchain_core.messages import AIMessage

from agent.config import AgentConfig
from agent.history import prepare_messages_for_agent
from agent.memory import TurnRecord
from agent.session import ChatSession


class _FakeGraph:
    async def astream(self, state, config=None, stream_mode="updates"):
        yield {"agent": {"messages": [AIMessage(content="ok")]}}


class _FakeHistoryStore:
    def __init__(self, raise_on_add: bool = False):
        self.adds: list[dict] = []
        self.raise_on_add = raise_on_add

    def add_turn(self, turn: TurnRecord, *, session_id: str, turn_id: int, timestamp: str) -> None:
        if self.raise_on_add:
            raise RuntimeError("ollama unavailable")
        self.adds.append(
            {
                "user_input": turn.user_input,
                "assistant_output": turn.assistant_output,
                "session_id": session_id,
                "turn_id": turn_id,
                "timestamp": timestamp,
            }
        )


class _PreparingGraph:
    def __init__(self, config: AgentConfig, history_store: _FakeHistoryStore):
        self.config = config
        self.history_store = history_store
        self.snapshots: list[dict] = []

    async def astream(self, state, config=None, stream_mode="updates"):
        prepared = prepare_messages_for_agent(
            state["messages"],
            max_messages=self.config.agent_max_messages,
            max_tool_interactions=self.config.agent_max_tool_interactions,
        )
        self.snapshots.append({
            "persisted_before_agent": [item["user_input"] for item in self.history_store.adds],
            "contents": [msg.content for msg in prepared],
        })
        yield {"agent": {"messages": [AIMessage(content="ok")]}}


@pytest.fixture
def make_session(monkeypatch, tmp_path):
    """Factory: build a ChatSession with a fake graph and fake history store."""
    monkeypatch.setattr(
        "agent.session.build_graph",
        lambda _cfg, extra_tools=None: _FakeGraph(),
    )

    def _make(window: int, history_store: _FakeHistoryStore | None = None):
        cfg = AgentConfig(persist_dir=str(tmp_path))
        cfg.agent_recent_turns_window = window
        store = history_store or _FakeHistoryStore()
        session = ChatSession(cfg, history_store=store)
        return session, store

    return _make


def test_no_eviction_below_window(make_session):
    session, store = make_session(window=3)
    for i in range(3):
        asyncio.run(session.turn(f"q{i}"))
    assert store.adds == []
    assert len(session.recent_turns) == 3


def test_overflow_evicts_oldest_into_history_store(make_session):
    session, store = make_session(window=3)
    for i in range(4):
        asyncio.run(session.turn(f"q{i}"))

    assert len(store.adds) == 1
    evicted = store.adds[0]
    assert evicted["user_input"] == "q0"
    assert evicted["assistant_output"] == "ok"
    assert evicted["session_id"] == session.session_id
    assert evicted["turn_id"] == 1
    assert evicted["timestamp"]

    assert len(session.recent_turns) == 3
    assert [t.user_input for t in session.recent_turns] == ["q1", "q2", "q3"]


def test_window_stays_bounded_across_many_turns(make_session):
    session, store = make_session(window=3)
    for i in range(10):
        asyncio.run(session.turn(f"q{i}"))

    assert len(session.recent_turns) == 3
    assert [t.user_input for t in session.recent_turns] == ["q7", "q8", "q9"]
    assert len(store.adds) == 7
    assert [a["user_input"] for a in store.adds] == [f"q{i}" for i in range(7)]


def test_eviction_failure_logs_and_keeps_turn(make_session, caplog):
    failing_store = _FakeHistoryStore(raise_on_add=True)
    session, _ = make_session(window=2, history_store=failing_store)

    with caplog.at_level(logging.WARNING, logger="agent.session"):
        for i in range(3):
            asyncio.run(session.turn(f"q{i}"))

    assert len(session.recent_turns) == 3
    assert any("eviction failed" in rec.message for rec in caplog.records)


def test_hard_cap_drops_oldest_after_persistent_failure(make_session, caplog):
    failing_store = _FakeHistoryStore(raise_on_add=True)
    session, _ = make_session(window=2, history_store=failing_store)

    with caplog.at_level(logging.ERROR, logger="agent.session"):
        # window=2, hard_cap = 6. Drive 8 turns; once recent_turns exceeds 6,
        # oldest gets dropped unrecorded on each subsequent overflow.
        for i in range(8):
            asyncio.run(session.turn(f"q{i}"))

    assert len(session.recent_turns) <= 6
    assert any("hard cap" in rec.message for rec in caplog.records)


def test_turn_leaving_message_cap_stays_visible_until_evicted(make_session):
    session, store = make_session(window=10)
    graph = _PreparingGraph(session.config, store)
    session.graph = graph

    for i in range(1, 12):
        asyncio.run(session.turn(f"q{i}"))

    turn_11 = graph.snapshots[10]
    assert turn_11["persisted_before_agent"] == []
    assert "q1" in turn_11["contents"]
    assert [item["user_input"] for item in store.adds] == ["q1"]


def test_failed_eviction_turn_stays_prompt_visible(make_session):
    failing_store = _FakeHistoryStore(raise_on_add=True)
    session, _ = make_session(window=2, history_store=failing_store)
    session.config.agent_max_messages = 4
    graph = _PreparingGraph(session.config, failing_store)
    session.graph = graph

    for i in range(4):
        asyncio.run(session.turn(f"q{i}"))

    turn_4 = graph.snapshots[3]
    assert "q0" in turn_4["contents"]


def test_flush_recent_turns_persists_short_session(make_session):
    session, store = make_session(window=10)
    for i in range(2):
        asyncio.run(session.turn(f"q{i}"))

    assert store.adds == []

    asyncio.run(session.flush_recent_turns())

    assert [item["user_input"] for item in store.adds] == ["q0", "q1"]
    assert session.recent_turns == []


def test_flush_recent_turns_is_idempotent(make_session):
    session, store = make_session(window=10)
    asyncio.run(session.turn("q0"))

    asyncio.run(session.flush_recent_turns())
    asyncio.run(session.flush_recent_turns())

    assert [item["user_input"] for item in store.adds] == ["q0"]


def test_flush_recent_turns_logs_and_keeps_turn_on_failure(make_session, caplog):
    failing_store = _FakeHistoryStore(raise_on_add=True)
    session, _ = make_session(window=10, history_store=failing_store)
    asyncio.run(session.turn("q0"))

    with caplog.at_level(logging.WARNING, logger="agent.session"):
        asyncio.run(session.flush_recent_turns())

    assert len(session.recent_turns) == 1
    assert any("shutdown flush failed" in rec.message for rec in caplog.records)
