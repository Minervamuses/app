"""Tests for ChatSession's recent-turns eviction into the long-term store.

Mocks both the LangGraph graph and the ChatHistoryStore so these tests
don't need an LLM, Ollama, or ChromaDB. The point is to verify the
eviction policy:

- on overflow, the oldest turn is sent to history_store.add_turn
- the recent_turns window stays bounded at the configured size
- store failures are logged and the turn stays put
- once recent_turns exceeds window * 3, oldest is dropped unrecorded
"""

import asyncio
import logging

import pytest
from langchain_core.messages import AIMessage

from agent.config import AgentConfig
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
