"""Tests for plan-mode markdown persistence."""

import asyncio

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agent.config import AgentConfig
from agent.memory import TurnRecord
from agent.session import ChatSession


class _FakeHistoryStore:
    def __init__(self):
        self.adds: list[TurnRecord] = []

    def add_turn(self, turn: TurnRecord, *, session_id: str, turn_id: int, timestamp: str) -> None:
        self.adds.append(turn)


class _AnswerGraph:
    def __init__(self, answer: str = "ok"):
        self.answer = answer

    async def astream(self, state, config=None, stream_mode="updates"):
        yield {"agent": {"messages": [AIMessage(content=self.answer)]}}


class _WebSearchGraph:
    async def astream(self, state, config=None, stream_mode="updates"):
        yield {
            "agent": {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "tavily_search",
                                "args": {"query": "plan mode"},
                                "id": "call-1",
                            }
                        ],
                    )
                ]
            }
        }
        yield {
            "tools": {
                "messages": [
                    ToolMessage(
                        content="search result payload",
                        name="tavily_search",
                        tool_call_id="call-1",
                    )
                ]
            }
        }
        yield {"agent": {"messages": [AIMessage(content="final answer")]}}


@pytest.fixture
def make_session(monkeypatch, tmp_path):
    monkeypatch.setattr("agent.session.find_app_root", lambda: tmp_path)
    monkeypatch.setattr(
        "agent.session.build_graph",
        lambda _cfg, extra_tools=None, history_store=None: _AnswerGraph(),
    )

    def _make(window: int = 2, graph=None, web_search_tool_names=None):
        cfg = AgentConfig(persist_dir=str(tmp_path / "persist"))
        cfg.agent_recent_turns_window = window
        store = _FakeHistoryStore()
        session = ChatSession(
            cfg,
            history_store=store,
            web_search_tool_names=web_search_tool_names,
        )
        if graph is not None:
            session.graph = graph
        return session, store, tmp_path / cfg.plan_logs_dir

    return _make


def test_plan_writes_md_immediately(make_session):
    session, store, log_dir = make_session(window=2)
    asyncio.run(session.enter_plan_mode())

    for index in range(3):
        asyncio.run(session.turn(f"q{index}"))
        content = session.plan_log_path.read_text(encoding="utf-8")
        assert f"## Turn {index + 1}" in content
        assert f"q{index}" in content

    assert store.adds == []
    assert log_dir.exists()


def test_exit_plan_keeps_recent_turns_visible(make_session):
    session, _store, _log_dir = make_session(window=10)
    asyncio.run(session.enter_plan_mode())
    asyncio.run(session.turn("plan q1"))
    asyncio.run(session.turn("plan q2"))

    asyncio.run(session.exit_plan_mode())

    assert [turn.user_input for turn in session.recent_turns] == [
        "plan q1",
        "plan q2",
    ]
    prompt_contents = [message.content for message in session._prompt_history()]
    assert "plan q1" in prompt_contents
    assert "plan q2" in prompt_contents


def test_no_chroma_leak_after_exit(make_session):
    session, store, _log_dir = make_session(window=2)
    asyncio.run(session.enter_plan_mode())
    for index in range(5):
        asyncio.run(session.turn(f"plan {index}"))

    asyncio.run(session.exit_plan_mode())
    for index in range(5):
        asyncio.run(session.turn(f"normal {index}"))
    asyncio.run(session.flush_recent_turns())

    assert [turn.user_input for turn in store.adds] == [f"normal {index}" for index in range(5)]


def test_md_write_failure_aborts_turn(make_session, monkeypatch):
    session, store, _log_dir = make_session(window=2)
    asyncio.run(session.enter_plan_mode())

    def fail_append(_path, _block):
        raise OSError("disk full")

    monkeypatch.setattr(session, "_append_block_to_md", fail_append)

    with pytest.raises(OSError, match="disk full"):
        asyncio.run(session.turn("q"))

    assert session.recent_turns == []
    assert session._turn_counter == 0
    assert store.adds == []


def test_web_search_capture_in_md(make_session):
    session, _store, _log_dir = make_session(
        window=2,
        graph=_WebSearchGraph(),
        web_search_tool_names={"tavily_search"},
    )
    asyncio.run(session.enter_plan_mode())
    asyncio.run(session.turn("search for plan mode"))

    content = session.plan_log_path.read_text(encoding="utf-8")
    assert "### Web search: tavily_search" in content
    assert '"query": "plan mode"' in content
    assert "search result payload" in content


def test_unknown_persist_target_raises(make_session):
    session, _store, _log_dir = make_session(window=2)
    turn = TurnRecord(
        user_input="q",
        assistant_output="a",
        turn_id=1,
        persist_target="mystery",
    )

    with pytest.raises(ValueError, match="unknown persist_target"):
        asyncio.run(session._store_turn(turn))
