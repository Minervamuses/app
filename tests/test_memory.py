"""Tests for turn-aware memory and rolling compaction behavior."""

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.memory import (
    COMPACTION_MEMORY_HEADER,
    TurnRecord,
    assemble_prompt_history,
    build_compaction_prompt,
    build_summary_message,
    compact_turns,
    render_turns,
)


def test_render_turns_flattens_in_order():
    turns = [
        TurnRecord(user_input="q1", assistant_output="a1"),
        TurnRecord(user_input="q2", assistant_output=""),
        TurnRecord(user_input="q3", assistant_output="a3"),
    ]
    msgs = render_turns(turns)
    assert [type(m) for m in msgs] == [HumanMessage, AIMessage, HumanMessage, HumanMessage, AIMessage]
    assert [m.content for m in msgs] == ["q1", "a1", "q2", "q3", "a3"]


def test_build_summary_message_none_when_empty():
    assert build_summary_message(None) is None
    assert build_summary_message("") is None


def test_build_summary_message_wraps_text():
    msg = build_summary_message("user wants X")
    assert isinstance(msg, SystemMessage)
    assert msg.content.startswith(COMPACTION_MEMORY_HEADER)
    assert "user wants X" in msg.content


def test_assemble_prompt_history_skips_summary_when_missing():
    sys_msg = SystemMessage(content="SYS")
    turns = [TurnRecord(user_input="q", assistant_output="a")]
    msgs = assemble_prompt_history(sys_msg, None, turns)
    assert msgs[0] is sys_msg
    assert [type(m) for m in msgs] == [SystemMessage, HumanMessage, AIMessage]


def test_assemble_prompt_history_includes_summary_between_system_and_turns():
    sys_msg = SystemMessage(content="SYS")
    turns = [TurnRecord(user_input="q", assistant_output="a")]
    msgs = assemble_prompt_history(sys_msg, "memory", turns)
    assert [type(m) for m in msgs] == [SystemMessage, SystemMessage, HumanMessage, AIMessage]
    assert msgs[1].content.startswith(COMPACTION_MEMORY_HEADER)


def test_build_compaction_prompt_includes_previous_summary_and_turns():
    prompt = build_compaction_prompt(
        previous_summary="old memory",
        turns_to_compact=[
            TurnRecord(user_input="hello", assistant_output="hi"),
            TurnRecord(user_input="what is X?", assistant_output="X is Y"),
        ],
    )
    assert "old memory" in prompt
    assert "hello" in prompt
    assert "what is X?" in prompt
    assert "X is Y" in prompt


def test_compact_turns_falls_back_to_previous_on_empty_response():
    turns = [TurnRecord(user_input="q", assistant_output="a")]
    result = compact_turns("prev", turns, summarize=lambda _p: "   ")
    assert result == "prev"


def test_compact_turns_returns_new_summary():
    turns = [TurnRecord(user_input="q", assistant_output="a")]
    result = compact_turns(None, turns, summarize=lambda _p: "new memory")
    assert result == "new memory"


def _make_session(monkeypatch, tmp_path, turns_per_compaction=3):
    from agent.session import ChatSession
    from agent.config import AgentConfig

    class DummyGraph:
        def __init__(self):
            self.calls: list[list] = []

        def invoke(self, state, config=None):
            self.calls.append(state["messages"])
            return {"messages": [*state["messages"], AIMessage(content="ok")]}

    dummy_graph = DummyGraph()
    monkeypatch.setattr("agent.session.build_graph", lambda _cfg, extra_tools=None: dummy_graph)

    cfg = AgentConfig(persist_dir=str(tmp_path))
    cfg.agent_turns_per_compaction = turns_per_compaction

    summarize_calls: list[str] = []

    def fake_summarize(prompt: str) -> str:
        summarize_calls.append(prompt)
        return f"summary-{len(summarize_calls)}"

    session = ChatSession(cfg, summarize_fn=fake_summarize)
    return session, dummy_graph, summarize_calls


def test_compaction_triggers_after_block(monkeypatch, tmp_path):
    session, dummy_graph, summarize_calls = _make_session(
        monkeypatch, tmp_path, turns_per_compaction=3
    )

    for i in range(3):
        session.turn(f"q{i}")

    assert len(summarize_calls) == 1
    assert session.rolling_summary_text == "summary-1"
    assert session.recent_turns == []
    assert session.completed_turn_count == 3


def test_compaction_is_block_based_not_message_based(monkeypatch, tmp_path):
    session, dummy_graph, summarize_calls = _make_session(
        monkeypatch, tmp_path, turns_per_compaction=3
    )

    for i in range(7):
        session.turn(f"q{i}")

    # 7 turns -> 2 compactions (at turn 3 and turn 6), 1 raw turn remaining
    assert len(summarize_calls) == 2
    assert len(session.recent_turns) == 1
    assert session.recent_turns[0].user_input == "q6"
    assert session.rolling_summary_text == "summary-2"


def test_prompt_history_after_compaction_has_single_summary(monkeypatch, tmp_path):
    session, dummy_graph, summarize_calls = _make_session(
        monkeypatch, tmp_path, turns_per_compaction=3
    )

    for i in range(6):
        session.turn(f"q{i}")

    # Two compactions, but still exactly one summary SystemMessage kept.
    msgs = session._prompt_history()
    system_msgs = [m for m in msgs if isinstance(m, SystemMessage)]
    # One fixed system prompt + one rolling summary system message.
    assert len(system_msgs) == 2
    assert system_msgs[1].content.startswith(COMPACTION_MEMORY_HEADER)


def test_compaction_failure_does_not_drop_turns(monkeypatch, tmp_path):
    from agent.session import ChatSession
    from agent.config import AgentConfig

    class DummyGraph:
        def invoke(self, state, config=None):
            return {"messages": [*state["messages"], AIMessage(content="ok")]}

    monkeypatch.setattr("agent.session.build_graph", lambda _cfg, extra_tools=None: DummyGraph())

    cfg = AgentConfig(persist_dir=str(tmp_path))
    cfg.agent_turns_per_compaction = 2

    def raising_summarize(_prompt: str) -> str:
        raise RuntimeError("llm unavailable")

    session = ChatSession(cfg, summarize_fn=raising_summarize)

    for i in range(2):
        session.turn(f"q{i}")

    assert session.rolling_summary_text is None
    assert len(session.recent_turns) == 2
