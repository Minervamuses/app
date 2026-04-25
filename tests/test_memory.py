"""Tests for turn-aware memory primitives."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.memory import TurnRecord, assemble_prompt_history, render_turns


def test_render_turns_flattens_in_order():
    turns = [
        TurnRecord(user_input="q1", assistant_output="a1"),
        TurnRecord(user_input="q2", assistant_output=""),
        TurnRecord(user_input="q3", assistant_output="a3"),
    ]
    msgs = render_turns(turns)
    assert [type(m) for m in msgs] == [HumanMessage, AIMessage, HumanMessage, HumanMessage, AIMessage]
    assert [m.content for m in msgs] == ["q1", "a1", "q2", "q3", "a3"]


def test_render_turns_skips_empty_assistant_output():
    turns = [TurnRecord(user_input="q", assistant_output="")]
    msgs = render_turns(turns)
    assert len(msgs) == 1
    assert isinstance(msgs[0], HumanMessage)


def test_assemble_prompt_history_emits_system_then_turns():
    sys_msg = SystemMessage(content="SYS")
    turns = [TurnRecord(user_input="q", assistant_output="a")]
    msgs = assemble_prompt_history(sys_msg, turns)
    assert msgs[0] is sys_msg
    assert [type(m) for m in msgs] == [SystemMessage, HumanMessage, AIMessage]


def test_assemble_prompt_history_with_empty_turns_returns_only_system():
    sys_msg = SystemMessage(content="SYS")
    msgs = assemble_prompt_history(sys_msg, [])
    assert msgs == [sys_msg]


def test_turn_record_carries_id_and_timestamp_metadata():
    turn = TurnRecord(
        user_input="hi",
        assistant_output="hello",
        turn_id=7,
        timestamp="2026-04-25T10:00:00+00:00",
    )
    assert turn.turn_id == 7
    assert turn.timestamp == "2026-04-25T10:00:00+00:00"
