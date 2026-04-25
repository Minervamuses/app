"""Tests for agent.history prompt pruning."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.history import prepare_messages_for_agent


def _tool_call(name: str, call_id: str, args: dict | None = None) -> dict:
    return {"name": name, "args": args or {}, "id": call_id}


def test_prepare_messages_preserves_recent_turns_over_max_messages():
    messages = [SystemMessage(content="SYS")]
    for i in range(1, 11):
        messages.extend([
            HumanMessage(content=f"q{i}"),
            AIMessage(content=f"a{i}"),
        ])
    messages.append(HumanMessage(content="current"))

    prepared = prepare_messages_for_agent(
        messages,
        max_messages=20,
        max_tool_interactions=4,
    )

    contents = [msg.content for msg in prepared]
    assert len(prepared) == 22
    assert "q1" in contents
    assert "a1" in contents
    assert contents[-1] == "current"


def test_prepare_messages_prunes_tool_traffic_not_conversation_history():
    messages = [
        SystemMessage(content="SYS"),
        HumanMessage(content="q1"),
        AIMessage(content="a1"),
        HumanMessage(content="current"),
        AIMessage(content="", tool_calls=[_tool_call("rag_search", "call-1")]),
        ToolMessage(content="old result", name="rag_search", tool_call_id="call-1"),
        AIMessage(content="", tool_calls=[_tool_call("rag_search", "call-2")]),
        ToolMessage(content="new result", name="rag_search", tool_call_id="call-2"),
    ]

    prepared = prepare_messages_for_agent(
        messages,
        max_messages=4,
        max_tool_interactions=1,
    )

    contents = [msg.content for msg in prepared]
    assert "q1" in contents
    assert "a1" in contents
    assert "current" in contents
    assert "old result" not in contents
    assert "new result" in contents
    assert any(
        isinstance(msg, SystemMessage)
        and "earlier tool results were truncated" in msg.content
        for msg in prepared
    )


def test_prepare_messages_keeps_full_multi_tool_call_group():
    messages = [
        SystemMessage(content="SYS"),
        HumanMessage(content="current"),
        AIMessage(
            content="",
            tool_calls=[
                _tool_call("rag_search", "call-1"),
                _tool_call("rag_get_context", "call-2"),
            ],
        ),
        ToolMessage(content="search result", name="rag_search", tool_call_id="call-1"),
        ToolMessage(content="context result", name="rag_get_context", tool_call_id="call-2"),
    ]

    prepared = prepare_messages_for_agent(
        messages,
        max_messages=4,
        max_tool_interactions=1,
    )

    contents = [msg.content for msg in prepared]
    assert "search result" in contents
    assert "context result" in contents
