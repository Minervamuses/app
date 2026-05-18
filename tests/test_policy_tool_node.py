"""Tests for skill-aware tool policy enforcement."""

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph

from agent.policy_tool_node import PolicyToolNode
from agent.state import AgentState


@tool("echo")
def _echo(text: str) -> str:
    """Echo text."""
    return text


@tool("bash")
def _bash(command: str) -> str:
    """Run shell."""
    return command


def _tool_call(name: str, call_id: str, args: dict) -> dict:
    return {
        "name": name,
        "args": args,
        "id": call_id,
        "type": "tool_call",
    }


def _invoke_policy_node(state: dict):
    graph = StateGraph(AgentState)
    graph.add_node("tools", PolicyToolNode([_echo, _bash]))
    graph.add_edge(START, "tools")
    return graph.compile().invoke(state)


def test_policy_tool_node_empty_policy_passthrough():
    ai_message = AIMessage(
        content="",
        tool_calls=[_tool_call("echo", "call-1", {"text": "ok"})],
    )

    result = _invoke_policy_node({"messages": [ai_message]})

    message = result["messages"][-1]
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "call-1"
    assert message.content == "ok"


def test_policy_tool_node_denies_matching_tool_with_same_call_id():
    ai_message = AIMessage(
        content="",
        tool_calls=[_tool_call("bash", "call-1", {"command": "ls"})],
    )

    result = _invoke_policy_node({
        "messages": [ai_message],
        "allowed_tools": ["echo"],
        "denied_tools": ["bash"],
        "tool_policy_active": True,
    })

    message = result["messages"][-1]
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "call-1"
    assert message.content == "Tool error: denied by active skill policy: bash"
    assert message.status == "error"


def test_policy_tool_node_handles_mixed_allowed_and_denied_calls():
    ai_message = AIMessage(
        content="",
        tool_calls=[
            _tool_call("echo", "call-1", {"text": "ok"}),
            _tool_call("bash", "call-2", {"command": "ls"}),
        ],
    )

    result = _invoke_policy_node({
        "messages": [ai_message],
        "allowed_tools": ["echo"],
        "denied_tools": ["bash"],
        "tool_policy_active": True,
    })

    messages = result["messages"][-2:]
    assert [message.tool_call_id for message in messages] == ["call-1", "call-2"]
    assert messages[0].content == "ok"
    assert messages[1].content == "Tool error: denied by active skill policy: bash"
    assert messages[1].status == "error"


def test_policy_tool_node_denies_unlisted_tool_when_allowlist_is_present():
    ai_message = AIMessage(
        content="",
        tool_calls=[_tool_call("bash", "call-1", {"command": "ls"})],
    )

    result = _invoke_policy_node({
        "messages": [ai_message],
        "allowed_tools": ["echo"],
        "denied_tools": [],
        "tool_policy_active": True,
    })

    message = result["messages"][-1]
    assert message.tool_call_id == "call-1"
    assert message.content == "Tool error: denied by active skill policy: bash"
    assert message.status == "error"


def test_policy_tool_node_active_empty_policy_denies_all_tools():
    ai_message = AIMessage(
        content="",
        tool_calls=[_tool_call("echo", "call-1", {"text": "ok"})],
    )

    result = _invoke_policy_node({
        "messages": [ai_message],
        "allowed_tools": [],
        "denied_tools": [],
        "tool_policy_active": True,
    })

    message = result["messages"][-1]
    assert message.tool_call_id == "call-1"
    assert message.content == "Tool error: denied by active skill policy: echo"
    assert message.status == "error"
