"""Utilities for tool-call traces and same-turn context pruning."""

from collections import Counter

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage


def extract_tool_calls(messages: list[BaseMessage]) -> list[dict]:
    """Extract normalized tool-call records from AI messages."""
    calls: list[dict] = []
    for message in messages:
        if not isinstance(message, AIMessage) or not message.tool_calls:
            continue
        for tool_call in message.tool_calls:
            if isinstance(tool_call, dict):
                name = tool_call.get("name", "unknown")
                args = tool_call.get("args", {})
                tool_id = tool_call.get("id")
            else:
                name = getattr(tool_call, "name", "unknown")
                args = getattr(tool_call, "args", {}) or {}
                tool_id = getattr(tool_call, "id", None)
            calls.append({
                "id": tool_id,
                "name": name,
                "args": args,
            })
    return calls


def format_tool_counts(tool_calls: list[dict]) -> str:
    """Render compact per-tool counts for logs and notes."""
    counts = Counter(call["name"] for call in tool_calls if call.get("name"))
    if not counts:
        return ""
    return ", ".join(f"{name} x{counts[name]}" for name in sorted(counts))


def trim_message_history(messages: list[BaseMessage], max_messages: int) -> list[BaseMessage]:
    """Trim prompt-visible history to the most recent non-system messages."""
    system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
    other_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]

    available = max(max_messages - len(system_messages), 0)
    if available == 0:
        return system_messages[:max_messages]
    return system_messages + other_messages[-available:]


def prepare_messages_for_agent(
    messages: list[BaseMessage],
    max_messages: int,
    max_tool_interactions: int,
) -> list[BaseMessage]:
    """Return the current prompt messages unchanged.

    This used to keep only the most recent ``max_tool_interactions`` tool
    results within a turn. That hid the agent's own earlier tool results once a
    turn exceeded the window, so it could not tell it had already searched and
    kept re-searching (a model-agnostic runaway). The per-turn tool budget
    (``agent_max_tool_interactions``, enforced in ``graph.agent_node`` and
    ``_cap_tool_calls``) now bounds how many tool calls a turn can make, so the
    count is already small and a separate visible window is unnecessary — the
    agent must see every tool result it produced this turn.

    ``max_messages`` / ``max_tool_interactions`` are retained for call-site
    compatibility but no longer drop messages.
    """
    del max_messages, max_tool_interactions
    return list(messages)
