"""Utilities for pruning agent context and recording compact turn history."""

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
    """Prepare a bounded prompt for the next agent step.

    Rules:
    - Never trim non-tool conversation history supplied by ChatSession; until a
      turn is successfully evicted into history_rag, it must remain visible.
    - Keep only the most recent ``max_tool_interactions`` tool interactions
      inside the current turn.
    - Add a compact truncation note when earlier tool activity was removed.
    - ``max_messages`` is kept as a compatibility knob for callers, but this
      function no longer enforces it by dropping unevicted conversation turns.
    """
    tool_call_names: dict[str, str] = {}
    tool_call_groups: list[set[str]] = []
    total_tool_interactions = 0
    recent_tool_call_ids: list[str] = []

    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            group: set[str] = set()
            for tool_call in message.tool_calls:
                if isinstance(tool_call, dict):
                    tool_id = tool_call.get("id")
                    tool_name = tool_call.get("name", "unknown")
                else:
                    tool_id = getattr(tool_call, "id", None)
                    tool_name = getattr(tool_call, "name", "unknown")
                if tool_id:
                    tool_call_names[tool_id] = tool_name
                    group.add(tool_id)
            if group:
                tool_call_groups.append(group)
        elif isinstance(message, ToolMessage):
            total_tool_interactions += 1

    if max_tool_interactions > 0:
        for message in reversed(messages):
            if not isinstance(message, ToolMessage):
                continue
            tool_call_id = getattr(message, "tool_call_id", None)
            if not tool_call_id:
                continue
            recent_tool_call_ids.append(tool_call_id)
            if len(recent_tool_call_ids) >= max_tool_interactions:
                break

    recent_ids = set(recent_tool_call_ids)
    kept_ids: set[str] = set()
    for group in tool_call_groups:
        if group & recent_ids:
            # Keep the full assistant tool-call group to preserve the chat
            # protocol: every kept assistant tool call should have its tool
            # response in the prompt.
            kept_ids.update(group)

    pruned_messages: list[BaseMessage] = []
    dropped_counts: Counter[str] = Counter()

    for message in messages:
        if isinstance(message, ToolMessage):
            tool_call_id = getattr(message, "tool_call_id", None)
            if tool_call_id in kept_ids:
                pruned_messages.append(message)
            else:
                dropped_counts[tool_call_names.get(tool_call_id, "unknown")] += 1
            continue

        if isinstance(message, AIMessage) and message.tool_calls:
            tool_ids = set()
            for tool_call in message.tool_calls:
                if isinstance(tool_call, dict):
                    tool_ids.add(tool_call.get("id"))
                else:
                    tool_ids.add(getattr(tool_call, "id", None))
            if tool_ids & kept_ids:
                pruned_messages.append(message)
            continue

        pruned_messages.append(message)

    if total_tool_interactions > max_tool_interactions and dropped_counts:
        dropped_summary = ", ".join(
            f"{name} x{dropped_counts[name]}"
            for name in sorted(dropped_counts)
        )
        note = SystemMessage(
            content=(
                "Context note: earlier tool results were truncated to keep this turn bounded. "
                f"Already-used tools outside the current window: {dropped_summary}. "
                "Prefer synthesizing an answer instead of repeating similar calls."
            )
        )
        system_prompt = [msg for msg in pruned_messages if isinstance(msg, SystemMessage)]
        non_system = [msg for msg in pruned_messages if not isinstance(msg, SystemMessage)]
        pruned_messages = system_prompt + [note] + non_system

    return pruned_messages
