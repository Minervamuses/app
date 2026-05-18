"""ToolNode wrapper that enforces active skill tool policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode


class PolicyToolNode(ToolNode):
    """Delegate to ToolNode, denying calls outside the active skill policy."""

    def __init__(self, tools: list, **tool_node_kwargs: Any):
        super().__init__(tools, **tool_node_kwargs)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        decision = self._partition(input)
        if decision is None:
            return super().invoke(input, config=config, **kwargs)
        allowed_input, denied_messages, call_order = decision
        if allowed_input is None:
            return {"messages": denied_messages}
        allowed_result = super().invoke(allowed_input, config=config, **kwargs)
        return {"messages": _merge_tool_messages(call_order, denied_messages, allowed_result)}

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        decision = self._partition(input)
        if decision is None:
            return await super().ainvoke(input, config=config, **kwargs)
        allowed_input, denied_messages, call_order = decision
        if allowed_input is None:
            return {"messages": denied_messages}
        allowed_result = await super().ainvoke(allowed_input, config=config, **kwargs)
        return {"messages": _merge_tool_messages(call_order, denied_messages, allowed_result)}

    def _partition(self, input: Any):
        if not isinstance(input, Mapping):
            return None

        policy_active = bool(input.get("tool_policy_active"))
        if not policy_active:
            return None
        allowed = set(input.get("allowed_tools") or [])
        denied = set(input.get("denied_tools") or [])

        messages = input.get("messages") or []
        if not messages:
            return None
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return None

        allowed_calls: list[dict] = []
        denied_messages: list[ToolMessage] = []
        call_order: list[str] = []
        for call in last_message.tool_calls:
            name = call.get("name", "")
            call_id = call.get("id", "")
            call_order.append(call_id)
            if (
                name in denied
                or (allowed and name not in allowed)
                or (not allowed and not denied)
            ):
                denied_messages.append(ToolMessage(
                    content=f"Tool error: denied by active skill policy: {name}",
                    tool_call_id=call_id,
                    name=name,
                    status="error",
                ))
            else:
                allowed_calls.append(call)

        if not denied_messages:
            return None

        if not allowed_calls:
            return None, denied_messages, call_order

        filtered_message = AIMessage(
            content=last_message.content,
            additional_kwargs=last_message.additional_kwargs,
            response_metadata=last_message.response_metadata,
            tool_calls=allowed_calls,
        )
        filtered_input = dict(input)
        filtered_input["messages"] = [*messages[:-1], filtered_message]
        return filtered_input, denied_messages, call_order


def _merge_tool_messages(
    call_order: list[str],
    denied_messages: list[ToolMessage],
    allowed_result: Any,
) -> list[ToolMessage]:
    messages_by_id: dict[str, ToolMessage] = {
        message.tool_call_id: message
        for message in denied_messages
    }
    if isinstance(allowed_result, Mapping):
        for message in allowed_result.get("messages", []):
            if isinstance(message, ToolMessage):
                messages_by_id[message.tool_call_id] = message
    return [
        messages_by_id[call_id]
        for call_id in call_order
        if call_id in messages_by_id
    ]
