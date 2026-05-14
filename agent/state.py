"""Agent state definition for LangGraph."""

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """State passed between graph nodes. Messages accumulate via add_messages reducer."""

    messages: Annotated[list[BaseMessage], add_messages]
    active_skill: str | None
    skill_root: str | None
    skill_instructions: str | None
    loaded_references: dict[str, str]
    task_mode: str | None
    allowed_tools: list[str]
    denied_tools: list[str]
    validation_errors: list[str]
    validation_attempts: int
