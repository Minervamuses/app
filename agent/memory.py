"""Turn-aware conversation memory.

Two-layer model:

1. Fixed system prompt (owned by :class:`ChatSession`).
2. Recent turns since session start (``recent_turns``).

When ``recent_turns`` exceeds ``agent_recent_turns_window``, the oldest
turn is evicted into the long-term ChromaDB chat-history store. There
is no LLM-driven compaction; spillover is preserved verbatim and
searchable via the ``recall_history`` tool.
"""

from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


@dataclass
class TurnRecord:
    """One completed turn, reduced to user input + final assistant answer."""

    user_input: str
    assistant_output: str
    turn_id: int = 0
    timestamp: str = ""

    def to_messages(self) -> list[BaseMessage]:
        msgs: list[BaseMessage] = [HumanMessage(content=self.user_input)]
        if self.assistant_output:
            msgs.append(AIMessage(content=self.assistant_output))
        return msgs


def render_turns(turns: list[TurnRecord]) -> list[BaseMessage]:
    """Flatten turn records into prompt-visible raw messages."""
    out: list[BaseMessage] = []
    for turn in turns:
        out.extend(turn.to_messages())
    return out


def assemble_prompt_history(
    system_prompt: SystemMessage,
    recent_turns: list[TurnRecord],
) -> list[BaseMessage]:
    """Build the long-term prompt history: system prompt followed by raw recent turns."""
    return [system_prompt, *render_turns(recent_turns)]
