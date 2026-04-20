"""Turn-aware conversation memory with rolling compaction.

Replaces raw message-count trimming with a three-layer model:

1. Fixed system prompt (owned by :class:`ChatSession`).
2. Single rolling compact summary (``rolling_summary_text``).
3. Recent raw turns since the last compaction (``recent_turns``).

Every ``agent_turns_per_compaction`` completed turns, the oldest block is
summarized into ``rolling_summary_text`` and dropped from ``recent_turns``.
Tool payloads are intentionally never carried into long-term memory.
"""

from dataclasses import dataclass
from typing import Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

COMPACTION_SYSTEM_PROMPT = """You are compacting an older block of a chat between a user and an assistant that queries a local knowledge base.

Produce a concise running memory that the assistant can rely on to stay coherent in future turns.

Keep:
- user goals, preferences, and accepted assumptions
- important factual findings
- decisions already made
- unresolved questions
- references to relevant files, modules, commands, or IDs needed for continuity
- meaningful tool outcomes that shaped answers (NOT raw tool payloads)

Drop:
- greetings and acknowledgements
- repeated failed searches unless they materially constrain future work
- redundant wording
- raw large tool outputs

Write tight prose or short bullet lines. Do not invent facts."""

COMPACTION_MEMORY_HEADER = "Compact memory of earlier conversation:"


@dataclass
class TurnRecord:
    """One completed turn, reduced to user input + final assistant answer."""

    user_input: str
    assistant_output: str

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


def build_summary_message(rolling_summary_text: str | None) -> SystemMessage | None:
    """Wrap the rolling summary as a single SystemMessage, or return None."""
    if not rolling_summary_text:
        return None
    return SystemMessage(
        content=f"{COMPACTION_MEMORY_HEADER}\n{rolling_summary_text}"
    )


def assemble_prompt_history(
    system_prompt: SystemMessage,
    rolling_summary_text: str | None,
    recent_turns: list[TurnRecord],
) -> list[BaseMessage]:
    """Build the base long-term prompt history (turn-aware, no same-turn pruning)."""
    messages: list[BaseMessage] = [system_prompt]
    summary_msg = build_summary_message(rolling_summary_text)
    if summary_msg is not None:
        messages.append(summary_msg)
    messages.extend(render_turns(recent_turns))
    return messages


def build_compaction_prompt(
    previous_summary: str | None,
    turns_to_compact: list[TurnRecord],
) -> str:
    """Build a single prompt-string asking the LLM to produce a new rolling summary."""
    lines: list[str] = [COMPACTION_SYSTEM_PROMPT, ""]
    if previous_summary:
        lines.append("Previous compact memory:")
        lines.append(previous_summary)
        lines.append("")
    lines.append("New turns to absorb:")
    for idx, turn in enumerate(turns_to_compact, start=1):
        lines.append(f"--- Turn {idx} ---")
        lines.append(f"User: {turn.user_input}")
        lines.append(f"Assistant: {turn.assistant_output}")
    lines.append("")
    lines.append(
        "Write the updated compact memory. Integrate new information with the "
        "previous memory. Do not repeat this instruction in the output."
    )
    return "\n".join(lines)


def compact_turns(
    previous_summary: str | None,
    turns_to_compact: list[TurnRecord],
    summarize: Callable[[str], str],
) -> str:
    """Run the supplied summarize() callable and return the new rolling summary text."""
    prompt = build_compaction_prompt(previous_summary, turns_to_compact)
    summary = summarize(prompt).strip()
    return summary or (previous_summary or "")
