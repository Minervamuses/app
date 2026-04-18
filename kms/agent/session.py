"""Multi-turn conversational session for the KMS agent."""

from langchain_core.messages import HumanMessage, SystemMessage

from kms.agent.graph import build_graph
from kms.agent.history import build_compact_turn, extract_tool_calls, format_tool_counts, trim_message_history
from kms.config import KMSConfig

SYSTEM_PROMPT = """You are a knowledge base assistant. You help users find information stored in an indexed repository.

You have three tools:

1. **explore** ??Discover what's in the knowledge base: categories, tags, date ranges, folder summaries.
   Use this first when you're unsure what the knowledge base contains.

2. **search** ??Semantic search with optional filters (category, file_type, date range).
   Use specific queries. You can search multiple times with different queries or filters.

3. **get_context** ??Expand a search result by retrieving surrounding chunks from the same file.
   Use when a result looks relevant but you need more context.

Workflow:
- If the question is vague or you don't know the structure of the knowledge base, start with explore.
- Use search with appropriate filters based on what you learned from explore.
- Use get_context if you need to see more around a promising result.
- After 1-3 searches, synthesize your answer. Don't keep searching for perfection.
- Do NOT make up information. Only answer based on tool results or your conversation with the user."""

DEFAULT_RECURSION_LIMIT = 16


class ChatSession:
    """Multi-turn conversational retrieval session backed by LangGraph."""

    def __init__(self, config: KMSConfig, recursion_limit: int = DEFAULT_RECURSION_LIMIT):
        self.config = config
        self.graph = build_graph(config)
        self.recursion_limit = recursion_limit
        self.history = [SystemMessage(content=SYSTEM_PROMPT)]
        self.turn_logs: list[dict] = []
        self.last_tool_calls: list[dict] = []

    def _run_turn(self, user_input: str) -> tuple[str, list[dict]]:
        """Process one turn and return the final answer plus tool-call trace."""
        input_messages = [*self.history, HumanMessage(content=user_input)]
        result = self.graph.invoke(
            {"messages": input_messages},
            config={"recursion_limit": self.recursion_limit},
        )
        new_messages = result["messages"][len(input_messages):]
        tool_calls = extract_tool_calls(new_messages)
        answer = result["messages"][-1].content or ""

        self.last_tool_calls = tool_calls
        self.turn_logs.append({
            "user_input": user_input,
            "tool_calls": tool_calls,
            "tool_counts": format_tool_counts(tool_calls),
        })

        self.history = trim_message_history(
            self.history + build_compact_turn(user_input, answer),
            max_messages=self.config.agent_max_messages,
        )
        return answer, tool_calls

    def turn(self, user_input: str) -> str:
        """Process one conversation turn. Returns the final text response."""
        answer, _tool_calls = self._run_turn(user_input)
        return answer

    def turn_with_trace(self, user_input: str) -> tuple[str, list[dict]]:
        """Process one turn and return the answer plus normalized tool trace."""
        return self._run_turn(user_input)
