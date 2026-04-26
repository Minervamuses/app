"""Multi-turn conversational session for the agent."""

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage

from agent.config import AgentConfig
from agent.graph import build_graph
from agent.history import (
    extract_tool_calls,
    format_tool_counts,
)
from agent.history_rag import ChatHistoryStore, get_chat_history_store
from agent.memory import (
    TurnRecord,
    assemble_prompt_history,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research assistant with access to four tool families.

Local knowledge base tools (always available):

1. **rag_explore** — Discover what's in the indexed knowledge base: categories, tags, date ranges, folder summaries.
   Use this first when you're unsure what the knowledge base contains.

2. **rag_search** — Semantic search with optional filters (folder_prefix, category, file_type, date range).
   Use specific queries. You can search multiple times with different queries or filters.

3. **rag_get_context** — Expand a search result by retrieving surrounding chunks from the same file.
   Use when a result looks relevant but you need more context.

Conversation history tool (always available):

4. **recall_history** — Search persisted prior chat turns from this user, including older parts of the current session and previous CLI sessions. Each user prompt and each assistant response is stored as a separate entry; results carry role, turn_id, and timestamp.
   Use when the user references earlier chat content that you cannot see in the current prompt.
   Do NOT call this for content already visible in the current conversation.
   Do NOT use this as a substitute for rag_search on general knowledge questions.

Web Search MCP tools (available only when configured):
- Use for current external information, general web discovery, or topics unlikely to exist in the local KB.

GitHub MCP tools (available only when configured):
- Use for remote GitHub state: repository content not in the local KB, pull requests, issues, Actions runs, code search across GitHub.
- Do NOT use GitHub MCP as a substitute for local git shell operations (clone, pull, rebase, commit). Those belong to the user's terminal, not to you.

Tool selection policy:
- Questions about the indexed project or research notes → prefer `rag_explore` / `rag_search` / `rag_get_context`.
- Questions about earlier chat history that is no longer visible → prefer `recall_history`.
- Questions needing live external information → prefer Web Search MCP.
- Questions about remote GitHub repos, PRs, issues, or Actions → prefer GitHub MCP.
- If a tool family is not listed in the bound tools for this session, treat it as unavailable and fall back to what you have.

Workflow:
- If the question is vague or you don't know the structure of the knowledge base, start with rag_explore.
- Use rag_search with appropriate filters based on what you learned from rag_explore.
- Use rag_get_context if you need to see more around a promising result.
- After 1-3 rag_search calls, synthesize your answer. Don't keep searching for perfection.
- Do NOT make up information. Only answer based on tool results or your conversation with the user."""

DEFAULT_RECURSION_LIMIT = 32


class ChatSession:
    """Multi-turn conversational retrieval session backed by LangGraph."""

    def __init__(
        self,
        config: AgentConfig,
        recursion_limit: int = DEFAULT_RECURSION_LIMIT,
        system_prompt: str = SYSTEM_PROMPT,
        extra_tools: list | None = None,
        history_store: ChatHistoryStore | None = None,
        progress_cb=None,
    ):
        self.config = config
        self.recursion_limit = recursion_limit

        self.system_prompt_message = SystemMessage(content=system_prompt)
        self.recent_turns: list[TurnRecord] = []

        self.session_id = uuid.uuid4().hex
        self._turn_counter = 0
        self.history_store = history_store or get_chat_history_store(config)
        self.graph = build_graph(
            config,
            extra_tools=extra_tools,
            history_store=self.history_store,
        )

        self.turn_logs: list[dict] = []
        self.last_tool_calls: list[dict] = []

        self._progress_cb = progress_cb

    def _prompt_history(self) -> list:
        return assemble_prompt_history(
            self.system_prompt_message,
            self.recent_turns,
        )

    async def _store_turn(self, turn: TurnRecord) -> None:
        await asyncio.to_thread(
            self.history_store.add_turn,
            turn,
            session_id=self.session_id,
            turn_id=turn.turn_id,
            timestamp=turn.timestamp,
        )

    async def _evict_overflow(self) -> None:
        """Spill turns past the window into the long-term store. Log + keep on failure."""
        window = self.config.agent_recent_turns_window
        hard_cap = window * 3
        while len(self.recent_turns) > window:
            oldest = self.recent_turns[0]
            try:
                await self._store_turn(oldest)
            except Exception as exc:
                logger.warning(
                    "history_rag: eviction failed for turn %s (kept in recent_turns): %s",
                    oldest.turn_id, exc,
                )
                if len(self.recent_turns) > hard_cap:
                    logger.error(
                        "history_rag: hard cap %d reached; dropping oldest turn %s unrecorded",
                        hard_cap, oldest.turn_id,
                    )
                    self.recent_turns.pop(0)
                break  # don't retry within the same turn
            self.recent_turns.pop(0)

    async def flush_recent_turns(self) -> None:
        """Persist all prompt-visible turns before the session is discarded."""
        while self.recent_turns:
            oldest = self.recent_turns[0]
            try:
                await self._store_turn(oldest)
            except Exception as exc:
                logger.warning(
                    "history_rag: shutdown flush failed for turn %s (left in recent_turns): %s",
                    oldest.turn_id, exc,
                )
                break
            self.recent_turns.pop(0)

    async def _run_turn(self, user_input: str) -> tuple[str, list[dict]]:
        """Process one turn and return the final answer plus tool-call trace.

        Streams graph updates so a progress callback (if provided) can
        surface tool calls as they happen; payloads are not forwarded.
        """
        input_messages = [*self._prompt_history(), HumanMessage(content=user_input)]
        messages: list = list(input_messages)
        async for update in self.graph.astream(
            {"messages": input_messages},
            config={"recursion_limit": self.recursion_limit},
            stream_mode="updates",
        ):
            for node_name, delta in update.items():
                new_msgs = delta.get("messages", []) if isinstance(delta, dict) else []
                messages.extend(new_msgs)
                if self._progress_cb is not None:
                    self._progress_cb(node_name, new_msgs)
        new_messages = messages[len(input_messages):]
        tool_calls = extract_tool_calls(new_messages)
        answer = messages[-1].content if messages else ""
        answer = answer or ""

        self.last_tool_calls = tool_calls
        self.turn_logs.append({
            "user_input": user_input,
            "tool_calls": tool_calls,
            "tool_counts": format_tool_counts(tool_calls),
        })

        self._turn_counter += 1
        self.recent_turns.append(
            TurnRecord(
                user_input=user_input,
                assistant_output=answer,
                turn_id=self._turn_counter,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        )
        await self._evict_overflow()

        return answer, tool_calls

    async def turn(self, user_input: str) -> str:
        """Process one conversation turn. Returns the final text response."""
        answer, _tool_calls = await self._run_turn(user_input)
        return answer

    async def turn_with_trace(self, user_input: str) -> tuple[str, list[dict]]:
        """Process one turn and return the answer plus normalized tool trace."""
        return await self._run_turn(user_input)

    @classmethod
    async def create(
        cls,
        config: AgentConfig,
        recursion_limit: int = DEFAULT_RECURSION_LIMIT,
        system_prompt: str = SYSTEM_PROMPT,
        history_store: ChatHistoryStore | None = None,
        load_mcp: bool = True,
        progress_cb=None,
    ) -> "ChatSession":
        """Async factory that loads MCP tools (if enabled) before graph construction.

        MCP tool loading is async; turn processing stays asynchronous via
        graph.astream once the session is built.
        """
        extra_tools: list = []
        if load_mcp:
            from agent.mcp import load_mcp_tools

            try:
                extra_tools = await load_mcp_tools()
            except Exception:
                extra_tools = []
        return cls(
            config,
            recursion_limit=recursion_limit,
            system_prompt=system_prompt,
            extra_tools=extra_tools,
            history_store=history_store,
            progress_cb=progress_cb,
        )
