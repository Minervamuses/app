"""Multi-turn conversational session for the KMS agent."""

from langchain_core.messages import HumanMessage, SystemMessage

from agent.config import AgentConfig
from agent.graph import build_graph
from agent.history import (
    extract_tool_calls,
    format_tool_counts,
)
from agent.llm.openrouter import OpenRouterLLM
from agent.memory import (
    TurnRecord,
    assemble_prompt_history,
    compact_turns,
)

SYSTEM_PROMPT = """You are a research assistant with access to four tool families.

Local knowledge base tools (always available):

1. **rag_explore** — Discover what's in the indexed knowledge base: categories, tags, date ranges, folder summaries.
   Use this first when you're unsure what the knowledge base contains.

2. **rag_search** — Semantic search with optional filters (folder_prefix, category, file_type, date range).
   Use specific queries. You can search multiple times with different queries or filters.

3. **rag_get_context** — Expand a search result by retrieving surrounding chunks from the same file.
   Use when a result looks relevant but you need more context.

Conversation history tool (always available):

4. **recall_history** — Search prior parts of this conversation that have aged out of the visible context. Each user prompt and each assistant response is stored as a separate entry; results carry role, turn_id, and timestamp.
   Use when the user references earlier content that you cannot see in the current prompt.
   Do NOT call this for content already visible in the current conversation.
   Do NOT use this as a substitute for rag_search on general knowledge questions.

Web Search MCP tools (available only when configured):
- Use for current external information, general web discovery, or topics unlikely to exist in the local KB.

GitHub MCP tools (available only when configured):
- Use for remote GitHub state: repository content not in the local KB, pull requests, issues, Actions runs, code search across GitHub.
- Do NOT use GitHub MCP as a substitute for local git shell operations (clone, pull, rebase, commit). Those belong to the user's terminal, not to you.

Tool selection policy:
- Questions about the indexed project or research notes → prefer `rag_explore` / `rag_search` / `rag_get_context`.
- Questions about earlier in this same conversation that are no longer visible → prefer `recall_history`.
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
        summarize_fn=None,
        progress_cb=None,
    ):
        self.config = config
        self.graph = build_graph(config, extra_tools=extra_tools)
        self.recursion_limit = recursion_limit

        self.system_prompt_message = SystemMessage(content=system_prompt)
        self.rolling_summary_text: str | None = None
        self.recent_turns: list[TurnRecord] = []
        self.completed_turn_count: int = 0

        self.turn_logs: list[dict] = []
        self.last_tool_calls: list[dict] = []

        self._summarize_fn = summarize_fn
        self._progress_cb = progress_cb

    def _get_summarize_fn(self):
        if self._summarize_fn is not None:
            return self._summarize_fn
        model_name = self.config.agent_compaction_model or self.config.llm_model
        llm = OpenRouterLLM(model_name=model_name, config=self.config)
        max_tokens = self.config.agent_compaction_max_tokens

        def _summarize(prompt: str) -> str:
            return llm.invoke(prompt, max_tokens=max_tokens, temperature=0.2)

        self._summarize_fn = _summarize
        return self._summarize_fn

    def _prompt_history(self) -> list:
        return assemble_prompt_history(
            self.system_prompt_message,
            self.rolling_summary_text,
            self.recent_turns,
        )

    def _maybe_compact(self) -> None:
        threshold = self.config.agent_turns_per_compaction
        if threshold <= 0 or len(self.recent_turns) < threshold:
            return
        block = self.recent_turns[:threshold]
        try:
            new_summary = compact_turns(
                self.rolling_summary_text,
                block,
                summarize=self._get_summarize_fn(),
            )
        except Exception:
            return
        self.rolling_summary_text = new_summary
        self.recent_turns = self.recent_turns[threshold:]

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

        self.recent_turns.append(TurnRecord(user_input=user_input, assistant_output=answer))
        self.completed_turn_count += 1
        self._maybe_compact()

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
        summarize_fn=None,
        load_mcp: bool = True,
        progress_cb=None,
    ) -> "ChatSession":
        """Async factory that loads MCP tools (if enabled) before graph construction.

        MCP tool loading is async; turn processing stays synchronous once the
        session is built.
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
            summarize_fn=summarize_fn,
            progress_cb=progress_cb,
        )
