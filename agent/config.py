"""Configuration for the agent host.

Extends :class:`rag.config.RAGConfig` with fields that only the agent layer
(conversation loop, evaluation, compaction) needs to know about. Kept here
so rag remains a framework-neutral library.
"""

from dataclasses import dataclass

from rag.config import RAGConfig


@dataclass
class AgentConfig(RAGConfig):
    """Runtime config for the LangGraph agent, eval harness, and CLI."""

    # Main chat LLM used by the agent's LangGraph loop.
    llm_model: str = "deepseek/deepseek-v4-pro"
    llm_max_tokens: int = 4096
    # Retry count delegated through LangChain/OpenAI clients. Single source of
    # truth for runtime and evaluation OpenRouter chat models.
    llm_max_retries: int = 10

    # Evaluation LLMs
    gen_llm_model: str = "google/gemini-3.1-pro-preview"
    judge_llm_model: str = "openai/gpt-5.2"
    filter_llm_model: str = "llama3.1:8b"

    # Extended thinking role models.
    thinking_reviewer_model: str = "anthropic/claude-haiku-4.5"
    thinking_reviewer_max_tokens: int = 4096
    thinking_rewrite_model: str = "openai/gpt-5-mini"
    thinking_repair_model: str = "openai/gpt-5-mini"

    # Extended thinking context caps.
    thinking_tool_trace_chars: int = 500
    thinking_tool_trace_total_chars: int = 4000
    thinking_rewrite_visible_chars: int = 2000
    thinking_rewrite_skill_chars: int = 4000

    # Extended-thinking fusion candidate panel (replaces the single writer stage
    # of /thinking extended). Empty proposer/aggregator slots resolve to existing
    # role models in agent.llm.thinking; no new slash command or mode is added.
    thinking_fusion_proposer_models: tuple[str, ...] = ()
    thinking_fusion_aggregator_model: str = ""
    thinking_fusion_aggregator_max_tokens: int = 4096
    thinking_fusion_proposer_tool_interactions: int = 2
    thinking_fusion_candidate_timeout_seconds: float = 180.0
    thinking_fusion_quorum: int = 2
    thinking_fusion_allow_side_effect_tools: bool = False

    # Agent context controls (same-turn bounds)
    agent_max_messages: int = 20
    # Per-turn hard cap on tool interactions (enforced in graph.agent_node and
    # graph._cap_tool_calls). Default 4 is data-backed: in the C1 dev routing
    # run, every normal eligible case fit within 0-4 tool calls, while the only
    # runaway (the embedding case) made 8 RAG calls because the topic was absent
    # from the indexed KB and the agent lacked give-up discipline -- not because
    # 4 was too low. So the fix is graceful give-up (prompt + eval), not a larger
    # cap. Scope is per turn, not per conversation.
    agent_max_tool_interactions: int = 4

    # Long-term memory: keep this many most-recent turns in the prompt;
    # evicted turns spill into the chat_history vector store.
    agent_recent_turns_window: int = 10

    # Plan mode markdown logs. Relative to the app project root.
    plan_logs_dir: str = "plan_logs"

    # Soft cap on a single ToolMessage payload written to a plan log
    # (UTF-8 chars). Truncation only affects the markdown copy; the LLM
    # still receives the full ToolMessage in its context window.
    plan_log_max_tool_chars: int = 65536

    # Optional local skills directory. When unset, defaults to `<repo>/skills`.
    skills_dir: str | None = None

    # Skill runtime controls.
    skill_validation_enabled: bool = True
    skill_max_validation_retries: int = 1
    skill_capability_map_path: str | None = None
    skill_max_pinned_reference_chars: int = 65536
    skill_max_total_skill_context_chars: int = 200000
