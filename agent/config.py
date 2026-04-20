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
    llm_model: str = "z-ai/glm-5"

    # Evaluation LLMs
    gen_llm_model: str = "google/gemini-3.1-pro-preview"
    judge_llm_model: str = "openai/gpt-5.2"
    filter_llm_model: str = "llama3.1:8b"

    # Agent context controls (same-turn bounds)
    agent_max_messages: int = 20
    agent_max_tool_interactions: int = 4

    # Turn compaction (every N completed turns, collapse oldest block into a
    # single rolling summary). agent_compaction_model=None falls back to
    # llm_model at runtime.
    agent_turns_per_compaction: int = 10
    agent_compaction_model: str | None = None
    agent_compaction_max_tokens: int = 800
