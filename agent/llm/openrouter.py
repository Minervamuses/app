"""OpenRouter chat-model factories for the agent layer."""

import os
from typing import Any

from langchain_openai import ChatOpenAI

from agent.config import AgentConfig


def get_chat_model(config: AgentConfig | None = None) -> ChatOpenAI:
    """Return a ChatOpenAI pointed at OpenRouter for use with LangGraph.

    Args:
        config: KMS configuration. Uses default if None.

    Returns:
        ChatOpenAI instance configured for OpenRouter.
    """
    config = config or AgentConfig()
    return get_openrouter_chat_model(
        config,
        max_tokens=config.llm_max_tokens,
    )


def get_openrouter_chat_model(
    config: AgentConfig | None = None,
    *,
    model_name: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    extra_body: dict[str, Any] | None = None,
) -> ChatOpenAI:
    """Return an OpenRouter-backed LangChain chat model.

    Core runtime, evaluation, and one-off prompt-to-text helpers all share this
    factory so the agent has one model access contract.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    config = config or AgentConfig()
    kwargs: dict[str, Any] = {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": api_key,
        "model": model_name or config.llm_model,
        "max_retries": config.llm_max_retries,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    return ChatOpenAI(**kwargs)
