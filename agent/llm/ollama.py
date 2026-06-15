"""Ollama chat-model factory for local eval/filter tasks."""

from typing import Any

from langchain_ollama import ChatOllama

from agent.config import AgentConfig


def get_ollama_chat_model(
    config: AgentConfig | None = None,
    *,
    model_name: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> ChatOllama:
    """Return a LangChain chat model backed by local Ollama."""
    config = config or AgentConfig()
    kwargs: dict[str, Any] = {
        "model": model_name or config.filter_llm_model,
    }
    if max_tokens is not None:
        kwargs["num_predict"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatOllama(**kwargs)
