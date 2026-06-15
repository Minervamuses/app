"""OpenRouter LLM provider for the agent layer."""

import os
from typing import Any

from langchain_openai import ChatOpenAI
from openai import OpenAI

from agent.config import AgentConfig

from agent.llm.base import BaseLLM


def get_chat_model(config: AgentConfig | None = None) -> ChatOpenAI:
    """Return a ChatOpenAI pointed at OpenRouter for use with LangGraph.

    Args:
        config: KMS configuration. Uses default if None.

    Returns:
        ChatOpenAI instance configured for OpenRouter.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    config = config or AgentConfig()
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=config.llm_model,
        # Reasoning models (e.g. deepseek-v4-pro) reject or ignore temperature;
        # uncomment if switching back to a non-reasoning chat model.
        # temperature=0.3,
        max_tokens=config.llm_max_tokens,
        max_retries=config.llm_max_retries,
    )


class OpenRouterLLM(BaseLLM):
    """LLM provider via OpenRouter API for prompt→text calls in eval/agent code.

    Rate-limit/retry handling is delegated to the official OpenAI client via its
    ``max_retries`` setting; there is no local backoff loop.
    """

    def __init__(self, model_name: str | None = None, config: AgentConfig | None = None):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        config = config or AgentConfig()
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_retries=config.llm_max_retries,
        )
        self.model = model_name or config.llm_model

    def invoke(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        """Send a prompt to the LLM and return the response."""
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if response_format is not None:
            kwargs["response_format"] = response_format
        if extra_body is not None:
            kwargs["extra_body"] = extra_body

        resp = self.client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content
        return content.strip() if content else ""
