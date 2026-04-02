"""OpenRouter LLM provider."""

import os
import time

from langchain_openai import ChatOpenAI
from openai import OpenAI, RateLimitError

from kms.config import KMSConfig
from kms.llm.base import BaseLLM


def get_chat_model(config: KMSConfig | None = None) -> ChatOpenAI:
    """Return a ChatOpenAI pointed at OpenRouter for use with LangGraph.

    Args:
        config: KMS configuration. Uses default if None.

    Returns:
        ChatOpenAI instance configured for OpenRouter.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    config = config or KMSConfig()
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=config.llm_model,
        temperature=0.3,
        max_tokens=1024,
        max_retries=10,
    )


class OpenRouterLLM(BaseLLM):
    """LLM provider via OpenRouter API. Used by LLMTagger for simple prompt→text calls."""

    MAX_RETRIES = 10
    INITIAL_DELAY = 10.0

    def __init__(self, model_name: str | None = None, config: KMSConfig | None = None):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        config = config or KMSConfig()
        self.model = model_name or config.llm_model

    def _call_with_retry(self, **kwargs):
        """Call the OpenAI API with exponential backoff on rate limits."""
        delay = self.INITIAL_DELAY
        last_err: Exception | None = None

        for _attempt in range(self.MAX_RETRIES):
            try:
                return self.client.chat.completions.create(**kwargs)
            except RateLimitError as e:
                last_err = e
                time.sleep(delay)
                delay *= 2

        raise RuntimeError(f"Failed after {self.MAX_RETRIES} retries") from last_err

    def invoke(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float | None = None,
    ) -> str:
        """Send a prompt to the LLM and return the response."""
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        resp = self._call_with_retry(**kwargs)
        return resp.choices[0].message.content.strip()
