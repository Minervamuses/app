"""OpenRouter LLM provider used internally by rag (tagger)."""

import os
import time
from typing import Any

from openai import OpenAI, RateLimitError

from rag.config import KMSConfig
from rag.llm.base import BaseLLM


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

        resp = self._call_with_retry(**kwargs)
        content = resp.choices[0].message.content
        return content.strip() if content else ""
