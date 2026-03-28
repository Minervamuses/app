"""OpenRouter LLM provider."""

import os
import time

from openai import OpenAI, RateLimitError

from kms.config import KMSConfig
from kms.llm.base import BaseLLM


class OpenRouterLLM(BaseLLM):
    """LLM provider via OpenRouter API."""

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

    def invoke(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float | None = None,
    ) -> str:
        """Send a prompt to the LLM and return the response."""
        max_retries = 10
        delay = 10.0
        last_err: Exception | None = None

        for _attempt in range(max_retries):
            try:
                kwargs: dict = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                }
                if temperature is not None:
                    kwargs["temperature"] = temperature

                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content.strip()
            except RateLimitError as e:
                last_err = e
                time.sleep(delay)
                delay *= 2

        raise RuntimeError(f"Failed after {max_retries} retries") from last_err
