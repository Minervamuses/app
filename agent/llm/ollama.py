"""Ollama LLM provider for the agent layer — local inference for fast/cheap tasks."""

import ollama as _ollama

from agent.config import AgentConfig

from agent.llm.base import BaseLLM


class OllamaLLM(BaseLLM):
    """LLM provider via local Ollama server."""

    def __init__(self, model_name: str | None = None, config: AgentConfig | None = None):
        config = config or AgentConfig()
        self.model = model_name or config.filter_llm_model
        self._client = _ollama.Client()

    def invoke(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float | None = None,
    ) -> str:
        """Send a prompt to the local Ollama model and return the response."""
        options: dict = {"num_predict": max_tokens}
        if temperature is not None:
            options["temperature"] = temperature
        resp = self._client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options=options,
        )
        content = resp.message.content
        return content.strip() if content else ""
