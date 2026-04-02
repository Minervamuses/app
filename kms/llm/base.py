"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract base class for language model providers."""

    @abstractmethod
    def invoke(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float | None = None,
    ) -> str:
        """Send a prompt to the LLM and return the response.

        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (None for model default).

        Returns:
            The model's text response.
        """
