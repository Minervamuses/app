"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""

    id: str
    name: str
    arguments: dict


@dataclass
class ChatResponse:
    """Response from a chat completion, may contain text or tool calls."""

    content: str | None
    tool_calls: list[ToolCall]

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


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

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> ChatResponse:
        """Send a multi-turn conversation with optional tool definitions.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            tools: Optional list of tool definitions (OpenAI format).
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (None for model default).

        Returns:
            ChatResponse with text content and/or tool calls.
        """
