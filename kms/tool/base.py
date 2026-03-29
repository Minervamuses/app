"""Abstract base class for agent tools."""

from abc import ABC, abstractmethod


class BaseTool(ABC):
    """Abstract base class for tools the agent LLM can call."""

    @abstractmethod
    def schema(self) -> dict:
        """Return the OpenAI-format tool definition dict.

        Returns:
            Tool definition with 'type' and 'function' keys.
        """

    @abstractmethod
    def execute(self, arguments: dict) -> str:
        """Execute the tool with the given arguments.

        Args:
            arguments: Parsed arguments from the LLM tool call.

        Returns:
            String result to feed back to the LLM.
        """
