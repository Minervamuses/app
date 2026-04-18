"""Abstract base class for embedders."""

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Abstract base class for text embedders."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.

        Args:
            text: Query string to embed.

        Returns:
            Embedding vector.
        """
