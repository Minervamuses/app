"""Abstract base class for document stores."""

from abc import ABC, abstractmethod
from langchain_core.documents import Document


class BaseStore(ABC):
    """Abstract base class for document stores."""

    @abstractmethod
    def add(self, documents: list[Document]) -> None:
        """Add documents to the store."""

    @abstractmethod
    def get(self, pid: str | None = None) -> list[Document]:
        """Retrieve documents, optionally filtered by pid."""

    @abstractmethod
    def delete(self, pid: str) -> None:
        """Delete all documents matching a pid."""
