"""Abstract base class for chunkers."""

from abc import ABC, abstractmethod
from langchain_core.documents import Document


class BaseChunker(ABC):
    """Abstract base class for document chunkers."""

    @abstractmethod
    def chunk(self, text: str, pid: str) -> list[Document]:
        """Split text into chunks.

        Args:
            text: The full document text to chunk.
            pid: Paper/document identifier for metadata.

        Returns:
            List of Documents with page_content and metadata.
        """
