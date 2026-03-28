"""Abstract base class for retrievers."""

from abc import ABC, abstractmethod
from langchain.schema import Document


class BaseRetriever(ABC):
    """Abstract base class for document retrievers."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int,
        pid_filter: list[str] | None = None,
    ) -> list[Document]:
        """Retrieve top-k documents for a query.

        Args:
            query: The search query.
            k: Number of results to return.
            pid_filter: Optional list of pids to restrict search.

        Returns:
            List of Documents ranked by relevance.
        """
