"""Vector retriever — semantic similarity search via ChromaDB."""

from langchain_core.documents import Document

from kms.retriever.base import BaseRetriever
from kms.store.chroma_store import ChromaStore


class VectorRetriever(BaseRetriever):
    """Retrieve documents using vector similarity search."""

    def __init__(self, chroma_store: ChromaStore):
        self.chroma_store = chroma_store

    def retrieve(
        self,
        query: str,
        k: int,
        pid_filter: list[str] | None = None,
        where: dict | None = None,
    ) -> list[Document]:
        """Retrieve top-k documents by semantic similarity."""
        retriever = self.chroma_store.as_retriever(k, pid_filter, where)
        return retriever.invoke(query)
