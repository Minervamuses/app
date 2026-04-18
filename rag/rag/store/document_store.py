"""Composite document store — writes to both Chroma and JSON."""

from langchain_core.documents import Document

from rag.store.chroma_store import ChromaStore
from rag.store.json_store import JSONStore


class DocumentStore:
    """Combines ChromaStore and JSONStore for dual persistence."""

    def __init__(self, chroma: ChromaStore, json_store: JSONStore):
        self.chroma = chroma
        self.json = json_store

    def add(self, documents: list[Document]) -> None:
        """Add documents to both stores."""
        self.json.add(documents)
        self.chroma.add(documents)

    def get(self, pid: str | None = None) -> list[Document]:
        """Retrieve from JSON store (authoritative for full content)."""
        return self.json.get(pid)

    def delete(self, pid: str) -> None:
        """Delete from both stores."""
        self.json.delete(pid)
        self.chroma.delete(pid)
