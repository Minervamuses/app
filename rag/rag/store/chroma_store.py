"""ChromaDB vector store."""

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from rag.config import KMSConfig
from rag.store.base import BaseStore


class ChromaStore(BaseStore):
    """Vector store backed by ChromaDB."""

    def __init__(
        self,
        collection_name: str,
        config: KMSConfig,
        use_embeddings: bool = True,
    ):
        self.config = config
        self.collection_name = collection_name
        if use_embeddings:
            embeddings = OllamaEmbeddings(model=config.embed_model)
        else:
            embeddings = None
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=config.persist_dir,
        )

    def add(self, documents: list[Document]) -> None:
        """Add documents to ChromaDB."""
        self._store.add_documents(documents)

    def get(self, pid: str | None = None) -> list[Document]:
        """Retrieve documents, optionally filtered by pid."""
        where = {"pid": pid} if pid else None
        results = self._store.get(where=where)
        docs = []
        for i, content in enumerate(results["documents"]):
            meta = results["metadatas"][i] if results["metadatas"] else {}
            docs.append(Document(page_content=content, metadata=meta))
        return docs

    def delete(self, pid: str) -> None:
        """Delete all documents matching a pid."""
        results = self._store.get(where={"pid": pid})
        if results["ids"]:
            self._store.delete(ids=results["ids"])

    def as_retriever(
        self,
        k: int,
        pid_filter: list[str] | None = None,
        where: dict | None = None,
    ):
        """Return a LangChain-compatible retriever for vector search.

        Args:
            k: Number of results.
            pid_filter: Legacy pid filter (shortcut for where={"pid": {"$in": ...}}).
            where: Arbitrary ChromaDB where clause for metadata filtering.
                   Supports $eq, $ne, $gt, $gte, $lt, $lte, $in, $and, $or.
        """
        search_kwargs: dict = {"k": k}
        if where:
            search_kwargs["filter"] = where
        elif pid_filter:
            search_kwargs["filter"] = {"pid": {"$in": pid_filter}}
        return self._store.as_retriever(search_kwargs=search_kwargs)
