"""Search tool — semantic search with metadata filtering."""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from kms.config import KMSConfig, KNOWLEDGE_COLLECTION
from kms.retriever.vector import VectorRetriever
from kms.store.chroma_store import ChromaStore


def _date_to_int(date_str: str) -> int:
    """Convert 'YYYY-MM-DD' to YYYYMMDD integer for ChromaDB numeric filtering."""
    return int(date_str.replace("-", ""))


def _build_where(
    category: str | None,
    file_type: str | None,
    date_from: str | None,
    date_to: str | None,
    folder_prefix: str | None = None,
) -> dict | None:
    """Build a ChromaDB where clause from filter arguments."""
    conditions = []
    if folder_prefix:
        prefix = folder_prefix.rstrip("/")
        conditions.append({"folder": {"$contains": prefix}})
    if category:
        conditions.append({"category": {"$eq": category}})
    if file_type:
        conditions.append({"file_type": {"$eq": file_type}})
    if date_from:
        conditions.append({"date": {"$gte": _date_to_int(date_from)}})
    if date_to:
        conditions.append({"date": {"$lte": _date_to_int(date_to)}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


class SearchInput(BaseModel):
    """Input schema for the search tool."""

    query: str = Field(description="Semantic search query — describe what you're looking for.")
    folder_prefix: str | None = Field(None, description="Filter by folder path prefix (e.g. 'Research_notes' or 'pidna2/src'). Matches the folder and all subfolders.")
    category: str | None = Field(None, description="Filter by category (e.g. 'source-code', 'research-notes').")
    file_type: str | None = Field(None, description="Filter by file extension (e.g. '.py', '.md', '.sql').")
    date_from: str | None = Field(None, description="Start date inclusive (YYYY-MM-DD).")
    date_to: str | None = Field(None, description="End date inclusive (YYYY-MM-DD).")
    k: int = Field(5, description="Number of results (default 5).")


def create_search_tool(config: KMSConfig):
    """Create a LangChain search tool bound to the given config."""
    store = ChromaStore(KNOWLEDGE_COLLECTION, config)
    retriever = VectorRetriever(store)

    @tool("search", args_schema=SearchInput)
    def search(
        query: str,
        folder_prefix: str | None = None,
        category: str | None = None,
        file_type: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        k: int = 5,
    ) -> str:
        """Search the knowledge base by semantic similarity with optional metadata filters.

        Use the 'explore' tool first if you're unsure what categories or tags are available.
        You can call this multiple times with different queries or filters.
        """
        where = _build_where(category, file_type, date_from, date_to, folder_prefix)
        docs = retriever.retrieve(query, k=k, where=where)

        if not docs:
            return "No results found."

        parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            header = f"[{i}] {meta.get('file_path', '?')}"
            cat = meta.get("category", "?")
            header += f" (category={cat})"
            date = meta.get("date")
            if date and date != 0:
                header += f" (date={date})"
            pid = meta.get("pid", "?")
            chunk_id = meta.get("chunk_id", "?")
            header += f" [pid={pid}, chunk_id={chunk_id}]"
            preview = doc.page_content[:600]
            parts.append(f"{header}\n{preview}")

        return "\n\n".join(parts)

    return search
