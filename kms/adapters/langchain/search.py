"""LangChain search tool adapter."""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from kms.api import search as api_search
from kms.config import KMSConfig


class SearchInput(BaseModel):
    """Input schema for the search tool."""

    query: str = Field(description="Semantic search query ??describe what you're looking for.")
    folder_prefix: str | None = Field(None, description="Filter by folder path prefix (e.g. 'Research_notes' or 'pidna2/src'). Matches the folder and all subfolders.")
    category: str | None = Field(None, description="Filter by category (e.g. 'source-code', 'research-notes').")
    file_type: str | None = Field(None, description="Filter by file extension (e.g. '.py', '.md', '.sql').")
    date_from: str | None = Field(None, description="Start date inclusive (YYYY-MM-DD).")
    date_to: str | None = Field(None, description="End date inclusive (YYYY-MM-DD).")
    k: int = Field(5, description="Number of results (default 5).")


def create_search_tool(config: KMSConfig):
    """Create a LangChain search tool bound to the given config."""

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
        hits = api_search(
            query,
            k=k,
            folder_prefix=folder_prefix,
            category=category,
            file_type=file_type,
            date_from=date_from,
            date_to=date_to,
            config=config,
        )
        if not hits:
            return "No results found."

        parts = []
        for i, hit in enumerate(hits, 1):
            header = f"[{i}] {hit.file_path or '?'}"
            header += f" (category={hit.category or '?'})"
            if hit.date and hit.date != 0:
                header += f" (date={hit.date})"
            header += f" [pid={hit.pid or '?'}, chunk_id={hit.chunk_id}]"
            parts.append(f"{header}\n{hit.text[:600]}")

        return "\n\n".join(parts)

    return search
