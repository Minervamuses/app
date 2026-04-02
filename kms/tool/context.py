"""Context tool — expand a search result with surrounding chunks."""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from kms.config import KMSConfig
from kms.store.json_store import JSONStore


class ContextInput(BaseModel):
    """Input schema for the get_context tool."""

    pid: str = Field(description="The pid (file identifier) from search results.")
    chunk_id: int = Field(description="The chunk_id from search results.")
    window: int = Field(1, description="How many chunks before/after to include (default 1, max 3).")


def create_context_tool(config: KMSConfig):
    """Create a LangChain context tool bound to the given config."""
    json_store = JSONStore(config.raw_json_path())

    @tool("get_context", args_schema=ContextInput)
    def get_context(pid: str, chunk_id: int, window: int = 1) -> str:
        """Get the full context around a search result.

        Given a pid and chunk_id from search results, returns that chunk
        plus its neighboring chunks (before and after) from the same file.
        Use this when a search result looks relevant but you need more context.
        """
        window = min(window, 3)
        all_docs = json_store.get(pid=pid)

        if not all_docs:
            return f"No chunks found for pid='{pid}'."

        all_docs.sort(key=lambda d: d.metadata.get("chunk_id", 0))

        target_idx = None
        for i, doc in enumerate(all_docs):
            if doc.metadata.get("chunk_id") == chunk_id:
                target_idx = i
                break

        if target_idx is None:
            available = [d.metadata.get("chunk_id") for d in all_docs]
            return f"chunk_id={chunk_id} not found in pid='{pid}'. Available: {available}"

        start = max(0, target_idx - window)
        end = min(len(all_docs), target_idx + window + 1)
        context_docs = all_docs[start:end]

        parts = []
        parts.append(f"Context for {pid} (chunks {start}~{end-1} of {len(all_docs)} total):\n")
        for doc in context_docs:
            cid = doc.metadata.get("chunk_id", "?")
            marker = " <<<" if cid == chunk_id else ""
            parts.append(f"--- chunk {cid}{marker} ---")
            parts.append(doc.page_content)

        return "\n".join(parts)

    return get_context
