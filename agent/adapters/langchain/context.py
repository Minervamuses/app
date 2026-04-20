"""LangChain context tool adapter."""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from rag.api import get_context as api_get_context
from agent.config import AgentConfig


class ContextInput(BaseModel):
    """Input schema for the get_context tool."""

    pid: str = Field(description="The pid (file identifier) from search results.")
    chunk_id: int = Field(description="The chunk_id from search results.")
    window: int = Field(1, description="How many chunks before/after to include (default 1, max 3).")


def create_context_tool(config: AgentConfig):
    """Create a LangChain context tool bound to the given config."""

    @tool("get_context", args_schema=ContextInput)
    def get_context(pid: str, chunk_id: int, window: int = 1) -> str:
        """Get the full context around a search result.

        Given a pid and chunk_id from search results, returns that chunk
        plus its neighboring chunks (before and after) from the same file.
        Use this when a search result looks relevant but you need more context.
        """
        try:
            context = api_get_context(pid, chunk_id, window=window, config=config)
        except ValueError as exc:
            return str(exc)

        if context is None:
            return f"No chunks found for pid='{pid}'."

        start = context.chunks[0].chunk_id if context.chunks else chunk_id
        end = context.chunks[-1].chunk_id if context.chunks else chunk_id

        parts = [f"Context for {pid} (chunks {start}~{end} of {context.total_chunks_in_doc} total):\n"]
        for chunk in context.chunks:
            marker = " <<<" if chunk.is_target else ""
            parts.append(f"--- chunk {chunk.chunk_id}{marker} ---")
            parts.append(chunk.text)

        return "\n".join(parts)

    return get_context
