"""Search tool — semantic search with metadata filtering."""

import json

from kms.config import KMSConfig, KNOWLEDGE_COLLECTION
from kms.retriever.vector import VectorRetriever
from kms.store.chroma_store import ChromaStore
from kms.tool.base import BaseTool


def _date_to_int(date_str: str) -> int:
    """Convert 'YYYY-MM-DD' to YYYYMMDD integer for ChromaDB numeric filtering."""
    return int(date_str.replace("-", ""))


class SearchTool(BaseTool):
    """Semantic search across the knowledge base with metadata filters."""

    def __init__(self, config: KMSConfig):
        self.config = config
        store = ChromaStore(KNOWLEDGE_COLLECTION, config)
        self._retriever = VectorRetriever(store)

    def schema(self) -> dict:
        """Return the OpenAI-format tool definition."""
        return {
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    "Search the knowledge base by semantic similarity with optional metadata filters. "
                    "Use the 'explore' tool first if you're unsure what categories or tags are available.\n\n"
                    "Tips:\n"
                    "- Use 'category' to narrow by broad type (e.g. 'source-code', 'research-notes')\n"
                    "- Use 'date_from'/'date_to' for time-bounded queries\n"
                    "- Use 'file_type' to filter by extension (e.g. '.py', '.md')\n"
                    "- You can call this multiple times with different queries or filters"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Semantic search query — describe what you're looking for.",
                        },
                        "category": {
                            "type": "string",
                            "description": "Filter by category (the broad first-tag, e.g. 'source-code', 'research-notes').",
                        },
                        "file_type": {
                            "type": "string",
                            "description": "Filter by file extension (e.g. '.py', '.md', '.sql').",
                        },
                        "date_from": {
                            "type": "string",
                            "description": "Start date inclusive (YYYY-MM-DD).",
                        },
                        "date_to": {
                            "type": "string",
                            "description": "End date inclusive (YYYY-MM-DD).",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results (default 5).",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def execute(self, arguments: dict) -> str:
        """Execute the search with optional metadata filters."""
        query = arguments["query"]
        k = arguments.get("k", 5)
        where = self._build_where(arguments)

        docs = self._retriever.retrieve(query, k=k, where=where)

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

    def _build_where(self, arguments: dict) -> dict | None:
        """Build a ChromaDB where clause from the filter arguments."""
        conditions = []

        if "category" in arguments:
            conditions.append({"category": {"$eq": arguments["category"]}})
        if "file_type" in arguments:
            conditions.append({"file_type": {"$eq": arguments["file_type"]}})
        if "date_from" in arguments:
            conditions.append({"date": {"$gte": _date_to_int(arguments["date_from"])}})
        if "date_to" in arguments:
            conditions.append({"date": {"$lte": _date_to_int(arguments["date_to"])}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
