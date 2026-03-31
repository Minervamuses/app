"""Context tool — expand a search result with surrounding chunks."""

from kms.config import KMSConfig
from kms.store.json_store import JSONStore
from kms.tool.base import BaseTool


class ContextTool(BaseTool):
    """Retrieve surrounding chunks for a given pid + chunk_id."""

    def __init__(self, config: KMSConfig):
        self.config = config
        self._json_store = JSONStore(config.raw_json_path())

    def schema(self) -> dict:
        """Return the OpenAI-format tool definition."""
        return {
            "type": "function",
            "function": {
                "name": "get_context",
                "description": (
                    "Get the full context around a search result. "
                    "Given a pid and chunk_id from search results, returns that chunk "
                    "plus its neighboring chunks (before and after) from the same file. "
                    "Use this when a search result looks relevant but you need more context."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pid": {
                            "type": "string",
                            "description": "The pid (file identifier) from search results.",
                        },
                        "chunk_id": {
                            "type": "integer",
                            "description": "The chunk_id from search results.",
                        },
                        "window": {
                            "type": "integer",
                            "description": "How many chunks before/after to include (default 1, max 3).",
                        },
                    },
                    "required": ["pid", "chunk_id"],
                },
            },
        }

    def execute(self, arguments: dict) -> str:
        """Retrieve the target chunk and its neighbors from the JSON store."""
        pid = arguments["pid"]
        target_id = arguments["chunk_id"]
        window = min(arguments.get("window", 1), 3)

        all_docs = self._json_store.get(pid=pid)

        if not all_docs:
            return f"No chunks found for pid='{pid}'."

        all_docs.sort(key=lambda d: d.metadata.get("chunk_id", 0))

        target_idx = None
        for i, doc in enumerate(all_docs):
            if doc.metadata.get("chunk_id") == target_id:
                target_idx = i
                break

        if target_idx is None:
            available = [d.metadata.get("chunk_id") for d in all_docs]
            return f"chunk_id={target_id} not found in pid='{pid}'. Available: {available}"

        start = max(0, target_idx - window)
        end = min(len(all_docs), target_idx + window + 1)
        context_docs = all_docs[start:end]

        parts = []
        parts.append(f"Context for {pid} (chunks {start}~{end-1} of {len(all_docs)} total):\n")
        for doc in context_docs:
            cid = doc.metadata.get("chunk_id", "?")
            marker = " <<<" if cid == target_id else ""
            parts.append(f"--- chunk {cid}{marker} ---")
            parts.append(doc.page_content)

        return "\n".join(parts)
