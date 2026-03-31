"""Search tool — semantic search across multiple collections."""

import json
from pathlib import Path

from kms.config import KMSConfig, SUMMARY_COLLECTION
from kms.retriever.vector import VectorRetriever
from kms.store.chroma_store import ChromaStore
from kms.tool.base import BaseTool


def _date_to_int(date_str: str) -> int:
    """Convert 'YYYY-MM-DD' to YYYYMMDD integer for ChromaDB numeric filtering."""
    return int(date_str.replace("-", ""))


def _load_collections(config: KMSConfig) -> list[str]:
    """Discover available collections from folder_meta.json.

    Collections are auto-generated from top-level directory names during
    ingest, so this adapts to any repo structure.
    """
    meta_path = Path(config.persist_dir) / "folder_meta.json"
    collections = {SUMMARY_COLLECTION}
    try:
        with open(meta_path) as f:
            folder_meta = json.load(f)
        for entry in folder_meta.values():
            col = entry.get("collection")
            if col:
                collections.add(col)
    except FileNotFoundError:
        pass
    return sorted(collections)


class SearchTool(BaseTool):
    """Search the knowledge base across multiple collections."""

    def __init__(self, config: KMSConfig):
        self.config = config
        self._retrievers: dict[str, VectorRetriever] = {}
        self._collections = _load_collections(config)

    def _get_retriever(self, collection: str) -> VectorRetriever:
        """Get or create a retriever for a collection."""
        if collection not in self._retrievers:
            store = ChromaStore(collection, self.config)
            self._retrievers[collection] = VectorRetriever(store)
        return self._retrievers[collection]

    def schema(self) -> dict:
        """Return the OpenAI-format tool definition."""
        content_collections = [c for c in self._collections if c != SUMMARY_COLLECTION]
        collection_list = "\n".join(f"- {c}" for c in self._collections)

        return {
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    "Search the lab knowledge base by semantic similarity. "
                    "Pick which collection(s) to search. "
                    "You can call this tool multiple times with different queries or collections.\n\n"
                    f"Available collections:\n{collection_list}\n\n"
                    f"'{SUMMARY_COLLECTION}' contains folder-level overviews of the entire project. "
                    "Other collections correspond to top-level directories in the repo."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Semantic search query — describe what you're looking for.",
                        },
                        "collections": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": self._collections,
                            },
                            "description": (
                                "Which collection(s) to search. "
                                "Defaults to all content collections (excluding summaries)."
                            ),
                        },
                        "date_from": {
                            "type": "string",
                            "description": "Start date inclusive (YYYY-MM-DD). For time-bounded queries.",
                        },
                        "date_to": {
                            "type": "string",
                            "description": "End date inclusive (YYYY-MM-DD). For time-bounded queries.",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results to return per collection (default 5).",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def execute(self, arguments: dict) -> str:
        """Execute the search across selected collections."""
        query = arguments["query"]
        k = arguments.get("k", 5)
        collections = arguments.get("collections", None)

        # Default: search all content collections (not summaries)
        if not collections:
            collections = [c for c in self._collections if c != SUMMARY_COLLECTION]

        # Build date filter if specified
        where = self._build_date_where(arguments)

        all_parts = []
        for collection in collections:
            if collection not in self._collections:
                continue
            retriever = self._get_retriever(collection)
            docs = retriever.retrieve(query, k=k, where=where)

            if not docs:
                continue

            for i, doc in enumerate(docs, 1):
                meta = doc.metadata
                header = f"[{collection}/{i}] {meta.get('file_path', meta.get('folder', '?'))}"
                date = meta.get("date")
                if date and date != 0:
                    header += f" (date={date})"
                preview = doc.page_content[:600]
                all_parts.append(f"{header}\n{preview}")

        if not all_parts:
            return "No results found."

        return "\n\n".join(all_parts)

    def _build_date_where(self, arguments: dict) -> dict | None:
        """Build a date range where clause if dates are specified."""
        conditions = []
        if "date_from" in arguments:
            conditions.append({"date": {"$gte": _date_to_int(arguments["date_from"])}})
        if "date_to" in arguments:
            conditions.append({"date": {"$lte": _date_to_int(arguments["date_to"])}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
