"""Search tool — semantic search across multiple collections."""

import json

from kms.config import KMSConfig, MODULE_TO_COLLECTION, SUMMARY_COLLECTION
from kms.retriever.vector import VectorRetriever
from kms.store.chroma_store import ChromaStore
from kms.tool.base import BaseTool


def _date_to_int(date_str: str) -> int:
    """Convert 'YYYY-MM-DD' to YYYYMMDD integer for ChromaDB numeric filtering."""
    return int(date_str.replace("-", ""))


# All available collections the agent can search
ALL_COLLECTIONS = [SUMMARY_COLLECTION] + sorted(set(MODULE_TO_COLLECTION.values())) + ["general"]


class SearchTool(BaseTool):
    """Search the knowledge base across multiple collections."""

    def __init__(self, config: KMSConfig):
        self.config = config
        self._retrievers: dict[str, VectorRetriever] = {}

    def _get_retriever(self, collection: str) -> VectorRetriever:
        """Get or create a retriever for a collection."""
        if collection not in self._retrievers:
            store = ChromaStore(collection, self.config)
            self._retrievers[collection] = VectorRetriever(store)
        return self._retrievers[collection]

    def schema(self) -> dict:
        """Return the OpenAI-format tool definition."""
        return {
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    "Search the lab knowledge base by semantic similarity. "
                    "Pick which collection(s) to search. "
                    "You can call this tool multiple times with different queries or collections.\n\n"
                    "Collections:\n"
                    f"- summaries: folder-level overviews of the entire project ({SUMMARY_COLLECTION})\n"
                    "- research_notes: research logs, progress reports, investigation notes\n"
                    "- source_code: core Python implementation (pidna2/)\n"
                    "- web: frontend (React) and backend (FastAPI)\n"
                    "- legacy: PiDNA1 PL/SQL reference code\n"
                    "- docs: documentation, plans, specifications\n"
                    "- general: root-level files (README, paper, etc.)"
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
                                "enum": ALL_COLLECTIONS,
                            },
                            "description": "Which collection(s) to search. Defaults to all content collections.",
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
            collections = [c for c in ALL_COLLECTIONS if c != SUMMARY_COLLECTION]

        # Build date filter if specified
        where = self._build_date_where(arguments)

        all_parts = []
        for collection in collections:
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
