"""Search tool — semantic search with metadata filtering."""

import json
from pathlib import Path

from kms.config import KMSConfig
from kms.retriever.vector import VectorRetriever
from kms.store.chroma_store import ChromaStore
from kms.tool.base import BaseTool


def _date_to_int(date_str: str) -> int:
    """Convert 'YYYY-MM-DD' to YYYYMMDD integer for ChromaDB numeric filtering."""
    return int(date_str.replace("-", ""))


class SearchTool(BaseTool):
    """Search the knowledge base by semantic similarity with metadata filters."""

    def __init__(self, config: KMSConfig):
        self.config = config
        self.chroma = ChromaStore(config.raw_collection, config)
        self.retriever = VectorRetriever(self.chroma)
        self._enums = self._load_enums()

    def _load_enums(self) -> dict[str, list[str]]:
        """Load available enum values from ingested metadata."""
        tags_path = Path(self.config.persist_dir) / "folder_tags.json"
        enums: dict[str, list[str]] = {
            "module": set(),
            "purpose": set(),
            "tags": set(),
        }

        try:
            with open(tags_path) as f:
                folder_tags = json.load(f)
            for folder_rel, tags in folder_tags.items():
                # Module = first path component
                parts = Path(folder_rel).parts
                if parts:
                    enums["module"].add(parts[0])
                for t in tags:
                    enums["tags"].add(t)
        except FileNotFoundError:
            pass

        # Purpose values are deterministic from _PURPOSE_MAP in ingest
        enums["purpose"] = {
            "implementation", "research-log", "documentation",
            "config", "legacy-reference", "test", "other",
        }

        return {k: sorted(v) for k, v in enums.items()}

    def schema(self) -> dict:
        """Return the OpenAI-format tool definition."""
        return {
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    "Search the lab knowledge base by semantic similarity. "
                    "Use filters to narrow results by metadata. "
                    "You can call this tool multiple times with different queries or filters."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Semantic search query — describe what you're looking for.",
                        },
                        "module": {
                            "type": "string",
                            "description": "Top-level project area.",
                            "enum": self._enums.get("module", []),
                        },
                        "file_type": {
                            "type": "string",
                            "description": "File extension filter (e.g. '.py', '.md', '.sql').",
                        },
                        "purpose": {
                            "type": "string",
                            "description": "Document purpose category.",
                            "enum": self._enums.get("purpose", []),
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Topic tags to filter by (match any). "
                                f"Available: {', '.join(self._enums.get('tags', []))}"
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
                            "description": "Number of results to return (default 5).",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def execute(self, arguments: dict) -> str:
        """Execute the search and return formatted results."""
        query = arguments["query"]
        k = arguments.get("k", 5)

        # Build ChromaDB where clause from filters
        where = self._build_where(arguments)

        docs = self.retriever.retrieve(query, k=k, where=where)

        if not docs:
            return "No results found."

        parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            header = (
                f"[{i}] {meta.get('file_path', '?')} "
                f"(chunk {meta.get('chunk_id', '?')}, "
                f"purpose={meta.get('purpose', '?')}, "
                f"module={meta.get('module', '?')}"
            )
            date = meta.get("date")
            if date:
                header += f", date={date}"
            header += ")"
            preview = doc.page_content[:600]
            parts.append(f"{header}\n{preview}")

        return "\n\n".join(parts)

    def _build_where(self, arguments: dict) -> dict | None:
        """Translate tool arguments into a ChromaDB where clause."""
        conditions = []

        if "module" in arguments:
            conditions.append({"module": {"$eq": arguments["module"]}})
        if "file_type" in arguments:
            conditions.append({"file_type": {"$eq": arguments["file_type"]}})
        if "purpose" in arguments:
            conditions.append({"purpose": {"$eq": arguments["purpose"]}})
        if "date_from" in arguments:
            # Convert YYYY-MM-DD string to YYYYMMDD int for ChromaDB numeric comparison
            conditions.append({"date": {"$gte": _date_to_int(arguments["date_from"])}})
        if "date_to" in arguments:
            conditions.append({"date": {"$lte": _date_to_int(arguments["date_to"])}})
        if "tags" in arguments:
            # Tags are stored as string repr of list, use $contains for substring match
            tag_conditions = [{"tags": {"$contains": t}} for t in arguments["tags"]]
            if len(tag_conditions) == 1:
                conditions.append(tag_conditions[0])
            elif tag_conditions:
                conditions.append({"$or": tag_conditions})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
