"""Explore tool — discover what's in the knowledge base."""

import json
from pathlib import Path

from kms.cli.ingest import _extract_date
from kms.config import KMSConfig
from kms.tool.base import BaseTool


class ExploreTool(BaseTool):
    """Let the agent discover available categories, tags, date ranges, and folder summaries."""

    def __init__(self, config: KMSConfig):
        self.config = config
        self._folder_meta = self._load_meta()

    def _load_meta(self) -> dict:
        meta_path = Path(self.config.persist_dir) / "folder_meta.json"
        try:
            with open(meta_path) as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def schema(self) -> dict:
        """Return the OpenAI-format tool definition."""
        return {
            "type": "function",
            "function": {
                "name": "explore",
                "description": (
                    "Discover what's available in the knowledge base. "
                    "Returns folder summaries, available categories, tags, and date ranges. "
                    "Use this BEFORE searching when you're unsure what the knowledge base contains "
                    "or which category/tags to filter by.\n\n"
                    "You can optionally filter by category to see only relevant folders."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Optional: only show folders matching this category (first tag).",
                        },
                    },
                    "required": [],
                },
            },
        }

    def execute(self, arguments: dict) -> str:
        """Execute the explore tool, returning knowledge base overview."""
        category_filter = arguments.get("category")

        categories: dict[str, int] = {}
        all_tags: set[str] = set()
        dates: list[int] = []
        folder_lines: list[str] = []

        for folder_rel, meta in self._folder_meta.items():
            tags = meta.get("tags", [])
            summary = meta.get("summary", "")
            cat = tags[0] if tags else "unknown"

            categories[cat] = categories.get(cat, 0) + 1
            all_tags.update(tags)

            d = _extract_date(folder_rel)
            if d:
                dates.append(d)

            if category_filter and cat != category_filter:
                continue

            display = folder_rel or "(root)"
            folder_lines.append(f"- [{cat}] {display}: {summary}")

        parts = []

        parts.append("## Categories (with folder count)")
        for cat, count in sorted(categories.items()):
            parts.append(f"  - {cat}: {count}")

        if dates:
            parts.append(f"\n## Date range: {min(dates)} ~ {max(dates)}")

        parts.append(f"\n## All tags ({len(all_tags)})")
        parts.append(", ".join(sorted(all_tags)))

        header = "\n## Folder summaries"
        if category_filter:
            header += f" (category={category_filter})"
        parts.append(header)
        parts.extend(folder_lines)

        return "\n".join(parts)
