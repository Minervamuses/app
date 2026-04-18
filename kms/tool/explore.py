"""Explore tool — discover what's in the knowledge base."""

import json
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from kms.config import KMSConfig
from kms.utils.paths import extract_date


class ExploreInput(BaseModel):
    """Input schema for the explore tool."""

    category: str | None = Field(None, description="Optional: only show folders matching this category (first tag).")


def _load_meta(config: KMSConfig) -> dict:
    """Load folder_meta.json from the persist directory."""
    meta_path = Path(config.folder_meta_path())
    try:
        with open(meta_path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def create_explore_tool(config: KMSConfig):
    """Create a LangChain explore tool bound to the given config."""
    folder_meta = _load_meta(config)

    @tool("explore", args_schema=ExploreInput)
    def explore(category: str | None = None) -> str:
        """Discover what's available in the knowledge base.

        Returns folder summaries, available categories, tags, and date ranges.
        Use this BEFORE searching when you're unsure what the knowledge base contains
        or which category/tags to filter by.
        """
        categories: dict[str, int] = {}
        all_tags: set[str] = set()
        dates: list[int] = []
        folder_lines: list[str] = []

        for folder_rel, meta in folder_meta.items():
            tags = meta.get("tags", [])
            summary = meta.get("summary", "")
            cat = tags[0] if tags else "unknown"

            categories[cat] = categories.get(cat, 0) + 1
            all_tags.update(tags)

            d = extract_date(folder_rel)
            if d:
                dates.append(d)

            if category and cat != category:
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
        if category:
            header += f" (category={category})"
        parts.append(header)
        parts.extend(folder_lines)

        return "\n".join(parts)

    return explore
