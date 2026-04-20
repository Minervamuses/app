"""LangChain explore tool adapter."""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from rag.api import explore as api_explore
from agent.config import AgentConfig


class ExploreInput(BaseModel):
    """Input schema for the explore tool."""

    category: str | None = Field(None, description="Optional: only show folders matching this category (first tag).")


def create_explore_tool(config: AgentConfig):
    """Create a LangChain explore tool bound to the given config."""

    @tool("explore", args_schema=ExploreInput)
    def explore(category: str | None = None) -> str:
        """Discover what's available in the knowledge base.

        Returns folder summaries, available categories, tags, and date ranges.
        Use this BEFORE searching when you're unsure what the knowledge base contains
        or which category/tags to filter by.
        """
        inventory = api_explore(category=category, config=config)
        parts = ["## Categories (with folder count)"]

        for cat, count in sorted(inventory.categories.items()):
            parts.append(f"  - {cat}: {count}")

        if inventory.date_range:
            parts.append(f"\n## Date range: {inventory.date_range[0]} ~ {inventory.date_range[1]}")

        parts.append(f"\n## All tags ({len(inventory.tags)})")
        parts.append(", ".join(inventory.tags))

        header = "\n## Folder summaries"
        if category:
            header += f" (category={category})"
        parts.append(header)

        for folder in inventory.folders:
            display = folder.folder or "(root)"
            parts.append(f"- [{folder.category}] {display}: {folder.summary}")

        return "\n".join(parts)

    return explore
