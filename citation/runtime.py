"""Bridge to the host project's config, chat model, and Web Search MCP.

Everything here is a *read-only consumer* of the existing ``agent`` package:
we import its config, its chat-model factory, and its MCP loader, but never
modify them. If those abstractions change, this prototype adapts here and only
here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger("citation.runtime")

# The three tools exposed by mrkrsl/web-search-mcp (family "web_search").
SUMMARIES_TOOL = "get-web-search-summaries"
FULL_SEARCH_TOOL = "full-web-search"
PAGE_CONTENT_TOOL = "get-single-web-page-content"
WEB_SEARCH_FAMILY = "web_search"


class WebSearchUnavailable(RuntimeError):
    """Raised when the Web Search MCP family is not loaded/enabled."""


@dataclass
class CitationRuntime:
    """Resolved, reusable handles for one prototype run."""

    config: object  # agent.config.AgentConfig
    llm: object | None  # langchain chat model, or None if no API key
    web_tools: dict[str, object]  # tool name -> LangChain tool
    app_root: Path

    @property
    def cite_dir(self) -> Path:
        return self.app_root / "citation" / "cite"

    def require_web_tool(self, name: str):
        tool = self.web_tools.get(name)
        if tool is None:
            raise WebSearchUnavailable(
                f"Web Search MCP tool {name!r} is not available. "
                "Enable it in .env (AGENT_ENABLE_MCP_WEB_SEARCH=1 plus the "
                "AGENT_MCP_WEB_SEARCH_COMMAND/ARGS), and make sure the MCP "
                "server starts cleanly."
            )
        return tool


def _load_env() -> None:
    """Load the host project's .env exactly like agent.cli.chat does."""
    from agent.paths import find_app_root

    env_path = find_app_root() / ".env"
    # override=False: never clobber variables already set in the real shell.
    load_dotenv(dotenv_path=env_path, override=False)


def _build_llm(config):
    """Return the host project's chat model, or None if it cannot be built.

    The prototype only uses the LLM to *enrich/parse* discovery results, never
    to generate BibTeX, so a missing key degrades gracefully.
    """
    try:
        from agent.llm import get_chat_model

        return get_chat_model(config)
    except Exception as exc:  # noqa: BLE001 - missing key etc. must not abort discovery
        logger.warning("chat model unavailable (%s); continuing without LLM enrichment", exc)
        return None


async def build_runtime(*, load_mcp: bool = True) -> CitationRuntime:
    """Assemble a :class:`CitationRuntime` using the host project's plumbing.

    Args:
        load_mcp: When False, skip MCP loading entirely (offline/unit use).
    """
    _load_env()

    from agent.config import AgentConfig
    from agent.paths import find_app_root

    config = AgentConfig()
    llm = _build_llm(config)

    web_tools: dict[str, object] = {}
    if load_mcp:
        from agent.mcp import load_mcp_tools_with_families

        try:
            tools, families = await load_mcp_tools_with_families()
        except Exception as exc:  # noqa: BLE001 - surfaced later as WebSearchUnavailable
            logger.warning("MCP loading failed: %s", exc)
            tools, families = [], {}
        for tool in tools:
            name = getattr(tool, "name", None)
            if name and families.get(name) == WEB_SEARCH_FAMILY:
                web_tools[name] = tool

    return CitationRuntime(
        config=config,
        llm=llm,
        web_tools=web_tools,
        app_root=find_app_root(),
    )
