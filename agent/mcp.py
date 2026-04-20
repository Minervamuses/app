"""MCP client loader for the agent.

Reads MCP server config from environment variables (keeping secrets and
machine-specific command paths out of tracked files) and asks
``langchain_mcp_adapters`` to load the merged tool list asynchronously.

Server enablement is opt-in per server; if an MCP server is disabled or
mis-configured, the rest of the agent still works with the local KB
tools only.
"""

from __future__ import annotations

import logging
import os
import shlex
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCPServerSpec:
    """Resolved stdio MCP server launch spec."""

    name: str
    command: str
    args: list[str]
    env: dict[str, str]


def _parse_args(raw: str | None) -> list[str]:
    if not raw:
        return []
    return shlex.split(raw)


def _env_truthy(name: str) -> bool:
    val = os.environ.get(name, "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _web_search_spec() -> MCPServerSpec | None:
    if not _env_truthy("AGENT_ENABLE_MCP_WEB_SEARCH"):
        return None
    command = os.environ.get("AGENT_MCP_WEB_SEARCH_COMMAND")
    if not command:
        logger.warning(
            "AGENT_ENABLE_MCP_WEB_SEARCH is set but AGENT_MCP_WEB_SEARCH_COMMAND is empty; "
            "skipping Web Search MCP."
        )
        return None
    return MCPServerSpec(
        name="web_search",
        command=command,
        args=_parse_args(os.environ.get("AGENT_MCP_WEB_SEARCH_ARGS")),
        env={},
    )


def _github_spec() -> MCPServerSpec | None:
    if not _env_truthy("AGENT_ENABLE_MCP_GITHUB"):
        return None
    command = os.environ.get("AGENT_MCP_GITHUB_COMMAND")
    if not command:
        logger.warning(
            "AGENT_ENABLE_MCP_GITHUB is set but AGENT_MCP_GITHUB_COMMAND is empty; "
            "skipping GitHub MCP."
        )
        return None
    token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    if not token:
        logger.warning(
            "AGENT_ENABLE_MCP_GITHUB is set but GITHUB_PERSONAL_ACCESS_TOKEN is empty; "
            "GitHub MCP will start without auth and likely refuse most calls."
        )
    toolsets = os.environ.get(
        "AGENT_MCP_GITHUB_TOOLSETS",
        "repos,pull_requests,issues,actions,context",
    )
    env = {
        "GITHUB_PERSONAL_ACCESS_TOKEN": token,
        "GITHUB_TOOLSETS": toolsets,
    }
    return MCPServerSpec(
        name="github",
        command=command,
        args=_parse_args(os.environ.get("AGENT_MCP_GITHUB_ARGS")),
        env=env,
    )


def resolve_mcp_specs() -> list[MCPServerSpec]:
    """Collect enabled MCP server specs from environment. Empty list = MCP off."""
    specs = []
    for resolver in (_web_search_spec, _github_spec):
        spec = resolver()
        if spec is not None:
            specs.append(spec)
    return specs


def _spec_to_connection(spec: MCPServerSpec) -> dict:
    # Some MCP servers (notably mrkrsl/web-search-mcp) emit verbose debug
    # output on stderr that the stdio transport forwards straight to the
    # parent terminal. Wrap the real command in /bin/sh so we can redirect
    # stderr to /dev/null without asking each server to behave.
    inner = shlex.join([spec.command, *spec.args])
    conn: dict = {
        "transport": "stdio",
        "command": "/bin/sh",
        "args": ["-c", f"exec {inner} 2>/dev/null"],
    }
    if spec.env:
        conn["env"] = dict(spec.env)
    return conn


async def load_mcp_tools(specs: list[MCPServerSpec] | None = None) -> list:
    """Start the configured MCP servers and return the merged LangChain tool list.

    Failures from any single server are logged and that server is skipped;
    tools from surviving servers are still returned.
    """
    if specs is None:
        specs = resolve_mcp_specs()
    if not specs:
        return []

    # Some upstream MCP servers (e.g. mrkrsl/web-search-mcp) print banners
    # on stdout instead of stderr. The stdio client logs each non-JSON line
    # as an exception; silence that channel so user-facing output stays clean.
    logging.getLogger("mcp.client.stdio").setLevel(logging.CRITICAL)

    from langchain_mcp_adapters.client import MultiServerMCPClient

    tools: list = []
    for spec in specs:
        connections = {spec.name: _spec_to_connection(spec)}
        try:
            client = MultiServerMCPClient(connections=connections)
            server_tools = await client.get_tools()
        except Exception as exc:
            logger.warning("MCP server %r failed to load: %s", spec.name, exc)
            continue
        tools.extend(server_tools)
    return tools
