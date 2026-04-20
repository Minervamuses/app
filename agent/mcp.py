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


def _mcp_log_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    path = os.path.join(base, "agent-mcp")
    os.makedirs(path, exist_ok=True)
    return path


def _spec_to_connection(spec: MCPServerSpec) -> dict:
    # Some MCP servers (notably mrkrsl/web-search-mcp) are careless about
    # what they write to stdout: startup banners, shutdown notices, etc.
    # The stdio transport then tries to JSON-parse those lines and, on
    # shutdown, the resulting ValidationError races against stream close
    # and surfaces as an ExceptionGroup[BrokenResourceError] inside the
    # tool call. Fix it at the subprocess level:
    #   1. stderr goes to a per-server log file (so we can still debug).
    #   2. stdout passes through grep that only forwards lines starting
    #      with '{' — JSON-RPC messages are always JSON objects, so this
    #      is a safe filter that drops every free-form stdout print.
    inner = shlex.join([spec.command, *spec.args])
    log_path = os.path.join(_mcp_log_dir(), f"{spec.name}.stderr.log")
    pipeline = (
        f"{inner} 2>>{shlex.quote(log_path)} "
        f"| grep --line-buffered '^{{'"
    )
    conn: dict = {
        "transport": "stdio",
        "command": "/bin/sh",
        "args": ["-c", pipeline],
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
