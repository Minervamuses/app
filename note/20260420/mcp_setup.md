# MCP Setup for the `app` Agent

Date: 2026-04-20

Implementation notes for the Web Search MCP and GitHub MCP integration
in the Python `app` agent.

## Two independent MCP layers

- `PiDNA2/opencode.json` — config for the **external `opencode` host**.
  Entries there do not affect the Python LangGraph agent.
- `app/agent/mcp.py` — **Python runtime** MCP loader for the `app` agent.
  Driven by env vars in `app/.env`, read at session startup.

Adding MCP entries in one layer does **not** make them available in the
other. Configure the layer(s) you actually need.

## Enable Web Search MCP for the `app` agent

```
AGENT_ENABLE_MCP_WEB_SEARCH=1
AGENT_MCP_WEB_SEARCH_COMMAND=npx
AGENT_MCP_WEB_SEARCH_ARGS=-y web-search-mcp@latest
```

See upstream: `mrkrsl/web-search-mcp`.

## Enable GitHub MCP for the `app` agent

```
AGENT_ENABLE_MCP_GITHUB=1
AGENT_MCP_GITHUB_COMMAND=/absolute/path/to/github-mcp-server
AGENT_MCP_GITHUB_ARGS=stdio
AGENT_MCP_GITHUB_TOOLSETS=repos,pull_requests,issues,actions,context
GITHUB_PERSONAL_ACCESS_TOKEN=ghp_xxx
```

See upstream: `github/github-mcp-server`.

Scope is intentionally limited to reading and navigating remote state.
GitHub MCP is **not** a substitute for local git shell workflow; local
clone/pull/rebase/commit continue to live in the terminal, not in the
agent.

## Run without MCP

Either leave both `AGENT_ENABLE_MCP_*` flags unset, or launch with
`--no-mcp`:

```
python -m agent.cli.chat --no-mcp
```

The agent still runs with local KB tools only (`explore`, `search`,
`get_context`).

## Failure behavior

If an enabled MCP server fails to launch, `agent.mcp.load_mcp_tools`
logs a warning and returns tools from the remaining servers only. The
session always starts, even if every MCP server fails — only the local
KB tools are bound in that case.

## Interaction with turn compaction

Unrelated: long-term conversation memory is now compacted every
`config.agent_turns_per_compaction` completed turns (default 10) into a
single rolling summary, independent of whether MCP is enabled.
