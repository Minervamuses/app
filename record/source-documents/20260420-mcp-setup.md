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

`mrkrsl/web-search-mcp` is **not** published on npm — it ships as a
GitHub release zip with pre-built `dist/`. Install once:

```
mkdir -p ~/.local/share/mcp-servers
cd ~/.local/share/mcp-servers
curl -sSL -o ws.zip https://github.com/mrkrsl/web-search-mcp/releases/latest/download/web-search-mcp-v0.3.2.zip
mkdir web-search-mcp && cd web-search-mcp
unzip -q ../ws.zip && rm ../ws.zip
npm install --omit=dev
npx playwright install chromium   # ~300 MiB of browser binaries
```

Then in `app/.env`:

```
AGENT_ENABLE_MCP_WEB_SEARCH=1
AGENT_MCP_WEB_SEARCH_COMMAND=node
AGENT_MCP_WEB_SEARCH_ARGS=/home/<user>/.local/share/mcp-servers/web-search-mcp/dist/index.js
```

Known upstream bug: the server prints human-readable banners on stdout
(e.g. `Web Search MCP Server started`). The MCP stdio client logs these
as invalid JSON-RPC and skips them; tool loading still succeeds.
Expected tools: `full-web-search`, `get-web-search-summaries`,
`get-single-web-page-content`.

## Enable GitHub MCP for the `app` agent

Install the upstream Go binary once:

```
curl -sSL -o /tmp/gh-mcp.tar.gz https://github.com/github/github-mcp-server/releases/latest/download/github-mcp-server_Linux_x86_64.tar.gz
tar -xzf /tmp/gh-mcp.tar.gz -C ~/.local/bin github-mcp-server
chmod +x ~/.local/bin/github-mcp-server
```

Then in `app/.env`:

```
AGENT_ENABLE_MCP_GITHUB=1
AGENT_MCP_GITHUB_COMMAND=/home/<user>/.local/bin/github-mcp-server
AGENT_MCP_GITHUB_ARGS=stdio
AGENT_MCP_GITHUB_TOOLSETS=repos,pull_requests,issues,actions,context
GITHUB_PERSONAL_ACCESS_TOKEN=ghp_xxx
```

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
