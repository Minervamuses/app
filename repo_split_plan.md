# Repo Split Plan

## Goal

Treat this repository as a neutral workspace with two peer packages:

- a RAG/core package that owns indexing, storage, retrieval, and the public retrieval API
- an agent/app package that owns tool adapters, graph orchestration, chat CLI, and evaluation

This is the staging plan toward a later MCP-style deployment boundary.

## Current Boundary

### `rag` represents RAG/core

`rag` is the source of truth for:

- public API: `rag.api`, `rag.types`, `rag.filters`, `rag.config`
- data pipeline: `rag.chunker`, `rag.embedder`, `rag.tagger`
- retrieval/storage: `rag.retriever`, `rag.store`
- shared LLM providers used by the core pipeline: `rag.llm`
- utilities and RAG-side CLIs: `rag.utils`, `rag.cli.ingest`, `rag.cli.query`

### `agent` represents the agent/app layer

`agent` is the source of truth for:

- graph/session/state/history: `agent.graph`, `agent.session`, `agent.state`, `agent.history`
- framework adapters: `agent.adapters`
- evaluation suites: `agent.evaluation`
- agent-side CLIs: `agent.cli.chat`, `agent.cli.eval`

## What Changed In This Stage

- the old `kms` / `kms_agent` names are gone from the source tree
- the repo no longer assumes the enclosing project is itself the agent app
- the workspace root now exists to host two peer packages rather than one package containing the other

## Remaining Work

### Step 3: Extract service boundary

Wrap the core API (`search`, `explore`, `get_context`) in a thin server layer. That can be:

- MCP tools
- HTTP/JSON RPC
- a local subprocess tool server

### Step 4: Physical repo split outside the nested `app/` checkout

The code is now structured as peers inside this repo, but the enclosing
filesystem directory is still named `app/`. Moving the checkout itself to a
new sibling path under `PiDNA2/` is a parent-repo/filesystem operation rather
than a git-tracked code change inside this repository.

When that happens, the target layout should look like:

- `agent/` or `agent-app/`
- `rag/`
