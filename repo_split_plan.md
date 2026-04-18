# Repo Split Plan

## Goal

Split the current single-package repository into:

- a RAG/core package that owns indexing, storage, retrieval, and the public retrieval API
- an agent/app package that owns tool adapters, graph orchestration, chat CLI, and evaluation

This is the staging plan toward a later MCP-style deployment boundary.

## Current Boundary After Step 1

### `kms` now represents RAG/core

`kms` remains the source of truth for:

- public API: `kms.api`, `kms.types`, `kms.filters`, `kms.config`
- data pipeline: `kms.chunker`, `kms.embedder`, `kms.tagger`
- retrieval/storage: `kms.retriever`, `kms.store`
- shared LLM providers used by the core pipeline: `kms.llm`
- utilities and RAG-side CLIs: `kms.utils`, `kms.cli.ingest`, `kms.cli.query`

### `kms_agent` now represents the agent/app layer

`kms_agent` is now the source of truth for:

- agent graph/session/state/history: `kms_agent.agent`
- framework adapters: `kms_agent.adapters`
- evaluation suites: `kms_agent.evaluation`
- agent-side CLIs: `kms_agent.cli.chat`, `kms_agent.cli.eval`

### Compatibility

Legacy imports under `kms.agent`, `kms.adapters`, `kms.evaluation`, `kms.cli.chat`, and `kms.cli.eval` are currently compatibility shims.

## Why This Is The First Split

This is the lowest-risk structural cut because:

- `kms` already had a mostly self-contained RAG core
- the agent layer already depended on `kms`, but not the other way around
- extracting the agent first reduces coupling without forcing a noisy `kms -> rag` rename yet

## Next Steps

### Step 2: Rename the core package

Rename `kms` to a clearer core name such as `rag` or `kms_rag` after callers stop depending on legacy `kms.*` agent paths.

### Step 3: Extract service boundary

Wrap the core API (`search`, `explore`, `get_context`) in a thin server layer. That can be:

- MCP tools
- HTTP/JSON RPC
- a local subprocess tool server

### Step 4: Remove compatibility shims

Once all imports move to the new package names, delete the shim modules under legacy `kms.agent`, `kms.adapters`, and `kms.evaluation`.

### Step 5: Physical repo split

At that point the code can be split into separate repositories with minimal churn:

- `app/` or `agent-app/`
- `rag/`
