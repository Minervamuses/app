# Project Turning Points

Scan date: 2026-06-20
Branch: `report`
Base commit scanned: `fa1cd8d36cce81cdb74da2608e124b95eea9f2d5`
Reachable commits scanned: 279

This file is a narrative index of the repository's major turns. It is based on the full commit log in `all-commits.tsv`, the broad candidate list in `relevant-commits.md`, and the copied source documents under `source-documents/`.

## 1. KMS / Multi-layer RAG Bootstrap (2026-03-28 to 2026-03-31)

The project began as a `kms` package: chunking, embedding, vector retrieval, Chroma-backed storage, and an ingestion CLI. It quickly moved from a simple indexed knowledge store into a tool-using agent prototype.

Key turns:

- `6097e03` created the multi-layer RAG KMS scaffold.
- `67212dc` changed environment setup to conda + Poetry and added batch repo ingestion.
- `bf2d065` added multi-layer ingestion with LLM folder tagging.
- `44ae6a3` rewrote `chat.py` as an agent loop with tool calling.
- `082403a`, `3d2e20d`, `ce97fbd`, `571780f`, and `95714ea` moved search/ingest through multi-collection and then single-collection metadata-filtered designs.

Plan/spec match: no durable plan file found for this early phase; the turn is reconstructed from commit history and changed paths.

## 2. LangGraph Agentization (2026-04-02)

The custom tool loop was replaced by a LangGraph-based agent stack. Tools became LangChain `@tool` factories, `kms/agent/` introduced StateGraph/state modules, and `cli/chat.py` was rewritten around a compiled graph.

Key commits: `36effd3`, `1b0a8e9`, `8013a1c`, `ca4fd5a`, `991cf00`, `20b9b36`.

Plan/spec match: `info.md` was added shortly after (`4f9365e`, `b36f65e`) to explain the LangGraph branch and diagrams, but no separate `plan` file was found for the initial conversion.

## 3. Evaluation Harness and First Reliability Failures (2026-04-05 to 2026-04-13)

The repo added behavior, retrieval, and end-to-end evaluators, then iterated quickly on judge parsing, recursion limits, model separation, and evaluation result tracking. The 2026-04-11 report shows the first explicit reliability diagnosis: many e2e failures were either recursion-limit loops or judge JSON parsing failures.

Key commits: `1955d15`, `2b33844`, `23479b4`, `62b665a`, `012621a`, `6a37d25`, `f32c10f`, `588c791`, `508e196`, `f8ac1ff`.

Preserved records:

- `source-documents/20260411-evaluation-run-report.md`
- `source-documents/20260413-agent-state-cleanup.md`

## 4. RAG / Agent Boundary Split (2026-04-18)

This is the first clearly documented large refactor. The code split the old `kms` surface into a RAG/core package and an agent/app package, then renamed the source tree toward `rag` and `agent`, nested `rag` as its own project, split workspace environments, and finally moved RAG out of `app` as a sibling dependency.

Key commits: `f6ce5c3`, `80eafb3`, `72177bc`, `833f84d`, `e0b9eedd`, `2192c7b`, `b94d55a`, `968edf0`, `8f676ca`, `6be7f94`, `4158b0c`, `e687925`, `365fef2`, `b17b855`, `0598ec3`, `8c320a3`, `1761983`.

Preserved record:

- `source-documents/20260418-repo-split-plan.md`

## 5. MCP and Async Session Runtime (2026-04-20)

The agent learned to load MCP stdio tools for Web Search and GitHub, moved agent-only config out of RAG, made `ChatSession.create()` async, loaded `.env`, and added multiple fixes for noisy stdio servers and async-only tools.

Key commits: `2a67618`, `42cfe45`, `4e62926`, `78bc8bd`, `3357bd3`, `dcd8fc2`, `7c09d47`, `aa4ba65`, `c0b8518`, `1fc93bc`, `01faf91`, `e381ddc`, `33d011b`, `9c39917`, `613dbcb`.

Preserved record:

- `source-documents/20260420-mcp-setup.md`

## 6. Long-term Memory via History RAG (2026-04-25 to 2026-04-27)

Conversation memory changed from LLM compaction to vector-store eviction and recall. The project added `history_rag`, `recall_history`, recent-turn pruning, chat exit flushing, and eval isolation from the real chat history store.

Key commits: `d795650`, `44cd3b3`, `e07c0e2`, `07ebc6b`, `a377ad1`, `d7b5063`, `b0fc5dd`, `eed4cfa`, `259ce19`.

Plan/spec match: no original plan file found for the first history_rag implementation, but later history-recall failures generated a full spec and diagnostics on 2026-05-27.

## 7. Prompt Toolkit CLI, Slash Commands, Plan Mode, and Bash Tool (2026-05-03 to 2026-05-10)

The CLI became more interactive through `prompt_toolkit`, slash commands, `/ingest`, `/sync`, `/prune`, and `/init`. A discussion mode evolved into plan mode with markdown plan logs. A bash tool was also added with mandatory approval and graph wiring.

Key commits: `97cd1c1`, `245edef`, `7a92f40`, `086cd58`, `51765b0`, `71bbbff`, `1ee453c`, `7920ab6`, `8241ab6`, `34c95a8`, `2031752`, `7193370`, `20882ba`.

Plan/spec match: no separate plan document found; this turn is reconstructed from commits.

## 8. Skill Runtime Overhaul (2026-05-14 to 2026-05-18)

The repo added a capability broker, skill runtime loader, skill slash command, skill-aware `read_file`, graph skill-loader node, runtime toggles, tool policy enforcement, per-skill tool binding, and response validation. Follow-up fixes hardened policy errors, resource scoping, sensitive file reads, manifest validation, and deterministic validators.

Key commits: `749494e`, `b635c8b`, `e5f38ee`, `037590a`, `cda88c3`, `77b8b6d`, `692763b`, `d03857a`, `7866359`, `473c010`, `e134d86`, `a0dc7c6`, `28e8ef8`, `bb7fed6`, `a09f4b0`, `dadd58e`, `9d3e07e`, `7856576`, `7dc721d`, `df31826`, `fd882f3`.

Plan/spec match: `info.md` and `SKILLS_GUIDE.md` were updated, but no original standalone implementation plan was found.

## 9. Extended Thinking v3.4 (2026-05-24)

`/thinking extended` was redesigned into prompt rewrite + writer graph + reviewer/reviser loop, with separate model slots for rewrite/review/repair roles, prompt-master vendoring, reviewer routing, and budget controls. This phase has the strongest plan coverage in the repo.

Key commits: `59aa8e0`, `a5dc6ff`, `02a674f`, `41c6bed`, `e8cd8d5`, `cfc79fd`, `a326675`, `ddaa7b9`, `1c18c0d`, `88fb116`, `5c8dbff`, `1b47138`, `7de10c1`, `c06d9fb`, `7e609e4`, `0587c67`, `147e918`.

Preserved records:

- `source-documents/20260524-thinking-extended-plan-v3-4.md`
- `source-documents/20260524-problem.md`

## 10. History Recall / Tool Availability Repair (2026-05-25 to 2026-05-27)

The failure mode from `problem.md` was traced to a mismatch between active skill policy and actual history tool availability. The repair sequence added `history.search`, shared tool availability context, rewrite/review prompt awareness, retrieval-gap routing, and docs. It landed through PR #1 and later had diagnostic notes preserved.

Key commits: `60945d3`, `fbfe550`, `d25e40d`, `37371d4`, `352afa6`, `7917b7e`, `4db22f3`, `61b87c4`, `fdf906e`, `55e604b`.

Preserved records:

- `source-documents/20260527-history-tool-availability-spec.md`
- `source-documents/20260527-p0-5-history-query-diagnostic.md`

## 11. Evaluation System Rebuild into C1-C4 Claims (2026-05-27 to 2026-05-31)

The project rejected the old eval numbers as not trustworthy and rebuilt evaluation around four claims: routing, retrieval/faithfulness, skill + extended-thinking safeguards, and end-to-end checklist tasks. The implementation added dataset schemas, append-only ledgers, reproducibility fingerprints, C1-C4 claim runners, metrics, BEIR spike code, an eval slash command, and README documentation.

Key commits: `55e604b`, `4b20d22`, `e261583`, `700a86b`, `f2947f5`, `015ac60`, `75282a6`, `eca66b7`, `447f035`, `dd4115b`, `28e996f`, `b2d4b8d`, `2e36b7a`, `955a5cb`, `f021510`, `d0d843f`, `3a254e2`, `342304d`, `4bbaff4`, `61be291`, `3c0f9ea`, `c59807d`, `c522681`.

Preserved records:

- `source-documents/20260527-evaluator-plan.md`
- `source-documents/20260530-c1-routing-findings.md`
- `source-documents/20260531-eval-claim-run.md`

## 12. Tool-call Runaway Debug and Fix (2026-05-30 to 2026-06-15)

The C1 evaluation exposed runaway tool-calling. The investigation first suspected model/tool preference, then narrowed the root cause to a combination of corpus gaps, weak give-up discipline, in-turn context loss, parallel tool-call budget overflow, and raw response tool-call leakage. Later commits aligned tool budget with visible tool history, stripped tool calls from exhausted raw model responses, reclassified the embedding case as graceful give-up, added answer scoring, and guarded C1/behavior spec parity.

Key commits: `72ff55f`, `370d0ac`, `21510ed`, `7e8c639`, `63a09de`, `4333fa4`, `6424539`, `bea1d6e`, `468126a`, `050e07e`, `ad72c34`, `d08ba17`, `48cd5b2`, `5070cc3`, `12508b5`.

Preserved records:

- `source-documents/20260530-tool-call-runaway-fix-plan.md`
- `source-documents/20260530-tool-call-runaway-debug.md`

## 13. Extended Thinking Scope Debug (2026-05-31)

A real extended-thinking academic-writing run showed that per-graph tool caps worked, but the total extended turn could still accumulate writer + reviser + validation tool budgets. It also exposed temporal scope leakage: later research notes were semantically relevant but outside the user's requested date window.

Key commits: `aff7f99` and the surrounding 2026-05-31 docs/eval commits.

Preserved records:

- `source-documents/20260531-extended-thinking-scope-debug.md`
- `source-documents/20260531-extended-thinking-example.md`

## 14. June Consolidation: Inventory, State, LLM Contract, Full Eval (2026-06-14 to 2026-06-15)

The final scanned phase organized task cards, renamed `agent.md` to `AGENTS.md`, fixed skill frontmatter parsing with PyYAML, single-sourced base tool inventory, derived tool availability fallback from inventory, locked in history recall regression tests, centralized skill state serialization, delegated OpenRouter retries to the client, standardized chat model access, added a full eval runner, and recorded a June 15 dev eval report.

Key commits: `c94604a`, `639f48d`, `c6f277c`, `6318493`, `62fac43`, `ca26bcf`, `b0e675f`, `e61c990`, `e43baaa`, `f213ada`, `e36ef78`, `ca8d08f`, `5ae47a6`, `e23d36f`, `fa1cd8d`.

Preserved records:

- `source-documents/20260614-complexity-audit.md`
- `source-documents/20260615-skill-state-single-serializer.md`
- `source-documents/20260615-recent-changes-eval-report.md`

## Plan Coverage Summary

Strong plan/spec coverage exists for the 4/18 repo split, 5/24 extended-thinking redesign, 5/27 history/tool-availability repair, 5/27 evaluator rebuild, 5/30 runaway debug/fix, and 5/31 extended-thinking scope debug.

The phases with weaker or missing plan files are early KMS/RAG evolution, initial LangGraph conversion, initial history_rag implementation, prompt-toolkit/plan-mode CLI work, and the initial skill-runtime implementation. Those are reconstructed from commit subjects and touched paths in `all-commits.tsv` and `relevant-commits.md`.
