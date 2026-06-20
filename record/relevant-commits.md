# Refactor / Major / Debug Commit Inventory

Scanned branch history at `fa1cd8d36cce81cdb74da2608e124b95eea9f2d5` on 2026-06-20: 279 commits reachable from `HEAD`.
This inventory is intentionally broad: it includes explicit `refactor`/`fix` commits, plan/spec/report commits, large diffs, and feature commits that changed architecture, runtime, evaluation, memory, skill, thinking, or tool behavior.

## Classification Notes

- `refactor`: commit subject indicates structural rewrite, split, rename, centralization, or contract cleanup.
- `debug/fix`: conventional `fix`, diagnostic, runaway, failure, regression, hardening, guard, or graceful-give-up work.
- `plan/spec/report`: commit touched a project note, plan, spec, report, or `to_be_solved` item.
- `major feature/architecture`: feature commits that introduced or materially changed a subsystem boundary, runtime capability, memory behavior, skill behavior, thinking workflow, or evaluator.
- `large diff`: 10 or more changed paths, included to catch large turns with neutral commit messages.

Total candidate commits: 227.

| Date | Commit | Type | Subject | Paths |
| --- | --- | --- | --- | ---: |
| 2026-03-28 | `6097e03` | major feature/architecture, large diff | feat: initial project scaffold — multi-layer RAG KMS | 20 |
| 2026-03-28 | `38f1a71` | debug/fix | fix: langchain_core import path and gitignore store/ pattern | 9 |
| 2026-03-28 | `67212dc` | refactor | refactor: conda+poetry env setup and batch repo ingestion | 5 |
| 2026-03-28 | `d63e8eb` | debug/fix | fix: use absolute path for store directory | 1 |
| 2026-03-28 | `bf2d065` | major feature/architecture | feat: multi-layer ingestion with LLM folder tagging | 6 |
| 2026-03-29 | `44ae6a3` | refactor | refactor: rewrite chat.py as agent loop with tool calling | 1 |
| 2026-03-29 | `a83df85` | debug/fix | fix: use $contains for tag filtering in SearchTool | 1 |
| 2026-03-29 | `94baf9f` | debug/fix | fix: store dates as YYYYMMDD integers for ChromaDB range queries | 2 |
| 2026-03-31 | `082403a` | refactor | refactor: rewrite SearchTool for multi-collection search | 1 |
| 2026-03-31 | `4cefa72` | debug/fix | fix: tune agent system prompt to prevent over-searching | 1 |
| 2026-03-31 | `3d2e20d` | refactor | refactor: auto-discover collections from repo directory structure | 3 |
| 2026-03-31 | `ce97fbd` | refactor | refactor: replace multi-collection constants with single KNOWLEDGE_COLLECTION | 1 |
| 2026-03-31 | `571780f` | refactor | refactor: rewrite SearchTool for single-collection metadata filtering | 1 |
| 2026-03-31 | `117306b` | refactor | refactor: update tool __init__.py exports with ExploreTool and ContextTool | 1 |
| 2026-03-31 | `95714ea` | refactor | refactor: rewrite ingest for single collection with category/tags metadata | 1 |
| 2026-03-31 | `0fe692e` | refactor | refactor: use KNOWLEDGE_COLLECTION in query CLI | 1 |
| 2026-03-31 | `e711e52` | refactor | refactor: register 3 tools in chat CLI with _tool_map dispatch | 1 |
| 2026-04-02 | `a316c5f` | major feature/architecture | feat: add langgraph and langchain-openai deps | 1 |
| 2026-04-02 | `36effd3` | refactor, major feature/architecture | refactor: add get_chat_model() factory for LangGraph | 2 |
| 2026-04-02 | `1b0a8e9` | refactor | refactor: convert tools to LangChain @tool factories | 4 |
| 2026-04-02 | `8013a1c` | major feature/architecture | feat: add kms/agent/ with LangGraph StateGraph | 3 |
| 2026-04-02 | `ca4fd5a` | refactor, major feature/architecture | refactor: rewrite cli/chat.py around LangGraph compiled graph | 1 |
| 2026-04-02 | `991cf00` | refactor | refactor: remove custom BaseTool ABC | 1 |
| 2026-04-02 | `20b9b36` | refactor | refactor: remove dead ChatResponse/ToolCall and OpenRouterLLM.chat() | 3 |
| 2026-04-02 | `4f9365e` | major feature/architecture | docs: add info.md explaining langgraph branch changes | 1 |
| 2026-04-05 | `1955d15` | major feature/architecture | feat: add kms/evaluation/ module with three evaluator types | 5 |
| 2026-04-05 | `628c070` | refactor, debug/fix, major feature/architecture | fix: remove folder_prefix from retrieval evaluator _build_where call | 1 |
| 2026-04-05 | `7ddb250` | debug/fix, major feature/architecture | fix: make behavior evaluator support multi-turn test cases | 1 |
| 2026-04-05 | `04148be` | debug/fix | fix: add _extract_json helper to handle markdown code fences | 3 |
| 2026-04-05 | `1be4e81` | major feature/architecture | feat: add chunk_hit_rate metric to end-to-end evaluator | 1 |
| 2026-04-05 | `2b33844` | major feature/architecture | feat: add kms/cli/eval.py for one-command evaluation runs | 2 |
| 2026-04-05 | `23479b4` | debug/fix | fix: use result messages instead of get_state, add filter_accuracy | 1 |
| 2026-04-05 | `341126f` | debug/fix, major feature/architecture | fix: use result messages instead of get_state in e2e evaluator | 1 |
| 2026-04-05 | `82b791f` | debug/fix | fix: avoid KeyError on multi-turn cases without 'question' key | 1 |
| 2026-04-05 | `62b665a` | debug/fix | fix: increase recursion limit to 32 and catch GraphRecursionError | 1 |
| 2026-04-05 | `012621a` | refactor, major feature/architecture | refactor: reposition RetrievalEvaluator as embedding quality unit test | 1 |
| 2026-04-05 | `6a37d25` | refactor, major feature/architecture | refactor: split e2e evaluator into separate gen and judge LLMs | 1 |
| 2026-04-05 | `f32c10f` | refactor, major feature/architecture | refactor: remove RetrievalEvaluator | 2 |
| 2026-04-05 | `2678ddf` | refactor, major feature/architecture | refactor: remove retrieval suite from eval CLI | 1 |
| 2026-04-05 | `d104a97` | debug/fix | fix: increase gen_llm max_tokens from 500 to 4096 | 1 |
| 2026-04-05 | `1889eac` | debug/fix, major feature/architecture | fix: catch all agent errors in e2e evaluator gracefully | 1 |
| 2026-04-05 | `55c9c0c` | debug/fix | fix: handle None content from LLM response in OpenRouterLLM.invoke() | 1 |
| 2026-04-05 | `588c791` | refactor | refactor: track store/eval/ results while keeping chroma store ignored | 5 |
| 2026-04-11 | `508e196` | plan/spec/report, large diff | feat: add Ollama chunk quality filter and swap agent LLM to glm-5 | 13 |
| 2026-04-13 | `e757dac` | plan/spec/report | Add experiment note | 1 |
| 2026-04-13 | `f8ac1ff` | debug/fix | fix: enforce structured judge output in e2e eval | 2 |
| 2026-04-18 | `f6ce5c3` | refactor | test: add smoke baseline for decoupling refactor | 1 |
| 2026-04-18 | `80eafb3` | refactor | refactor: move _extract_date to kms/utils/paths | 4 |
| 2026-04-18 | `72177bc` | refactor | refactor: centralise folder_meta.json path in KMSConfig | 3 |
| 2026-04-18 | `833f84d` | refactor | refactor: extract where-clause builder to kms/filters | 2 |
| 2026-04-18 | `e0b9eed` | major feature/architecture | feat: add framework-neutral public API (kms.api) | 1 |
| 2026-04-18 | `2192c7b` | refactor, large diff | refactor: move kms/tool/ to kms/adapters/langchain as thin wrappers | 10 |
| 2026-04-18 | `b94d55a` | refactor, major feature/architecture | refactor: move ChatSession to kms/agent/session, decouple evaluation from cli | 4 |
| 2026-04-18 | `caa76a4` | major feature/architecture | feat: expose public API at kms top-level | 1 |
| 2026-04-18 | `d52b4d1` | refactor | chore: verify decoupled library boundaries | 0 |
| 2026-04-18 | `968edf0` | refactor, plan/spec/report, large diff | refactor: extract agent layer from kms core | 36 |
| 2026-04-18 | `8f676ca` | refactor, large diff | refactor: rename core packages to rag and agent | 74 |
| 2026-04-18 | `6be7f94` | refactor, plan/spec/report, large diff | refactor: treat repo root as a rag-agent workspace | 11 |
| 2026-04-18 | `4158b0c` | refactor | refactor(rag): stop guessing host project via __file__ | 2 |
| 2026-04-18 | `e687925` | refactor | refactor(agent): own LLM providers instead of importing rag.llm | 6 |
| 2026-04-18 | `b17b855` | refactor | refactor(rag): drop langchain-openai from rag — agent owns that helper now | 2 |
| 2026-04-18 | `0598ec3` | refactor, large diff | refactor(rag): nest package inside its own project directory | 31 |
| 2026-04-18 | `8c320a3` | refactor | build: split workspace into agent and rag — each with own conda env + poetry | 5 |
| 2026-04-18 | `0ad1922` | refactor, plan/spec/report | chore: drop stale repo_split_plan.md | 1 |
| 2026-04-18 | `1761983` | refactor, large diff | refactor: move rag out of app; depend on it via ../rag path | 33 |
| 2026-04-18 | `7b9c8d5` | debug/fix | fix: point rag path dep at ../rag in pyproject.toml | 1 |
| 2026-04-18 | `0500305` | refactor | refactor: move store/eval → eval | 8 |
| 2026-04-18 | `9121162` | major feature/architecture | feat(eval): default --output to eval/ | 1 |
| 2026-04-20 | `2a67618` | major feature/architecture | feat(agent): add turn-aware memory module with rolling compaction | 1 |
| 2026-04-20 | `42cfe45` | refactor | refactor(graph): accept optional extra_tools kwarg | 1 |
| 2026-04-20 | `4e62926` | refactor | refactor(session): turn-aware state with rolling compaction | 1 |
| 2026-04-20 | `94a7e14` | major feature/architecture | test(memory): cover turn compaction triggers and prompt assembly | 1 |
| 2026-04-20 | `593d4a4` | major feature/architecture | build(deps): add langchain-mcp-adapters and mcp | 1 |
| 2026-04-20 | `78bc8bd` | major feature/architecture | feat(agent): add MCP stdio loader for Web Search and GitHub servers | 1 |
| 2026-04-20 | `dcd8fc2` | major feature/architecture | feat(cli): async startup with MCP loading, plus .env.example and tests | 3 |
| 2026-04-20 | `7c09d47` | plan/spec/report, major feature/architecture | docs(note): MCP setup and opencode vs app-runtime distinction | 1 |
| 2026-04-20 | `aa4ba65` | refactor, large diff | refactor(config): move agent-only settings from rag to AgentConfig | 16 |
| 2026-04-20 | `c0b8518` | debug/fix, major feature/architecture | fix(mcp): silence stdio parse-error spam from noisy MCP servers | 1 |
| 2026-04-20 | `1fc93bc` | debug/fix, plan/spec/report, major feature/architecture | fix(session): make turn() async so MCP async-only tools work | 6 |
| 2026-04-20 | `01faf91` | debug/fix | fix(cli): survive GraphRecursionError instead of killing the chat session | 2 |
| 2026-04-20 | `33d011b` | debug/fix, major feature/architecture | fix(mcp): silence MCP server stderr by wrapping launch in /bin/sh | 1 |
| 2026-04-20 | `9c39917` | debug/fix, major feature/architecture | fix(mcp): filter subprocess stdout so only JSON-RPC lines reach the client | 1 |
| 2026-04-20 | `613dbcb` | debug/fix | fix(graph): turn tool exceptions into tool messages instead of crashing turn | 2 |
| 2026-04-24 | `ef2ff46` | refactor, large diff | refactor(agent): bind rag tool contract | 10 |
| 2026-04-25 | `fff1699` | refactor, debug/fix | refactor: drop dead RAG adapters, dead helper, fix chat CLI, update info title | 8 |
| 2026-04-25 | `d795650` | major feature/architecture | feat(config): add agent_recent_turns_window for upcoming history_rag eviction | 1 |
| 2026-04-25 | `44cd3b3` | major feature/architecture | feat(history_rag): add ChatHistoryStore with module-level cache | 3 |
| 2026-04-25 | `e07c0e2` | major feature/architecture | feat(history_rag): add recall_history StructuredTool factory | 3 |
| 2026-04-25 | `07ebc6b` | major feature/architecture | feat(graph): wire recall_history into the agent toolset | 3 |
| 2026-04-25 | `a377ad1` | refactor | refactor(session): replace LLM compaction with vector-DB eviction | 5 |
| 2026-04-25 | `50bb2c4` | major feature/architecture | docs(info): describe history_rag long-term memory | 1 |
| 2026-04-26 | `d7b5063` | debug/fix | fix(history): preserve unevicted turns during tool pruning | 1 |
| 2026-04-26 | `31b8620` | major feature/architecture | docs(history): clarify chat history persistence | 2 |
| 2026-04-26 | `b0fc5dd` | debug/fix | fix(session): flush recent turns on chat exit | 2 |
| 2026-04-26 | `10eed6f` | debug/fix | fix(cli): normalize quit commands before dispatch | 1 |
| 2026-04-27 | `eed4cfa` | debug/fix | fix(session): inject history store into recall tool | 5 |
| 2026-04-27 | `35b17a6` | major feature/architecture | feat(eval): load configured MCP tools | 5 |
| 2026-04-27 | `a20268d` | major feature/architecture | feat(eval): expand behavior tool routing cases | 2 |
| 2026-04-27 | `3c0f3b7` | major feature/architecture | feat(eval): record e2e tool traces | 2 |
| 2026-04-27 | `ada3ec9` | debug/fix, major feature/architecture | fix(eval): merge expected_first_tool with expected_first_tool_in instead of overwriting | 2 |
| 2026-04-27 | `7131f8b` | major feature/architecture | feat(eval): skip behavior cases when required tools are not loaded | 2 |
| 2026-04-27 | `6318fb7` | major feature/architecture | test(eval): allow rag_explore as first tool for direct-search behavior cases | 1 |
| 2026-04-27 | `259ce19` | debug/fix, major feature/architecture | fix(eval): isolate end-to-end runs from the real chat history store | 2 |
| 2026-05-03 | `97cd1c1` | refactor | refactor(cli): inject chat line reader | 2 |
| 2026-05-03 | `245edef` | major feature/architecture | feat(cli): use prompt_toolkit for chat input | 4 |
| 2026-05-03 | `7a92f40` | major feature/architecture | feat(cli): add slash commands and completion | 6 |
| 2026-05-04 | `086cd58` | major feature/architecture | feat(cli): add /ingest, /sync, /prune slash commands | 1 |
| 2026-05-04 | `a06b760` | debug/fix | fix(cli): make /clear also wipe the scrollback buffer | 1 |
| 2026-05-07 | `b0b2184` | major feature/architecture | feat(tools): add agent/tools package with read_file tool | 2 |
| 2026-05-07 | `7863868` | major feature/architecture | feat(graph): register read_file tool in chat graph | 1 |
| 2026-05-07 | `1d7ba89` | major feature/architecture | test(tools): cover read_file happy path and error cases | 1 |
| 2026-05-07 | `00e10c8` | large diff | Add local skill tracing and academic writing skill | 10 |
| 2026-05-10 | `b4e983b` | major feature/architecture | Add MCP tool family metadata | 2 |
| 2026-05-10 | `71bbbff` | refactor, plan/spec/report, major feature/architecture | Rename discussion mode to plan mode internally | 6 |
| 2026-05-10 | `7920ab2` | plan/spec/report | Log all tool results in plan log markdown | 3 |
| 2026-05-10 | `8241ab6` | plan/spec/report | Inject plan-mode hint into prompt history | 2 |
| 2026-05-10 | `34c95a8` | plan/spec/report | Gitignore plan_logs/ | 1 |
| 2026-05-10 | `2031752` | major feature/architecture | Add bash tool with mandatory user approval | 3 |
| 2026-05-10 | `7193370` | major feature/architecture | Wire bash tool into chat graph and system prompt | 3 |
| 2026-05-14 | `20882ba` | plan/spec/report, major feature/architecture | docs: refresh architecture overview to cover bash, skills, and plan mode | 1 |
| 2026-05-14 | `a3c640e` | major feature/architecture | chore(skills): add academic paper manifest | 1 |
| 2026-05-14 | `749494e` | major feature/architecture | feat(skills): add capability broker | 5 |
| 2026-05-14 | `b635c8b` | major feature/architecture | feat(skills): add skill runtime loader | 3 |
| 2026-05-14 | `e5f38ee` | major feature/architecture | feat(state): add skill runtime fields | 2 |
| 2026-05-14 | `037590a` | major feature/architecture | feat(session): wire active skill runtime | 2 |
| 2026-05-14 | `cda88c3` | major feature/architecture | feat(cli): add skill slash command | 2 |
| 2026-05-14 | `77b8b6d` | major feature/architecture | feat(tools): make read_file skill-root aware | 2 |
| 2026-05-14 | `692763b` | plan/spec/report | feat(graph): add skill loader node | 6 |
| 2026-05-14 | `d03857a` | major feature/architecture | feat(config): expose skill runtime toggles | 2 |
| 2026-05-14 | `a0dc7c6` | major feature/architecture | test(skills): add runtime adherence coverage | 1 |
| 2026-05-14 | `28e8ef8` | refactor, major feature/architecture | refactor(skills): remove static metadata block from system prompt | 4 |
| 2026-05-14 | `1580f39` | refactor | refactor(cli): drop skill startup banner | 6 |
| 2026-05-14 | `4e17416` | major feature/architecture | feat(skills): clarify user-activation model in system prompt | 1 |
| 2026-05-14 | `4cc018d` | debug/fix | fix(cli): treat blank input as redisplay, not exit | 2 |
| 2026-05-14 | `bb7fed6` | refactor, major feature/architecture | docs: rewrite info.md for skill runtime overhaul | 2 |
| 2026-05-18 | `a09f4b0` | debug/fix | fix(cli): translate skill activation errors | 2 |
| 2026-05-18 | `dadd58e` | debug/fix, major feature/architecture, large diff | fix(skills): make tool policy explicit | 13 |
| 2026-05-18 | `9d3e07e` | debug/fix, major feature/architecture | fix(skills): mark denied tools as errors | 2 |
| 2026-05-18 | `7856576` | debug/fix | fix(tools): keep skill resources scoped | 2 |
| 2026-05-18 | `6729d77` | major feature/architecture | feat(skills): cap pinned context size | 4 |
| 2026-05-18 | `7dc721d` | debug/fix | fix(tools): block sensitive file reads | 2 |
| 2026-05-18 | `df31826` | major feature/architecture | feat(skills): validate manifests | 3 |
| 2026-05-18 | `fd882f3` | refactor, major feature/architecture | refactor(skills): register deterministic validators | 2 |
| 2026-05-18 | `ad78d02` | plan/spec/report, major feature/architecture | docs(skills): expand academic writing references | 6 |
| 2026-05-18 | `72b915f` | major feature/architecture | docs(skills): align runtime documentation | 2 |
| 2026-05-24 | `59aa8e0` | major feature/architecture | Add thinking mode slash command | 4 |
| 2026-05-24 | `a5dc6ff` | major feature/architecture | Add extended thinking schemas | 2 |
| 2026-05-24 | `02a674f` | major feature/architecture | Wire extended thinking controller | 2 |
| 2026-05-24 | `41c6bed` | major feature/architecture | Add thinking reviewer eval suite | 5 |
| 2026-05-24 | `e8cd8d5` | major feature/architecture | Prioritize blocker review routing | 2 |
| 2026-05-24 | `cfc79fd` | plan/spec/report, major feature/architecture | docs(thinking): replace v2 plan with v3 design | 1 |
| 2026-05-24 | `ddaa7b9` | debug/fix, major feature/architecture | Guard extended thinking activation | 2 |
| 2026-05-24 | `88fb116` | major feature/architecture | Implement v3.4 extended thinking flow | 5 |
| 2026-05-24 | `5c8dbff` | major feature/architecture | Document extended thinking skill helper | 1 |
| 2026-05-24 | `1b47138` | major feature/architecture | Configure extended thinking models | 5 |
| 2026-05-24 | `7e609e4` | refactor, debug/fix, major feature/architecture | fix(thinking): enforce Traditional Chinese in reviewer / rewrite / repair prompts | 1 |
| 2026-05-24 | `c43ed9a` | debug/fix | fix(cli): drop description column from /skill picker | 1 |
| 2026-05-24 | `0587c67` | debug/fix, major feature/architecture | fix(config): raise thinking_reviewer_max_tokens to 4096 | 1 |
| 2026-05-24 | `147e918` | plan/spec/report | Problem to be solved saved | 3 |
| 2026-05-25 | `fbfe550` | major feature/architecture | P1a: add shared tool availability helper | 2 |
| 2026-05-25 | `d25e40d` | refactor, major feature/architecture | P1b: pass tool availability to rewrite prompts | 2 |
| 2026-05-25 | `37371d4` | major feature/architecture | P1c: pass tool availability to review prompts | 2 |
| 2026-05-25 | `352afa6` | major feature/architecture | P1d: inject tool availability into extended session | 2 |
| 2026-05-25 | `7917b7e` | major feature/architecture | P2: route retrieval gaps back to reviser | 2 |
| 2026-05-25 | `4db22f3` | major feature/architecture | P4: document skill tool availability rules | 2 |
| 2026-05-25 | `61b87c4` | debug/fix | Merge pull request #1 from Minervamuses/fix/history-tool-availability | 0 |
| 2026-05-25 | `fdf906e` | debug/fix, major feature/architecture | fix(thinking): harden review routing safeguards | 2 |
| 2026-05-27 | `55e604b` | debug/fix, plan/spec/report, major feature/architecture | docs: preserve evaluation and history availability notes | 4 |
| 2026-05-28 | `4b20d22` | major feature/architecture | eval: add dataset loader schema validation | 3 |
| 2026-05-28 | `e261583` | major feature/architecture | eval: add append-only run ledger | 2 |
| 2026-05-28 | `700a86b` | major feature/architecture | eval: add reproducibility metadata fingerprints | 2 |
| 2026-05-28 | `f2947f5` | major feature/architecture | eval: extract tool routing scorer | 4 |
| 2026-05-28 | `015ac60` | major feature/architecture | eval: include local tools in routing universe | 5 |
| 2026-05-28 | `75282a6` | major feature/architecture | eval: add c1 routing claim runner | 5 |
| 2026-05-28 | `eca66b7` | major feature/architecture | eval: add c1 claim cli entry | 2 |
| 2026-05-28 | `447f035` | major feature/architecture | eval: handle array embeddings in store fingerprints | 2 |
| 2026-05-28 | `dd4115b` | major feature/architecture | eval: add ranked retrieval metrics | 3 |
| 2026-05-28 | `28e996f` | major feature/architecture | eval: add c2 retrieval runner | 6 |
| 2026-05-28 | `b2d4b8d` | major feature/architecture | eval: wire c2 claim cli | 2 |
| 2026-05-28 | `2e36b7a` | major feature/architecture | eval: add c3 validator evaluator | 5 |
| 2026-05-28 | `955a5cb` | major feature/architecture | eval: add c3 reviewer classifier evaluator | 5 |
| 2026-05-28 | `f021510` | major feature/architecture | eval: add c3 session validation evaluator | 5 |
| 2026-05-28 | `d0d843f` | major feature/architecture | eval: wire c3 claim cli | 2 |
| 2026-05-28 | `3a254e2` | major feature/architecture | eval: add c4 checklist evaluator | 5 |
| 2026-05-28 | `342304d` | major feature/architecture | eval: wire c4 claim cli | 2 |
| 2026-05-28 | `4bbaff4` | major feature/architecture | eval: add beir scifact benchmark spike | 3 |
| 2026-05-28 | `61be291` | major feature/architecture | eval: add slash command | 2 |
| 2026-05-28 | `3c0f9ea` | major feature/architecture | docs: add evaluation package readme | 1 |
| 2026-05-30 | `a572356` | debug/fix, plan/spec/report | chore: consolidate stray notes/ into note/ convention | 1 |
| 2026-05-30 | `379e4b6` | plan/spec/report, major feature/architecture | docs: record C1 routing eval first-run findings | 1 |
| 2026-05-30 | `72ff55f` | debug/fix, major feature/architecture | feat(eval): instrument C1 runner for runaway diagnosis (Phase 0) | 1 |
| 2026-05-30 | `370d0ac` | debug/fix, plan/spec/report | docs: add fix plan for agent tool-call runaway | 1 |
| 2026-05-30 | `21510ed` | debug/fix, plan/spec/report | docs: add tool-call runaway debug note | 1 |
| 2026-05-30 | `7e8c639` | debug/fix, plan/spec/report | docs: reprioritize fix plan after Phase 0 findings | 1 |
| 2026-05-30 | `63a09de` | debug/fix | fix(agent): align tool budget with visible tool history | 4 |
| 2026-05-31 | `4333fa4` | debug/fix, plan/spec/report | fix(agent): strip tool calls from exhausted raw model response | 4 |
| 2026-05-31 | `c59807d` | major feature/architecture | eval: record dev claim run | 8 |
| 2026-05-31 | `c522681` | plan/spec/report, major feature/architecture | docs(eval): add 2026-05-31 claim run note with dataset inventory | 1 |
| 2026-05-31 | `c88b504` | major feature/architecture | Example for Extended Thinking mode added | 1 |
| 2026-05-31 | `aff7f99` | debug/fix, plan/spec/report, major feature/architecture | Document extended thinking scope behavior | 2 |
| 2026-06-14 | `c94604a` | debug/fix, plan/spec/report, large diff | docs: organize pending problem books | 13 |
| 2026-06-14 | `c6f277c` | debug/fix, plan/spec/report | fix: parse skill frontmatter with pyyaml | 3 |
| 2026-06-14 | `147754e` | refactor, plan/spec/report | chore: remove completed frontmatter task | 1 |
| 2026-06-14 | `6318493` | refactor, major feature/architecture, large diff | refactor: single-source the base tool inventory | 13 |
| 2026-06-14 | `53ef61c` | refactor, plan/spec/report | chore: close base-tool-inventory-single-source task | 1 |
| 2026-06-14 | `62fac43` | debug/fix | fix: derive tool-availability fallback from the base inventory | 4 |
| 2026-06-14 | `857ff0b` | plan/spec/report | chore: close agent-history-tool-availability task | 1 |
| 2026-06-14 | `ca26bcf` | debug/fix | test: regress archived history-recall failure scenario | 1 |
| 2026-06-14 | `b265a42` | debug/fix, plan/spec/report | chore: close agent-history-recall-user-facing-failure task | 1 |
| 2026-06-15 | `b0e675f` | refactor | refactor: centralize skill state serialization | 6 |
| 2026-06-15 | `e61c990` | debug/fix | fix(llm): delegate OpenRouter retries to client | 3 |
| 2026-06-15 | `5947563` | plan/spec/report | chore: close openrouter-retry-cleanup task | 1 |
| 2026-06-15 | `e43baaa` | debug/fix | fix(llm): forward llm_max_retries to thinking role models | 2 |
| 2026-06-15 | `6424539` | debug/fix | fix(agent): add graceful give-up rule to base tool workflow prompt | 2 |
| 2026-06-15 | `bea1d6e` | debug/fix, major feature/architecture | test(eval): reclassify C1 embedding case as graceful give-up | 1 |
| 2026-06-15 | `468126a` | major feature/architecture | feat(eval): add give-up answer scoring and progress to C1 routing | 5 |
| 2026-06-15 | `d08ba17` | debug/fix, plan/spec/report | chore: mark agent-tool-call-runaway-followups task done | 1 |
| 2026-06-15 | `48cd5b2` | debug/fix, major feature/architecture | fix(eval): sync BehaviorEvaluator embedding case to graceful give-up | 2 |
| 2026-06-15 | `5070cc3` | debug/fix, plan/spec/report, major feature/architecture | test(eval): guard against C1/behavior spec drift | 1 |
| 2026-06-15 | `12508b5` | refactor, debug/fix, plan/spec/report | chore: remove completed runaway follow-up task | 1 |
| 2026-06-15 | `f213ada` | refactor, plan/spec/report, large diff | refactor(llm): standardize chat model access | 15 |
| 2026-06-15 | `f826e41` | refactor, plan/spec/report, major feature/architecture | chore(tasks): remove solved llm access card | 1 |
| 2026-06-15 | `e36ef78` | major feature/architecture | chore(eval): add full eval runner | 1 |
| 2026-06-15 | `ca8d08f` | debug/fix, major feature/architecture | fix(eval): keep full runner ledger-only | 1 |
| 2026-06-15 | `5ae47a6` | major feature/architecture | test(eval): record full dev eval run | 8 |
| 2026-06-15 | `e23d36f` | plan/spec/report | docs(note): add june 15 eval report | 1 |
| 2026-06-15 | `fa1cd8d` | plan/spec/report | docs(note): revise june 15 report focus | 1 |
