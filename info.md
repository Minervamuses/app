# `main` branch — 架構說明

`agent/` package：LangGraph agent、tool adapters、skills runtime、extended thinking、chat / eval CLI。`rag`（建索引、儲存、檢索）是獨立 repo，在 `../rag`（github.com:Minervamuses/rag），以 Poetry path dep 引入。

兩條獨立 pipeline，唯一交會點是磁碟上的 store：**Ingest**（rag 建索引）、**Chat**（agent 對話）。

---

## Pipeline 1：Ingest

![Ingest Pipeline](ingest_pipeline.svg)

只在資料更新時跑，把 repo 檔案變成可搜尋 chunks：

1. `_collect_folders()` 掃 repo，按 `TEXT_EXTENSIONS` 過濾、`SKIP_DIRS`（含 `plan_logs`）排除，依**所在目錄**分組。`.md` 額外檢查 `do_not_index: true` frontmatter sentinel，命中即 skip。
2. `LLMTagger.tag()` 每目錄呼叫一次 LLM（路徑 + 檔名 + 預覽），產出 2–4 tags 與 summary，存 `folder_meta.json`。
3. `TokenChunker` 切 1200-token chunk，metadata 從 `folder_meta.json` 繼承 `category`（第一個 tag）與 `tags`。
4. chunk 寫入 ChromaDB collection，備份至 `raw.json`。

磁碟產物：`chroma.sqlite3`（向量索引）、`raw.json`（全文備份）、`folder_meta.json`（目錄摘要）。

`ingest_single(path)` 走單檔 upsert，同樣套 SKIP_DIRS lineage 檢查與 sentinel 檢查。

---

## Pipeline 2：Chat（`agent/graph.py`）

四 node LangGraph：

```
START → skill_loader → agent ⇄ tools (PolicyToolNode)
                         ↓
                    skill_validator → (END 或回 agent)
```

- **skill_loader**：有 active skill 而 state 未注入時，把 `SkillRuntime` 的 instructions / pinned references / allowed / denied / tool_policy_active 拷進 state；否則 no-op。
- **agent**：以 `(active_skill, task_mode, tool_policy_active, allowed, denied)` 為 key 的 LRU cache 取 rebound model（依當前 skill 過濾 tools）。每輪在 prompt 尾端注入 `[Tool budget]` SystemMessage 告知已用 / 上限（`agent_max_tool_interactions`，預設 4）；用滿時改注入 `[Tool budget exhausted]` 並停叫工具。`_cap_tool_calls` 裁掉超出 budget 的並行 tool call，保持協定合法。
- **tools (PolicyToolNode)**：包在內建 `ToolNode` 外。`tool_policy_active=False` → 直接 delegate；`True` → 拆 allowed / denied（deny 優先於 grant），denied 回 `ToolMessage(status="error")` 對齊原 call ID。tool 例外包成 `Tool error: ...` 回 agent，不炸整輪。
- **skill_validator**：僅在有 active skill 且最終 AIMessage 無 tool_calls 時跑確定性檢查；違規且 `validation_attempts < skill_max_validation_retries` → 注入 `[Skill validation errors]` 回 agent，否則 END。

### 六個工具家族

base session 工具集。skill 啟用且 `tool_policy_active=True` 時，schema binding 與 dispatch 改以 runtime 的 `allowed_tools` / `denied_tools` 為準，writer 另收 ephemeral `[Tool availability]` hint。

**① 本地知識庫**（由 `rag.TOOL_SCHEMAS` + `rag.dispatch` 生成）

| Tool | 讀什麼 | 用途 |
|------|--------|------|
| `rag_explore` | `folder_meta.json` | 列 categories / tags / date 範圍 / 目錄 summary |
| `rag_search` | ChromaDB | 語義搜尋，支援 `folder_prefix` / `category` / `file_type` / `date_from` / `date_to` |
| `rag_get_context` | `raw.json` | 以 `pid` 取同檔 chunks，回傳目標前後 N 個 |
| `rag_list_chunks` | `raw.json` | 不做 embedding 直接列舉；保留給 eval / audit，預設不綁進 chat |

**② 對話歷史**

| `recall_history` | `<persist_dir>/chat_history/` ChromaDB | 語義搜尋已持久化、不在當前 prompt 的舊對話；可選 `role` filter |

非 rag_search 替代品；KB 文件問題仍走 rag_search。對 skill author 是兩個 capability：`rag.search`（查 indexed KB）vs `history.search`（查 chat history）。

**③ 本地檔案**

| `read_file` | 讀單一 UTF-8 文字檔（≤1 MB）。**active skill 下**，`references/`/`assets/`/`scripts/` 開頭相對路徑只 resolve 到 `skill_root`，找不到報錯不 fallback；其他相對路徑走 cwd。擋 path traversal 與敏感路徑 denylist。只能讀檔，列目錄用 `bash` |

**④ Shell**

| `bash` | 執行 shell command，**每次呼叫**跳 `Approve? [y/N]` |

預設 deny（y/yes 以外皆拒）；**非 TTY（eval / pytest / pipe）自動拒絕**；`subprocess.run(shell=True, stdin=DEVNULL, cwd=app_root)`，timeout 30s（上限 300s），輸出 >256 KB 截斷；拒絕時 LLM 收 `{"approved": false, ...}`。

**⑤ Web Search MCP**（env 開關；目前啟用）— `mrkrsl/web-search-mcp` 子進程（Playwright）

| `full-web-search` | 搜尋 + 抓全文（最吃 token） |
| `get-web-search-summaries` | 只回 snippets（便宜） |
| `get-single-web-page-content` | 抽單一 URL 內文 |

**⑥ GitHub MCP**（env 開關；預設關閉）— `github/github-mcp-server` 子進程 + PAT，讀遠端 repos / PR / issues / actions。非本地 git 操作替代品。

### Tool selection policy（system prompt）

專案 / 研究筆記 → 本地 KB；提到不在 prompt 的舊對話 → `recall_history`；本地檔案 → `read_file`；列目錄 / utility → `bash`（必填 description）；當下外部資訊 → Web Search MCP；遠端 GitHub 狀態 → GitHub MCP；某家族未綁（env 關）→ 視為不存在。

### 工具註冊三層

1. **功能綁定** — `graph.py` 依 active skill 動態 filter 後 `model.bind_tools(...)`，LRU cache 避免每輪 rebuild。
2. **Dispatch 把關** — `PolicyToolNode` 在執行前再 check allow/deny，防 cached schema 或 prompt injection。
3. **語意提示** — `session.py:SYSTEM_PROMPT` 描述各家族；active skill 下用 ephemeral `[Tool availability]` 覆蓋。

加新工具：`agent/tools/<name>.py` factory → `tools/__init__.py` re-export → `graph.py` import + append → `session.py` SYSTEM_PROMPT 加段 → `tests/test_<name>_tool.py` → 更新 `tests/test_mcp.py` 預期 bound 列表。

### Language policy（system prompt）

以使用者輸入語言回應；中文一律繁體，**即使輸入混簡體也不輸出簡體**；其他語言直接 match。

---

## 長期記憶（`agent/memory.py`）

不無限塞 prompt、也不走 LLM 壓縮。兩層：固定 system prompt + recent turns（`agent_recent_turns_window`，預設 10 輪）。

`TurnRecord.persist_target`（`chroma` / `plan_log` / `none`）決定落地：
- 正常：`chroma`，eviction 時 `ChatHistoryStore.add_turn()` 寫 `<persist_dir>/chat_history/`
- Plan：`plan_log`，per-turn 即時 append 到 md，eviction noop

每輪後 `_evict_overflow()` 超出 window 就把最舊輪交 `_store_turn()` 分派，寫成功才移除；CLI 結束（`q`/EOF/Ctrl-C）呼叫 `flush_recent_turns()` 收尾。embedding 用本地 Ollama bge-m3（無 OpenRouter 成本），原文逐字保留，同 persist dir 跨 session 累積。

失敗處理：chroma add 例外 → warning，留 recent_turns 下輪重試；連續失敗超 `window*3` → error 丟最舊輪防膨脹；plan log 寫入失敗 → 整輪視為失敗（不入 recent_turns、不遞增 counter），例外向 CLI 拋出。

---

## Plan mode（`/mode plan`）

| 維度 | 正常 | Plan |
|---|---|---|
| 對話落地 | ChromaDB `chat_history` | `plan_logs/plan-{session_id}-{ts}.md` |
| 工具結果 | 只進 prompt | **完整渲染進 md**（args + content；超 `plan_log_max_tool_chars` 只截 md，不影響 LLM context） |
| 跨 session 索引 | `recall_history` 可搜 | **永不入 chroma**（gitignored + SKIP_DIRS + sentinel 三層防護） |

切 mode **不清空 recent_turns**，各輪保有原 persist_target。當 recent_turns 含 `plan_log` turn 時，`_prompt_history()` 插入 ephemeral mode hint 提醒 LLM「那些 turn 在 plan_logs/、別用 recall_history 找」。`/mode` 是 framework：`_MODE_REGISTRY` = `{name → ModeSpec(enter, exit)}`；無參數 = 互動選單，`/mode <name>` = one-shot。

---

## Extended thinking（`/thinking`，`agent/thinking.py`）

跟 plan / skill 正交的多段推理層。`/thinking normal` = 預設直接 agent flow；`/thinking extended` 包住 base turn 跑 rewrite + reviewer/reviser loop（切 extended 前 `require_thinking_models` 檢查角色模型是否齊備，缺則報錯）：

1. **Rewrite** — `_prompt-master` persona 把 user prompt 改寫成給 agent 的乾淨指令；事實不足時走 `<<CLARIFY>>` 回問使用者，不自行補 citation / data。
2. **Draft** — base agent 依改寫後 prompt 產候選答案（含工具）。
3. **Review** — reviewer model 對照 user intent、skill policy、tool availability、evidence trace 回結構化 `ReviewReport`（decision pass/revise/block + findings，每個 finding 帶 `failure_mode`）。`route_review_report()` 依 failure_mode 路由：recoverable（retrieval_not_attempted / empty）→ revise；user-blocking（tool_unavailable / user_input_missing）或 fabrication_risk → ask_user。
4. **Revise** — 要改時 rewrite model 產最終答案（DRAFT / REBUTTAL 分段，內部討論留 REBUTTAL 不外洩）；reviewer / reviser 輸出 malformed 時由 repair model 修復，修不好則保守 fallback 並加格式警告。上限 `MAX_REVIEW_ATTEMPTS=2`。

角色模型在 config：reviewer `anthropic/claude-haiku-4.5`、rewrite / repair `openai/gpt-5-mini`，各有 context cap。離線 reviewer / session 評估在 `agent/evaluation/`（eval 的 C3b/C3c）。

---

## Skills（user-activated runtime）

使用者顯式啟用的工作流，非模型自選提示。目錄：

```
skills/<name>/
  SKILL.md          # 工作流指示
  manifest.yaml     # capabilities / task_modes / tool_policy
  references/*.md    # 可標 pinned: true 啟用時預載
```

啟用流程：
1. `/skill` → picker 列各 SKILL.md 的 name / description（`[0] none` 停用）。manifest 有 `task_modes` 則二段選；`/skill <name> [mode]` 為 one-shot；錯誤轉 `SlashCommandError` 不炸 loop。
2. `activate_skill()`：讀 SKILL.md → schema validate manifest → 經 capability broker 把 capability 映射到實際 tool（含 MCP family），required 無法解析則 fail fast → load pinned references（檢查單檔 / total 上限）→ 組 `SkillRuntime`。
3. 每輪 `_prompt_history()` 插兩條 ephemeral SystemMessage：`context_block()`（`[Active skill]` + SKILL.md 全文 + pinned refs）與 `[Tool availability]`。皆不持久化。
4. `skill_loader_node` 每輪把 policy 拷進 state；agent filter + rebind；PolicyToolNode 再 enforce。只有 `tool_policy_active=False` 才代表 no policy。
5. `skill_validator_node` 對最終答案跑 per-skill 確定性檢查。

**Capability map**（`agent/skills/capability_map.yaml`，啟動載一次 frozen）把抽象 capability 映射到 tool / MCP family，manifest 只宣告需求不綁 implementation：

```yaml
file.read:      { local_tools: [read_file] }
rag.search:     { local_tools: [rag_search, rag_explore, rag_get_context] }
history.search: { local_tools: [recall_history] }
web.search:     { mcp_families: [web_search] }
shell.execute:  { local_tools: [bash] }
```

**禁止繞道**：`SkillRuntime.read_skill_resource()` 強制 join `skill_root` + `is_relative_to` guard；`read_file` 在 active skill 下對 bundle 相對路徑只解析到 skill_root。skills 不自動進 prompt（不 inject metadata、不印 banner），模型對「有哪些 skill」色盲，被問起時 `bash ls skills/` 現查。

### 已安裝 skills

| Skill | capabilities (required / optional) | task_modes | tool_policy |
|---|---|---|---|
| `academic-paper-writing` | file.read, rag.search, history.search / web.search | revision, literature-review, drafting, submission-support | disallow: bash |
| `_prompt-master` | 無（純生成） | — | disallow: 全部 tool |

**`academic-paper-writing`**：把 agent 約束成學術寫作工作流（read before writing → 判斷任務類型 → 抓核心貢獻 → 先修高槓桿問題 → integrity check）。不編造 data / citation / DOI；缺資料用 `[insert citation]` 等 placeholder。validator 有 academic-specific 檢查（如百分比敘述須有 citation marker）。References：`section-playbooks.md`（pinned）、`literature-review.md`、`reporting-guidelines.md`、`qualitative-research.md`、`submission-and-integrity.md`。

**`_prompt-master`**：prompt engineering skill，manifest disallow 所有 tool（純文字生成，不碰 KB / shell / 檔案）。references 為 `patterns.md` / `templates.md`。同一份 persona 也被 extended thinking 的 rewrite step 重用（`session._prompt_master_skill_text()`）。

---

## Session lifecycle（async）

MCP tools 是 async `StructuredTool`，整條路徑 async：

- `ChatSession.create(config, load_mcp=True)` — async factory，啟 MCP 子進程（`load_mcp_tools_with_families()` 同時拿 tool list 與 `tool_name → family` map）、build graph
- `await session.turn(user_input)` — `graph.astream(stream_mode="updates")` 跑一輪，邊跑邊餵 `progress_cb`（chat CLI 印 `→ calling <tool>` / `✓ <tool> returned`）
- `await session.flush_recent_turns()` — CLI 結束前呼叫

**MCP 子進程兩個 quirk**（`agent/mcp.py` 處理）：① stderr 灌 debug log → 導到 `~/.cache/agent-mcp/<server>.stderr.log`；② stdout 摻非 JSON 行致 `BrokenResourceError` → shell pipeline 掛 `grep '^{'` 只放行 JSON-RPC。

---

## CLI

```bash
python -m agent.cli.chat            # 互動 chat
python -m agent.cli.eval --suite behavior
python -m agent.cli.eval --claim c1 --split dev --allow-skips
```

`chat` 參數：`--no-mcp`（只綁本地 tool：rag 三個 + recall_history + read_file + bash）、`--max-turns N`（單輪 recursion 深度上限，預設 32）。

### Slash commands

| 命令 | 用途 |
|---|---|
| `/help` | 列所有命令 |
| `/status` | session_id、turn / recent-turn count、recursion limit、last tool counts、plan_mode、thinking_mode、active_skill、task_mode |
| `/mode [name]` | 切 session mode（`plan`；`normal` = 退出；無參 = 選單） |
| `/thinking [name]` | 切 thinking mode（`normal` / `extended`；無參 = 選單） |
| `/skill [name [mode]]` | 啟用 / 切換 / 停用 skill（`/skill none` = 停用） |
| `/eval <c1\|c2\|c3\|c4> [dev\|test] [--allow-skips]` | 跑單一確定性 eval claim |
| `/init` | Ingest 父 repo（排除本 app 目錄） |
| `/ingest <path>` | Upsert 單檔或目錄到 rag store |
| `/sync [path]` | dry-run 顯示 store 與磁碟差異 |
| `/prune [path] [--yes]` | 移除 store 裡 source 已不存在的 entry |
| `/clear` | 清螢幕 |
| `/quit`（alias `/exit`） | 離開 |

rag 端 `ValueError`（如 plan_logs 被 SKIP_DIRS / sentinel 擋）翻成 `SlashCommandError` 顯示，不炸 loop。空白 input 重顯 `>>`；退出僅 `q`/`quit`/`exit`/`/quit`/EOF/Ctrl-C。

---

## Evaluation（`agent/evaluation/`）

- **Suites**（`--suite`）：`behavior`、`e2e`、`thinking`。
- **Claims**（`--claim c1..c4 --split dev|test`）：C1 工具路由、C2 檢索（recall@k / MRR / nDCG）、C3 驗證+審稿+session（a/b/c）、C4 端到端 checklist。資料集在 `eval/datasets/c{1-4}/{dev,test}.jsonl`，結果寫 `eval/runs/`（摘要 `c{n}.jsonl` + 明細 `details/`），append-only ledger + store fingerprint 守 reproducibility。

---

## 配置（`agent/config.py:AgentConfig` ← `rag.RAGConfig`）

| 欄位 | 預設 | 說明 |
|---|---|---|
| `llm_model` | `deepseek/deepseek-v4-pro` | Chat 主模型 |
| `llm_max_tokens` | 4096 | |
| `gen_llm_model` | `google/gemini-3.1-pro-preview` | Eval generator |
| `judge_llm_model` | `openai/gpt-5.2` | Eval judge |
| `filter_llm_model` | `llama3.1:8b` | Eval filter |
| `thinking_reviewer_model` | `anthropic/claude-haiku-4.5` | extended thinking reviewer |
| `thinking_rewrite_model` / `thinking_repair_model` | `openai/gpt-5-mini` | rewrite / repair |
| `thinking_*_chars` | — | thinking 各段 context cap |
| `agent_max_messages` | 20 | 同輪 prompt message 上限 |
| `agent_max_tool_interactions` | 4 | 同輪 tool round-trip 上限 |
| `agent_recent_turns_window` | 10 | prompt-visible 對話 window |
| `plan_logs_dir` | `plan_logs` | plan 模式 md 目錄 |
| `plan_log_max_tool_chars` | 65536 | 單 ToolMessage 寫 md 軟 cap |
| `skills_dir` | `None` | None → `<repo>/skills` |
| `skill_validation_enabled` | `True` | 是否跑 validator node |
| `skill_max_validation_retries` | 1 | validator 違規回改寫次數 |
| `skill_capability_map_path` | `None` | None → `agent/skills/capability_map.yaml` |
| `skill_max_pinned_reference_chars` | 65536 | 單一 pinned reference 上限 |
| `skill_max_total_skill_context_chars` | 200000 | skill context 總上限 |
