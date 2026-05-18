# `main` branch — 架構說明

這個 repo 是 **agent** 專案，放 `agent/` package：LangGraph agent、tool adapters、skills runtime、chat / eval CLI。`rag`（建索引、儲存、檢索、public retrieval API）是獨立 repo，在 `../rag`（github.com:Minervamuses/rag），透過 Poetry path dep 引入。

系統分成兩條獨立 pipeline：**Ingest**（rag 負責建索引）和 **Chat**（agent 驅動對話），唯一的交會點是磁碟上的 store。

---

## Pipeline 1：Ingest（建索引）

![Ingest Pipeline](ingest_pipeline.svg)

這條 pipeline 只跑一次（或資料更新時重跑），把 repo 裡的檔案變成可搜尋的 chunks。流程四步：

1. `_collect_folders()` 掃描 repo，按 `TEXT_EXTENSIONS` 過濾、按 `SKIP_DIRS`（含 `plan_logs`）排除，把檔案按**所在目錄**分組。對 `.md` 檔額外檢查 `do_not_index: true` frontmatter sentinel；命中就 skip — 雙保險防止 plan-mode 對話紀錄回流到 KB。

2. `LLMTagger.tag()` 對每個目錄呼叫一次 LLM，輸入是目錄路徑 + 檔名 + 檔案預覽，輸出是 2–4 個 tags 和一段 summary，存進 `folder_meta.json`。

3. `TokenChunker` 把每個檔案切成 1200 token 的 chunk。切完之後，每個 chunk 的 metadata 會從 `folder_meta.json` 繼承 `category`（第一個 tag）和 `tags`。

4. 所有 chunk 寫入 ChromaDB collection，同時備份到 `raw.json`。

磁碟上三個檔案：`chroma.sqlite3`（向量索引）、`raw.json`（全文備份）、`folder_meta.json`（目錄摘要）。

`ingest_single(path)` 走獨立路徑（單檔 upsert），同樣套用 SKIP_DIRS lineage 檢查與 sentinel 檢查 — 任何位置的 `do_not_index` md 都會被拒。

---

## Pipeline 2：Chat（Agent 查詢）

`agent/graph.py` 是個四個 node 的 LangGraph：

```
START → skill_loader → agent ⇄ tools (PolicyToolNode)
                         ↓
                    skill_validator → (END 或回 agent)
```

**skill_loader**：如果 session 有 active skill 而 state 還沒注入，就把 `SkillRuntime` 的 `instructions` / `pinned_references` / `allowed_tools` / `denied_tools` / `tool_policy_active` 拷貝進 state。沒 active skill 時是 no-op。

**agent**：用「目前 active skill 的 tool 過濾結果」rebound model（per-turn cache key 是 `(active_skill, task_mode, tool_policy_active, allowed, denied)`），把 `_prompt_history()` 組好的 messages 丟給 LLM，回傳「純文字」或「tool call 請求」。

**tools (PolicyToolNode)**：包在 LangGraph 內建 `ToolNode` 外面。`tool_policy_active=False` → 直接 delegate；`tool_policy_active=True` → 把 LLM 的 tool_calls 拆成 allowed / denied 兩堆，deny rules 優先於 grants。denied 那堆生成 `ToolMessage(content="Tool error: denied by active skill policy: ...", status="error")`、`tool_call_id` 對齊原本的 call ID，allowed 那堆走真正的 ToolNode。tool 拋例外時會被包成 `Tool error: ...` 丟回 agent，**不會炸掉整輪對話**。

**skill_validator**：只在 active skill 存在、且當前 AIMessage 沒有 tool_calls（= 最終回答）時跑。執行確定性檢查（regex 抓百分比後無 citation marker 之類），命中違規且 `validation_attempts < skill_max_validation_retries` → 注入 `[Skill validation errors]` SystemMessage 並 route 回 agent；否則 → END。

### 六個工具家族

Agent 啟動時把下列所有 tool 綁到 LLM；LLM 依照 system prompt 的「tool selection policy」自行決定該用哪個。

**① 本地知識庫（永遠可用）**

由 `rag.TOOL_SCHEMAS` + `rag.dispatch(...)` 生成 LangChain tools；chat graph 不再維護一份獨立的 RAG tool schema。一般 chat 預設只綁互動檢索需要的三個 tool，`rag_list_chunks` 保留給 eval / audit 類內部流程。

| Tool | 讀什麼 | 用途 |
|------|--------|------|
| `rag_explore` | `folder_meta.json` | 列出 categories、tags、date 範圍、每個目錄的 summary |
| `rag_search` | ChromaDB 向量索引 | 語義搜尋，支援 `folder_prefix` / `category` / `file_type` / `date_from` / `date_to` filter |
| `rag_get_context` | `raw.json` | 用 `pid` 找到同一檔案的所有 chunks，回傳目標 chunk 前後 N 個 |
| `rag_list_chunks` | `raw.json` | 不做 embedding，直接列舉 chunks；預設不綁進 chat |

**② 對話歷史（永遠可用）**

| Tool | 讀什麼 | 用途 |
|------|--------|------|
| `recall_history` | `<rag persist_dir>/chat_history/` ChromaDB | 語義搜尋已成功持久化、不在目前 prompt 裡的舊對話內容；可選 `role` filter |

詳細生命週期見「長期記憶」段。**不是 rag_search 的替代品** — 知識庫文件問題仍走 rag_search。

**③ 本地檔案（永遠可用）**

| Tool | 用途 |
|------|------|
| `read_file` | 讀單一 UTF-8 文字檔。預設是 absolute / cwd-relative path；**active skill 存在時**，`references/`、`assets/`、`scripts/` 開頭的相對路徑只會 resolve 到 `skill_root`，找不到就回 error，不會 fall back 到 cwd；其他相對路徑仍走 cwd。擋 path traversal、敏感路徑 denylist，1 MB 上限 |

讀本地草稿、reviewer comments、plan log、active skill bundle 裡的 reference 檔等。**只能讀檔，不能列目錄** — 列目錄要用 `bash`。

**④ Shell（永遠可用，但每次呼叫都要批准）**

| Tool | 用途 |
|------|------|
| `bash` | 執行 shell command。**每次呼叫**都會跳 prompt 讓使用者 y/n 批准 |

設計：
- 工具入參含必填 `description`，使用者讀這個來決定批准與否
- 預設 deny：空輸入或 y/yes 以外都視為拒絕，prompt 文字明示 `Approve? [y/N] (Enter = no)`
- **非 TTY 環境（eval / pytest / pipe）自動拒絕** — 不會 hang 也不會意外執行
- `subprocess.run(shell=True, stdin=DEVNULL, cwd=app_root)`；timeout 預設 30s 上限 300s；輸出超過 256 KB 截斷
- 拒絕時 LLM 收到 `{"approved": false, "error": "..."}`，依 system prompt 不該固執重試

**⑤ Web Search MCP（由環境變數開關；目前啟用）**

透過 `mrkrsl/web-search-mcp` 子進程，背後跑 Playwright 開 Chromium/Firefox 做搜尋。三個 tool：

| Tool | 用途 |
|------|------|
| `full-web-search` | 搜尋 + 抓每個 hit 的全文，吃 token 最多 |
| `get-web-search-summaries` | 只回傳搜尋 snippets，不抓全文，便宜很多 |
| `get-single-web-page-content` | 把一個 URL 的內文抽出來 |

**⑥ GitHub MCP（由環境變數開關；預設關閉）**

透過 `github/github-mcp-server` 子進程，用 PAT 讀遠端 GitHub 狀態。啟用時 agent 會多出 repos / pull_requests / issues / actions / context 等工具。**不是 git 本地操作的替代品** — clone / pull / commit 還是在終端做。

### Tool selection policy（寫在 system prompt）

- 問題是關於這個專案本身或研究筆記 → 用本地 KB（rag_explore / rag_search / rag_get_context）
- 問題提到之前 chat 講過的事，但已不在 prompt 裡 → 用 `recall_history`
- 問題是關於本地檔案 → 用 `read_file`
- 需要列目錄、找檔、跑 utility 等 shell 操作 → 用 `bash`（必填一句話 description）
- 問題需要當下的外部資訊 → 用 Web Search MCP
- 問題是遠端 GitHub 狀態（PR、issue、CI） → 用 GitHub MCP
- 某個家族沒綁上來（環境變數關掉）→ 當它不存在，用手上有的

### 工具註冊三層機制

1. **功能綁定** — `agent/graph.py` build 一個 list，**按 active skill 動態 filter** 後餵給 `model.bind_tools(...)`（送 schema 給 LLM）。`(active_skill, task_mode, tool_policy_active, allowed, denied)` 為 key 的 LRU cache 避免每 turn rebuild
2. **Dispatch 把關** — `PolicyToolNode` 在 tool 真正執行前再 check 一次 allow/deny list；防止模型用緩存 schema、或被 prompt injection 誘導 call denied tool
3. **語意提示** — `agent/session.py:SYSTEM_PROMPT` 列出每個工具家族的描述、選用時機、注意事項

三層缺一不可：第一層讓 LLM 看到精確 schema，第二層 enforce runtime policy（即使第一層被繞過），第三層讓 LLM 知道何時該選哪個。

加新工具的最小步驟：寫 `agent/tools/<name>.py` 提供 factory → `agent/tools/__init__.py` re-export → `agent/graph.py` import + append 到 tools list → `agent/session.py` SYSTEM_PROMPT 加一段 → 寫 `tests/test_<name>_tool.py` → 更新 `tests/test_mcp.py` 預期 bound 列表。

### Language policy（寫在 system prompt）

模型必須以使用者輸入語言回應；中文一律使用繁體中文，**即使使用者輸入混雜簡體字也不能輸出簡體**；其他語言則直接 match。這條規則寫死在 SYSTEM_PROMPT 裡，所有 turn 都套用。

### 長期記憶（vector-DB chat history）

對話歷史不是無限塞進 prompt，但也不走 LLM 壓縮。`agent/memory.py` 是兩層結構：

1. **固定 system prompt**
2. **recent turns**（最近 `config.agent_recent_turns_window` 輪，預設 10）

`TurnRecord` 帶 `persist_target` 欄位（`"chroma"` / `"plan_log"` / `"none"`），決定該 turn 落地到哪：
- 正常模式：`chroma`，eviction 時走 `ChatHistoryStore.add_turn()` 寫入 `<rag persist_dir>/chat_history/`
- Plan 模式：`plan_log`，per-turn 即時 append 到 `plan_logs/plan-{session_id}-{ts}.md`，eviction 時 noop

每完成一輪後 `_evict_overflow()` 檢查 `recent_turns`：超過 window 就把最舊那輪交給 `_store_turn()` 走 dispatcher 分派。寫入後才從 `recent_turns` 移除；CLI 收到 `q` / EOF / Ctrl-C 結束時呼叫 `flush_recent_turns()` 把剩下完成的 turn 處理完。

**沒有 LLM 壓縮成本**：embedding 用本地 Ollama bge-m3，沒有 OpenRouter 呼叫。**內容不失真**：原文逐字保留，需要時透過 `recall_history` 或讀 plan log 取回。**跨 session 累積**：同一個 persist dir 永遠只有一個 chat_history collection。

**失敗處理（明示）**：
- chroma `add_turn` 拋例外 → `logger.warning`，turn 留在 `recent_turns` 下輪再試
- 連續失敗導致 `recent_turns` 超過 `window * 3` → `logger.error`，丟棄最舊那輪防止無限膨脹
- plan log md 寫入失敗 → 整個 turn 視為失敗，**不入 recent_turns 也不遞增 turn_counter**，例外向 CLI 拋出

### Plan mode（`/mode plan`）

研究討論用的特殊模式，跟正常模式並存：

| 維度 | 正常模式 | Plan 模式 |
|---|---|---|
| 對話落地 | ChromaDB `chat_history` | `plan_logs/plan-{session_id}-{ts}.md` |
| 工具結果 | 只進 prompt context | **完整渲染進 md**（args + ToolMessage content；單個結果超過 `plan_log_max_tool_chars` 截斷只影響 md，不影響 LLM context） |
| 跨 session 索引 | 可被 `recall_history` 搜到 | **永不入 chroma**（gitignored + rag SKIP_DIRS + frontmatter sentinel 三層防護） |
| 切換時機 | `/mode normal` | `/mode plan` |

切 mode 時**不清空 `recent_turns`** — 切換前的 turn 保有原本 persist_target，切換後新 turn 用新 target，eviction 各自走各自的目的地。

**Mode hint 注入**：當 `recent_turns` 含 `persist_target="plan_log"` 的 turn 時，`_prompt_history()` 會在 system prompt 之後動態插入一條 ephemeral `SystemMessage`，提醒 LLM「最近某些 turn 在 plan_logs/、不在 chroma，不要呼叫 recall_history 找它們」。Hint 不持久化，turn 之間動態加減。

`/mode` 是個 framework：`agent/cli/slash_commands.py:_MODE_REGISTRY` 是 `{name → ModeSpec(enter, exit)}` 的 dict。`/mode` 無參數 → 互動選單；`/mode <name>` → one-shot 切換。

### Skills（user-activated runtime）

Skills 是「使用者顯式啟用」的工作流，**不是模型自選的提示**。架構與 Claude Skill 體感的差距，已經透過下面這層 runtime 補齊。

**目錄結構：**

```
skills/<name>/
  SKILL.md            # 工作流指示
  manifest.yaml       # capabilities、resources、task_modes、tool_policy
  references/*.md     # 進階參考（可標 pinned: true 在啟用時預載）
```

**啟用流程：**

1. 使用者 `/skill` → 互動 picker 列出 `skills/<name>/` 下所有 SKILL.md 的 name 與 description（只給使用者選，不寫進 prompt）。`[0] none` 用來 deactivate。若 skill manifest 宣告 `task_modes`，picker 走二段：先選 skill，再選 task mode。`/skill <name> [mode]` 為 one-shot；輸入錯誤會轉成 `SlashCommandError`，不炸掉 chat loop。
2. `session.activate_skill(name, mode)` 同步：讀 SKILL.md → 解析並 schema validate manifest → 透過 capability broker 把宣告的 capability 對應到實際 tool 名稱（含 MCP family）→ required capability 無法解析就 fail fast → load 所有 `pinned: true` 的 reference 並檢查單檔 / total context 上限 → 組 `SkillRuntime` 寫進 `session.active_skill_runtime`。
3. 之後每 turn，`_prompt_history()` 在 system prompt 之後插入一條 ephemeral `SystemMessage`（內容是 SkillRuntime.context_block()），帶 `[Active skill]` 標題、SKILL.md 全文、pinned references。Hint 不持久化。
4. graph `skill_loader_node` 在每 turn 開頭把 SkillRuntime 的 `tool_policy_active` / `allowed_tools` / `denied_tools` 拷貝進 state；agent_node 依此 filter tools 重新 bind；PolicyToolNode 在 dispatch 層再 enforce 一次。只有 `tool_policy_active=False` 才代表 no policy；不能再用 allowed / denied 是否同時為空推論。
5. skill_validator_node 在最終 AIMessage 上跑 deterministic checks（per-skill rule）；違規且未超 retry 上限 → 回 agent 改寫；否則 → END。

**Capability map（`agent/skills/capability_map.yaml`）：**

把抽象 capability 映射到實際 tool / MCP family，session 啟動載一次 frozen。Skill manifest 只宣告需要什麼 capability，不綁死哪個 implementation。例：

```yaml
capabilities:
  file.read:    { local_tools: [read_file] }
  rag.search:   { local_tools: [rag_search, rag_explore, rag_get_context] }
  web.search:   { mcp_families: [web_search] }
  shell.execute: { local_tools: [bash] }
```

**禁止繞道**：`agent/skills/runtime.py:SkillRuntime.read_skill_resource()` 強制把 `rel_path` join 到 `skill_root` 然後 `is_relative_to(root)` traversal guard；`read_file` 在 active skill 下對 `references/`、`assets/`、`scripts/` 開頭的 relative path 也只解析到 skill bundle，找不到就報錯，不 fallback 到 cwd。讀一般 cwd 草稿仍使用普通相對路徑或 absolute path；absolute path 會先經過敏感檔名 / path segment denylist。

**Skills 不會自動進 prompt**：startup 不再 inject metadata block、不再印 banner。模型對「有哪些 skill 可用」是色盲的；被使用者問起時透過 `bash ls skills/` 現查（需要使用者批准）。`/skill` 是唯一啟用入口。

### Session lifecycle（async）

MCP tools 是純 async `StructuredTool`，所以 `ChatSession` 整條路徑也是 async：

- `ChatSession.create(config, load_mcp=True)` — async factory，啟動 MCP subprocess（透過 `load_mcp_tools_with_families()` 同時拿到 tool list 與 `tool_name → family` map）、build graph
- `await session.turn(user_input)` — 用 `graph.astream(stream_mode="updates")` 跑一輪，邊跑邊把「哪個 node 產出什麼訊息」餵給 `progress_cb`
- `await session.flush_recent_turns()` — CLI 結束前呼叫

chat CLI 的 `progress_cb` 把每一次 tool call 印成 `→ calling <tool>` / `✓ <tool> returned`。

### MCP subprocess 的兩個 quirk

mrkrsl/web-search-mcp 有兩個壞習慣，在 `agent/mcp.py` 處理掉：

1. **stderr 灌 debug log** — 每個 MCP server 都在 `/bin/sh` 裡啟動，stderr 導到 `~/.cache/agent-mcp/<server>.stderr.log`
2. **stdout 摻非 JSON 行**（`"Shutting down gracefully..."` 之類）會讓 stdio client 炸 `BrokenResourceError`。同一層 shell pipeline 掛了 `grep '^{'`，只讓 JSON-RPC 訊息通過

---

## Slash commands

| 命令 | 用途 |
|---|---|
| `/help` | 列出所有 slash commands |
| `/status` | 顯示 session_id、turn count、recursion limit、last tool counts、`plan_mode`、`plan_log_path`、`active_skill`、`task_mode` |
| `/mode [name]` | 切換 session mode（`normal` / `plan`；無參數 = 互動選單） |
| `/skill [name [mode]]` | 啟用 / 切換 / 停用 local skill（無參數 = 互動選單；`/skill none` = deactivate） |
| `/init` | Ingest 父 repo（自動排除本 app 專案目錄） |
| `/ingest <path>` | Upsert 單檔或整個目錄到 rag store |
| `/sync [path]` | 顯示「磁碟有但 store 沒有」/「store 有但磁碟沒有」（dry run） |
| `/prune [path] [--yes]` | 移除 store 裡 source 已不存在的 entry（`--yes` 才實際執行） |
| `/clear` | 清螢幕 |
| `/quit` (alias `/exit`) | 離開 |

`/ingest` `/sync` `/prune` 在 rag 端拋 `ValueError`（例如 plan_logs 路徑被 SKIP_DIRS / sentinel 擋）時，會被翻譯成 `SlashCommandError` 顯示給使用者，**不炸掉 chat loop**。

CLI loop 對空白 input（純空白 / 純不可見字元）會直接重顯 `>>`，**不再當成 exit**；唯一退出方式是 `q` / `quit` / `exit` / `/quit` / EOF / Ctrl-C。

---

## 執行長相（chat CLI）

```
$ python -m agent.cli.chat
Agent Chat (LangGraph mode). Type 'q' to quit.
Mode: default

>> /skill
Current skill: none
Available skills:
  [0] none  - deactivate active skill
  [1] academic-paper-writing  - <description>
Select (number or name; Enter to cancel): 1

Task mode for academic-paper-writing:
Available modes:
  [0] none  - no task mode
  [1] revision
  [2] literature-review
  [3] drafting
  [4] submission-support
Select (number or name; Enter for none): 1

skill -> academic-paper-writing (revision)

>> 幫我改寫這段摘要：...
  → calling read_file       # 讀 references/section-playbooks.md（解析到 skill bundle）
  ✓ read_file returned

<agent 改寫結果，全程繁體中文>
```

啟動參數：
- `--no-mcp` — 不載入任何 MCP server，只綁本地 agent tool（rag 三個 + recall_history + read_file + bash）
- `--max-turns N` — 一輪對話內最多幾次 tool round-trip（預設 32）

---

## 配置（`agent/config.py:AgentConfig`）

繼承 `rag.RAGConfig`，新增：

| 欄位 | 預設 | 說明 |
|---|---|---|
| `llm_model` | `"z-ai/glm-5"` | Chat 主模型 |
| `gen_llm_model` | `"google/gemini-3.1-pro-preview"` | Eval generator |
| `judge_llm_model` | `"openai/gpt-5.2"` | Eval judge |
| `filter_llm_model` | `"llama3.1:8b"` | Eval filter |
| `agent_max_messages` | 20 | 同輪 prompt message 上限 |
| `agent_max_tool_interactions` | 4 | 同輪 tool round-trip 上限（與 `--max-turns` 不同層） |
| `agent_recent_turns_window` | 10 | prompt-visible 對話 window |
| `plan_logs_dir` | `"plan_logs"` | Plan 模式 md 落地目錄（相對 app root） |
| `plan_log_max_tool_chars` | 65536 | 單個 ToolMessage 寫入 md 的軟 cap（不影響 LLM context） |
| `skills_dir` | `None` | Skill 目錄；None → `<repo>/skills` |
| `skill_validation_enabled` | `True` | 是否跑 skill_validator node |
| `skill_max_validation_retries` | 1 | validator 違規時最多回 agent 改寫幾次 |
| `skill_capability_map_path` | `None` | capability map yaml 路徑；None → `agent/skills/capability_map.yaml` |
| `skill_max_pinned_reference_chars` | 65536 | 單一 pinned reference 可進 skill context 的最大字元數 |
| `skill_max_total_skill_context_chars` | 200000 | SKILL.md + pinned references 組成的 skill context 最大字元數 |
