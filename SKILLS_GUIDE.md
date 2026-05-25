# Skills 規範與建立指南

> 這份文件是本專案 skills 的**權威格式參考**。在新增、修改、或重構 skill 時請以此為準。
> 適用對象：人類開發者、Claude Code、Codex、其他 AI 編碼助手。

---

## 為什麼需要這份文件

Skills 有一份開放標準（Agent Skills），但各家 runtime（特別是 Claude Code）在標準之上加了許多自己的擴充欄位和語法。網路上的教學經常把兩者混在一起，導致：

- AI 助手依照 Claude Code 文件寫 skill，但本專案 agent 並非 Claude Code，那些擴充功能不會生效
- 使用了非標準欄位（如 `arguments: [...]`、`context: fork`），讓 skill 失去可攜性
- 在 SKILL.md 內文使用 `` !`command` ``、`$ARGUMENTS` 等 Claude Code 專屬語法，在其他 runtime 變成字面字串

**本專案以 Agent Skills 標準為準。** 任何標準之外的欄位或語法，除非在本文件中明確列為「本專案實作的擴充」，否則一律不使用。

---

## 一、什麼是 Skill

Skill 是一個資料夾，至少包含一個 `SKILL.md` 檔案。`SKILL.md` 包含：

1. **YAML frontmatter** — 給 `/skill` picker 顯示與辨識的 metadata
2. **Markdown 內文** — 使用者明確啟用後，agent 該照著做的指令
3. **manifest.yaml（選用，本專案擴充）** — 宣告 task modes、resource、capability、tool policy

Skill 採用**漸進式揭露（progressive disclosure）**：

- **啟動時**：agent 不把 skill 清單、frontmatter 或 description 自動塞進 system prompt
- **啟用時**：只有使用者透過 `/skill` 或 `/skill <name> [mode]` 明確選擇，runtime 才載入完整 `SKILL.md`
- **延伸時**：`SKILL.md` 可引用同目錄的其他檔案；active skill 下，`references/`、`assets/`、`scripts/` 開頭的路徑會被限制在 skill bundle 內

這個機制讓我們可以維持可預期的手動啟用路徑，避免 agent 自行掃描、判斷或自動啟用 skills。

### Internal helper skill：`_prompt-master`

`skills/_prompt-master/` 是 `/thinking extended` controller 使用的內部 helper。它一次性 vendor 自 `nidhinjs/prompt-master`，controller 只直接讀取 `SKILL.md` 作為 prompt rewrite 的 system context，不透過 skill loader 自動啟用，也不會改變使用者當前 active skill。

如果使用者手動執行 `/skill _prompt-master`，它仍會走一般 skill runtime，並套用自己的 `manifest.yaml` tool policy。這個資料夾名稱前面的 `_` 是內部 helper 例外；一般新增給使用者選用的 skill 仍應使用 kebab-case。

---

## 二、標準格式

### 目錄結構

```
skills/
└── <skill-name>/
    ├── SKILL.md           # 必要
    ├── manifest.yaml      # 選用，本專案 runtime metadata
    ├── references/*.md    # 選用，補充文件
    ├── scripts/           # 選用，可執行腳本（需 bash 工具支援）
    └── assets/            # 選用，模板或資源檔
```

**命名規則：**

- 資料夾名稱使用 **kebab-case**（小寫字母、數字、連字號）
- 長度上限 64 字元
- 必須叫 `SKILL.md`（大小寫敏感）。`skill.md`、`Skill.md`、`README.md` 都不會被識別
- 只有 internal helper 可以使用 `_` 前綴，例如 `skills/_prompt-master/`

### SKILL.md 結構

```markdown
---
name: skill-name
description: Use when the user wants to ... [具體適用情境]
---

# Skill 標題

## 任務說明
[祈使句寫的指令]

## 步驟
1. ...
2. ...
```

### Frontmatter 欄位（標準）

只有兩個欄位你需要關心：

| 欄位 | 必要性 | 說明 |
|------|--------|------|
| `name` | 建議 | Skill 識別碼。省略時會用資料夾名稱推導。kebab-case。 |
| `description` | **強烈建議** | `/skill` picker 顯示給使用者看的辨識文字。詳見下方寫作指引。 |

**標準也定義但本專案通常不用：**

| 欄位 | 說明 |
|------|------|
| `license` | Skill 的授權條款 |
| `compatibility` | 宣告此 skill 相容於哪些 runtime |
| `metadata` | 自訂 metadata（鍵值對） |
| `allowed-tools` | 標準中標記為 experimental，行為各 runtime 不一，本專案不依賴 |

**就這樣。其他你在網路上看到的欄位都不是標準的一部分**（詳見第五節）。

---

### manifest.yaml 欄位（本專案擴充）

`manifest.yaml` 是本專案 runtime 使用的嚴格 schema。未知 top-level key、型別錯誤、空的 `capabilities: {}` 都會在 skill 啟用時 raise `ValueError`，讓問題早點暴露。

| 欄位 | 型別 | 說明 |
|------|------|------|
| `capabilities.required` | string list | 必要 capability。必須存在於 `agent/skills/capability_map.yaml`，且能解析到目前可用 tool；否則啟用失敗。 |
| `capabilities.optional` | string list 或 `{id, use_when}` list | 選用 capability。不存在或目前不可用時不阻止啟用，但仍視為 active policy，不會退回全工具開放。 |
| `resources` | list | 每項需有 `path: string`，可選 `use_when: string`、`pinned: bool`。`pinned: "yes"` 這類字串不是 bool，會被拒絕。 |
| `task_modes` | string list | `/skill <name> <mode>` 可選模式。非法 mode 會回 slash command error，不會炸掉 CLI loop。 |
| `tool_policy.disallow` | string list | 明確禁止 tool。deny rules 優先於 capability grants。 |

範例：

```yaml
capabilities:
  required:
    - file.read
    - rag.search
    - history.search
  optional:
    - id: web.search
      use_when: target journal guidelines or current venue information is needed

resources:
  - path: references/checklist.md
    use_when: checklist-heavy tasks
    pinned: false

task_modes:
  - revision
  - drafting

tool_policy:
  disallow:
    - bash
```

Pinned resources 會在啟用 skill 時直接放進每回合 context，受 `skill_max_pinned_reference_chars` 與 `skill_max_total_skill_context_chars` 限制。只 pin 每次都必要、且很小的檔案；其他 reference 讓 agent 在 active skill 下按需讀取。

Capability 命名要精確：

- `rag.search` 只授權 indexed KB 工具（知識庫文件、研究筆記、已 ingest 的資料）。
- `history.search` 只授權 persisted chat history 工具（舊對話、較早 session、被 recent window eviction 的 turn）。
- 如果 skill 可能需要使用者過去對話脈絡，例如「你自行看我一月上半做了什麼」這類請求，manifest 必須宣告 `history.search`。只宣告 `rag.search` 不會讓 active skill 下的 writer 看到 history retrieval schema。
- Plan mode logs 不進 Chroma `chat_history`，所以即使宣告 `history.search`，也不能承諾能搜尋 plan-mode-only 的紀錄；需要時應請 agent 讀 `plan_logs/` 檔案或請使用者指出位置。

## 三、Description 寫作指引

`description` 不會讓 agent 自動啟用 skill。本專案只允許使用者透過 `/skill` 明確啟用；description 的作用是讓 picker 中的選項容易辨認，也讓人類維護者快速理解用途。

### 公式

```
What it does + When to use it + （選用）Specific signals / Negative cases
```

### 不好的寫法

```yaml
description: Translates text to formal Chinese.
```

問題：只說功能，使用者在 picker 裡不容易判斷該不該選它。

### 好的寫法

```yaml
description: Use when the user wants to translate text into formal written
  Traditional Chinese suitable for business letters, official emails, or
  professional documents. Do NOT use for casual translation or spoken Chinese.
```

差別：明確列出適用情境（商務書信、正式 email、專業文件）和不適用情境（口語、休閒翻譯）。使用者選 skill 時比較不容易選錯。

### Description 寫作清單

- [ ] 寫出**做什麼**（What）
- [ ] 寫出**何時用**（When）— 列出具體的使用者請求型態
- [ ] 必要時寫出**何時不用**（When NOT）— 用 "Do NOT use for..." 或 "Not for..."
- [ ] 用英文寫（方便維護與跨 runtime 閱讀，內文可用中文）
- [ ] 不超過 3-4 句

### 選項辨識度

如果發現使用者常選錯或不知道該選哪個 skill，可以把 description 寫得更具體一點：

```yaml
description: Use when the user wants to translate ... Especially relevant for
  formal letters, official documents, business communication, or 公文-style
  translation, even if the user does not explicitly say "formal".
```

---

## 四、SKILL.md 內文寫作指引

### 基本原則

- 用**祈使句**：「Read the file」「Use this template」，不要「The skill will read...」
- 控制在 500 行內。超過就拆成 reference 檔案
- 解釋**為什麼**這麼做，不要堆疊 MUST、ALWAYS、NEVER
- Skill 是寫給 agent 看的，不是寫給人看的文件——別寫「本 skill 旨在...」這種廢話

### 結構建議

```markdown
---
name: ...
description: ...
---

# Skill 名稱

## When to use
（補充 frontmatter 的 description，講細節）

## Process / Steps
1. 第一步
2. 第二步
   - 子步驟
3. 第三步

## Output format
（明確規定輸出格式，可以給範本）

## Examples
**Input:** ...
**Output:** ...

## Edge cases
- 情況 A：怎麼處理
- 情況 B：怎麼處理
```

### Extended Thinking 與 Skills

`/thinking extended` 不會自動啟用任何使用者 skill。它保留目前 active skill 的 context 與 tool policy，另外用 `_prompt-master` helper 把使用者輸入重寫成較清楚的 agent prompt。

Extended mode 的 rewriter、writer、reviewer 都會收到同一份 runtime tool availability block。`SYSTEM_PROMPT` 中的「always available」是 base session 描述；active skill 啟用後，實際工具集合一律以 `tool_policy_active`、`allowed_tools`、`denied_tools` 為準，不要在 skill 內文或測試裡假設 base 工具一定可用。

啟用 `/thinking extended` 前，必須直接在 `agent/config.py` 的 `AgentConfig` 填入三個角色 model 欄位：

```python
thinking_reviewer_model: str = "anthropic/claude-haiku-4.5"
thinking_reviewer_max_tokens: int = 1024
thinking_rewrite_model: str = "openai/gpt-5-mini"
thinking_repair_model: str = "openai/gpt-5-mini"
```

這些欄位直接由 `AgentConfig` 決定；任一被設為空字串時，`/thinking extended` 會拒絕切換，避免 Extended mode 靜默退回 `llm_model` 造成同 model 自審。第一版不從 `.env` 或 CLI 參數讀取這三個欄位；`.env` 只保留 `OPENROUTER_API_KEY` 這類 secret。

### 漸進式揭露（檔案拆分）

當 SKILL.md 接近 500 行，開始拆檔。在 SKILL.md 裡明確指引何時讀子檔案：

```markdown
## Routing

- 處理表單填寫 → 讀 `forms.md`
- 抽取表格 → 讀 `tables.md`
- 一般文字抽取 → 繼續看下面
```

子檔案路徑是相對於 SKILL.md 所在目錄。

### 多領域組織

當一個 skill 涵蓋多個變體（例如多雲端）時，按變體組織：

```
cloud-deploy/
├── SKILL.md           # 共通流程 + 路由
└── references/
    ├── aws.md
    ├── gcp.md
    └── azure.md
```

SKILL.md 裡寫清楚「使用者提到 AWS → 讀 references/aws.md」。

---

## 五、⚠️ 不在標準裡的東西

下列項目經常出現在 Claude Code 文件或網路教學中，但**不是 Agent Skills 標準的一部分**。本專案不使用，看到請改寫。

### Claude Code 專屬 Frontmatter 欄位（不要用）

```yaml
disable-model-invocation: true   # ❌ Claude Code 擴充
user-invocable: false            # ❌ Claude Code 擴充
context: fork                    # ❌ Claude Code 擴充（子 agent 隔離）
agent: Explore                   # ❌ Claude Code 擴充
effort: high                     # ❌ Claude Code 擴充
paths: "src/**,*.md"             # ❌ Claude Code 擴充
argument-hint: [issue-number]    # ❌ Claude Code 擴充
model: claude-sonnet-4-...       # ❌ Claude Code 擴充
hooks: ...                       # ❌ Claude Code 擴充
mode: true                       # ❌ Claude Code 擴充
```

這些欄位寫了不會出錯，但本專案 agent 不會解讀，等於沒效果，反而誤導後續維護者以為 skill 有那些行為。

### Claude Code 專屬內文語法（不要用）

```markdown
!`git diff HEAD`              # ❌ Claude Code 的 bash 預執行注入
$ARGUMENTS                     # ❌ Claude Code 的參數替換
$0  $1  $2                     # ❌ Claude Code 的位置參數
${CLAUDE_SKILL_DIR}            # ❌ Claude Code 的環境變數
${CLAUDE_SESSION_ID}           # ❌ Claude Code 的環境變數
```

在本專案 agent 眼中，這些都是普通字串，會原封不動傳給 model，不會有任何替換或執行行為。

### 不存在的欄位（網路上的訛傳）

```yaml
arguments: [arg1, arg2]        # ❌ 這個欄位根本不存在；正確的是 argument-hint（仍是 Claude Code 擴充）
$name                          # ❌ 沒有具名參數這種東西
```

### 簡單判別法

如果某個欄位或語法**不在本文件列出的標準 frontmatter、SKILL.md 內文寫法，或本專案 `manifest.yaml` 擴充**中，就不要用。

---

## 六、完整範例

### 範例 1：純文字指令型 skill

`skills/formal-chinese-translation/SKILL.md`

```markdown
---
name: formal-chinese-translation
description: Use when the user wants to translate text into formal written
  Traditional Chinese suitable for business letters, official emails, or
  professional documents. Do NOT use for casual translation or spoken Chinese.
---

# Formal Chinese Translation

## Process

When translating into formal Traditional Chinese:

1. Use 您 instead of 你 when addressing the reader
2. Replace colloquial vocabulary with formal equivalents:
   - 給 → 致 / 予
   - 因為 → 由於 / 緣於
   - 但是 → 然而 / 惟
   - 現在 → 現今 / 目前
3. Use complete sentence structures; avoid 啊、啦、欸、耶
4. End requests with formal closings: 敬請查照、煩請惠覆、謹此致謝
5. Preserve original meaning precisely — do not embellish

## Output format

Provide the translation directly without explanation, unless the user
specifically asks for notes on word choices.

## Examples

**Input:** 跟你說一下，那個案子我們可能要延後
**Output:** 茲告知，該案恐須延後辦理。
```

### 範例 2：多檔案 skill

```
skills/code-review/
├── SKILL.md
├── security.md
├── performance.md
└── style.md
```

`skills/code-review/SKILL.md`

```markdown
---
name: code-review
description: Use when the user asks for code review, requests feedback on
  a pull request, asks about code quality, or wants to identify issues in
  existing code.
---

# Code Review

## Process

1. Read the code the user provided
2. Determine which review dimensions apply (often multiple)
3. For each dimension, read the corresponding reference and apply its checklist
4. Aggregate findings into the output format below

## Routing

- Security concerns (auth, input validation, secrets) → read `security.md`
- Performance concerns (algorithms, queries, memory) → read `performance.md`
- Code style / readability → read `style.md`

If unsure, default to applying all three.

## Output format

Group findings by severity:

### 🔴 Critical
- [檔案:行號] 問題描述 + 建議修法

### 🟡 Should fix
- ...

### 🟢 Nice to have
- ...
```

---

## 七、新建 Skill 的工作流程

當 AI 助手或開發者要新增一個 skill，依序做：

1. **確認流程已成熟**
   - 你能用口頭跟新進同事講清楚這件事怎麼做嗎？不能 → 還沒到寫成 skill 的時機
   - 流程是否會穩定重複出現？只用一次 → 不需要 skill

2. **建立目錄**
   ```
   skills/<skill-name>/SKILL.md
   ```

3. **撰寫 frontmatter**
   - `name`：與資料夾同名
   - `description`：套用第三節的公式

4. **視需要撰寫 manifest.yaml**
   - 需要限制 tool 或宣告 capability 時，使用 `capabilities` / `tool_policy.disallow`
   - 需要使用者過去對話脈絡時，加 `history.search`；需要 indexed KB 時才加 `rag.search`
   - 需要 task mode 時，使用 `task_modes`
   - 需要 reference routing 時，使用 `resources`
   - 不要寫空的 `capabilities: {}`；沒有 policy 就省略 `capabilities` / `tool_policy`

5. **撰寫內文**
   - 祈使句、結構化、舉例
   - 控制在 500 行以內

6. **本地驗證**
   - 啟動 agent，用 `/skill <name>` 或 `/skill <name> <mode>` 明確啟用
   - 確認啟用時沒有 manifest validation / capability resolution 錯誤
   - 確認 agent 真的有讀 `SKILL.md` 並照做

7. **不需要的東西不要加**
   - 不要為了「看起來專業」加一堆 Claude Code 專屬欄位
   - 不要在內文塞 `` !`command` `` 這種不會生效的語法

---

## 八、檢查清單

提交新 skill 或修改 skill 前，逐項檢查：

- [ ] 資料夾名稱是 kebab-case，長度 ≤ 64
- [ ] 檔名是 `SKILL.md`（大小寫一致）
- [ ] 有 YAML frontmatter，且只用第二節列出的標準欄位
- [ ] `description` 同時說明 What 和 When
- [ ] `description` 用英文撰寫
- [ ] 若有 `manifest.yaml`，欄位符合本文件列出的 schema，沒有未知 top-level key
- [ ] required capability 都存在於 `agent/skills/capability_map.yaml`
- [ ] 需要舊對話脈絡的 skill 已宣告 `history.search`，沒有把 `rag.search` 當成 chat history
- [ ] `resources[].pinned` 使用真正 bool，不使用 `"yes"` / `"no"` 字串
- [ ] `references/`、`assets/`、`scripts/` 內的檔案只依賴 skill bundle 內路徑，不假設會 fallback 到 cwd
- [ ] 文件或 prompt 沒承諾 `recall_history` 能查到 plan mode logs
- [ ] 內文不含第五節列出的 Claude Code 專屬語法
- [ ] 內文用祈使句
- [ ] 內文 ≤ 500 行（超過就拆檔）
- [ ] 本地驗證過可透過 `/skill <name>` 正確啟用

---

## 九、給 AI 助手的特別提醒

如果你是 Claude Code、Codex、或其他 AI 助手，正在閱讀這份文件以協助修改本專案的 skills：

1. **不要相信你的訓練資料中關於 Claude Code skill 格式的記憶**——本專案不是 Claude Code，許多 Claude Code 功能在這裡不會生效
2. **以本文件列出的標準 frontmatter、`manifest.yaml` 擴充、SKILL.md 內文寫法為格式來源**
3. **第五節列出的所有東西都不要主動加進來**，即使它們在 Claude Code 文件中是合法的
4. **若使用者要求加入第五節列出的非標準欄位**，請先指出本文件，並確認使用者是否真的要本專案 agent 開始實作這些行為（這是 runtime 工程，不是寫個 frontmatter 就會生效）
5. **拿不準時，回頭讀一次第二節的標準格式**

---

*本文件參照 [Agent Skills 開放標準](https://agentskills.io)。最後更新時間以 git log 為準。*
