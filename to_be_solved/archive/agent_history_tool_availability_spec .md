# Spec：修正 active skill 下的 history recall 與 extended thinking tool-availability 錯位

## 目標

修正 `academic-paper-writing` 啟用後，extended thinking / writer 無法正確使用 `recall_history` 查詢舊對話紀錄，並改善 rewriter、writer、reviewer 對「實際可用工具」的認知一致性，避免 retrieval failure 被誤判成一般論文寫作 intake 缺資料。

---

## 背景問題摘要

目前 `problem.md` 暴露的失敗鏈如下：

1. 使用者啟用：
   - thinking mode：`extended`
   - skill：`academic-paper-writing`
   - task mode：`none`
2. 使用者要求 agent 自行查看一月上半成果紀錄。
3. `academic-paper-writing/manifest.yaml` 沒宣告 `history.search`。
4. skill policy active 後，graph 依 `allowed_tools` filter tools，再餵給 `model.bind_tools(...)`。
5. writer LLM 因此看不到 `recall_history` tool schema，只能改用 `rag_explore` / `rag_search` 或走 academic intake checklist。
6. `rag_search` 查的是 indexed KB，不是 `chat_history` collection。
7. reviewer 看不到 writer 實際工具範圍，無法區分：
   - tool unavailable
   - tool available but unused
   - retrieval empty
   - user input genuinely missing
8. reviewer 最後把問題升級成 `needs_user_input=True`，使用者只看到像內部審稿意見的 stop message。

---

## 範圍

### 動到的檔案 / 模組

#### 必改

- `skills/academic-paper-writing/manifest.yaml`
  - 加入 `history.search` capability。
- `agent/session.py`
  - 在 active skill 啟用時，向 writer prompt 注入 skill-aware tool availability hint。
  - 在 extended thinking 呼叫 rewrite / review 時傳入實際 tool availability。
- `agent/thinking.py`
  - `rewrite_messages(...)` 不再 hardcode 完整工具清單。
  - `review_messages(...)` / `review_draft(...)` 接收 tool availability context。
  - reviewer prompt 增加 retrieval failure 判讀規則。
- `info.md`
  - 更新 skill policy 與「always available」工具在 active skill 下的關係。

#### 可能需要改

- `tests/`
  - 優先延伸既有 skill / graph / thinking tests。
  - 若沒有合適檔案，再新增對應測試檔。
- `skills/academic-paper-writing/SKILL.md`
  - 只有在需要補充「當使用者要求查先前紀錄時，優先查 history」的語意規則時才改。
  - 不要把這份 SKILL.md 改成過度工具導向的 prompt。

#### 明確不要動

- 不要修改 `agent/history_rag/store.py` 的 collection 名稱、persist dir、寫入格式。
- 不要修改 `recall_history` 的基本 tool schema，除非測試證明現有 schema 無法支援驗收條件。
- 不要修改 `rag_search` / `rag_explore` 的語意，這次不把 RAG KB 和 chat history 混在一起。
- 不要修改 plan mode 的儲存策略：plan logs 仍不進 chroma。
- 不要新增外部依賴。
- 不要把 `bash` 加回 `academic-paper-writing` 的 allowed tools。
- 不要重構整個 graph 架構；這次是局部修正，不做大改。

---

## 需求

### P0 — 讓 academic-paper-writing 可以使用 `recall_history`

- `skills/academic-paper-writing/manifest.yaml` 的 `capabilities.required` 必須包含 `history.search`。
- 啟用 `academic-paper-writing` 後，resolved `allowed_tools` 必須包含：
  - `read_file`
  - `rag_explore`
  - `rag_search`
  - `rag_get_context`
  - `recall_history`
- 啟用 `academic-paper-writing` 後，resolved `denied_tools` 必須仍包含或有效排除：
  - `bash`
- required capability 無法解析時，skill activation 必須 fail fast，而不是 silent degradation。

#### 概念示例

```yaml
capabilities:
  required:
    - file.read
    - rag.search
    - history.search
```

---

### P0.5 — 驗證資料前提與後續處置

這不是功能修正，而是診斷步驟。目的是在 ship P0 之前先知道：對使用者實際的 `chat_history` collection 來說，P0 是真的能 unblock `problem.md` 這個 case，還是只是修了一個必要但不充分的條件。

#### 診斷步驟

本機跑 `recall_history`，不要只用一種 query 風格。需要至少跑四種：

1. **語意敘述**：`"我一月上半做的研究成果"` / `"early January research progress"`
2. **時間錨點 + topic**：`"一月 人工智慧"` / `"AI January experiments"`
3. **Role filter**：對 step 1 / 2 各加 `role="assistant"` 跑一次，看看 assistant 之前是否總結過該段
4. **Bulk inspection**：`k=20` 不加 query 或用空泛 query，看 collection 裡到底有沒有東西、最舊那筆 timestamp 是何時

只有上述四種都返回空，才能下「empty」結論。單一語意 query 返回空有可能是 false negative（bge-m3 embedding 跟使用者當時用字差太遠）。

#### 後續處置（不論結果如何，P0 都要 ship）

manifest 本來就漏寫 `history.search`，這是 spec bug，不是「等資料確認再修」的東西。因此：

- **Hit case**：照 spec 走 P1 / P2 / P3。Integration test 用 fake history store 注入 seed turn，驗證 retrieval path 真的被用到。
- **Empty case**：P0 仍 ship；但 P3 escape hatch 的對話內容必須涵蓋這個情境——明確告訴使用者「我這邊 `chat_history` collection 對你這個查詢是空的，可能原因有 (a) 那段對話還在 recent_turns window 內就沒被持久化、(b) 你當時在 `/mode plan` 下做的、(c) 對話發生在這個 app 的持久化機制接上之前」。**不要讓 agent 假裝查過、也不要讓它把 empty 包成「使用者沒提供資料」**。
- **不確定 case**（query 之間結果不一致）：spec 不在這層處理，留給 reviewer 的 retrieval failure 規則自然消化。

---

### P1 — Rewriter / Writer 都要知道實際工具可用性

目前 `thinking.py` 的 rewriter wrapper hardcode 一份完整工具清單，這會與 active skill filter 後的實際工具不一致。

修正後：

- `rewrite_messages(...)` 必須接收一段 tool availability context。
- wrapper 裡的「可用工具」必須由實際 allowed / denied / tool_policy_active 狀態產生，不可 hardcode 全量工具清單。
- 若 active skill 啟用後某工具被排除，rewriter 必須知道該工具不可用。
- writer prompt 也要收到 skill-aware tool availability hint，避免 base `SYSTEM_PROMPT` 的「always available」與 active skill 實際工具集合矛盾。
- tool availability hint 必須是 ephemeral system context，不要持久化進 chat history。

#### 建議介面形狀

```text
[Active skill tool availability]
tool_policy_active: true
available_tools: read_file, rag_explore, rag_search, rag_get_context, recall_history
excluded_tools: bash
note: Active skill policy overrides the base "always available" wording.
```

#### `tool_availability` 預設值與 fallback 行為

`rewrite_messages` 和 `review_messages` 的 `tool_availability: str = ""` 預設值必須有明定語意：

- 預設空字串 = **沒有 active skill policy**，等同 `tool_policy_active=False`。
- 此時 wrapper **不退回現在的 hardcoded 完整工具清單**，而是改由共用 helper（譬如 `agent/thinking.py` 內或 `agent/skills/runtime.py` 內新增的 `render_tool_availability_block(...)`）從 base 工具集合產生一段等效但動態的描述。
- 共用 helper 是 single source of truth：active skill 啟用時從 `SkillRuntime.allowed_tools` / `denied_tools` 產生；無 active skill 時從 graph 的 base tool list 產生。
- **絕對不可以在 `thinking.py` 裡留任何寫死的工具名稱清單。** 這是這次修正的核心目的。

**為什麼不退回 hardcoded list**：退回 hardcoded list 等於把這個 bug 的觸發條件從「啟用 academic-paper-writing」延後到「未來新增任何縮減工具的 skill」。spec 的 P4 已經明說要避免後續新 skill 踩同坑，所以 fallback 也必須走動態路徑，不能留後門。

---

### P2 — Reviewer 要能區分 retrieval failure 類型

不要求第一版修改 `ReviewFinding` schema。先用 reviewer prompt 規則完成。

#### Finding routing contract（最高優先規則）

在寫任何 finding 之前，reviewer 必須先決定該 finding 走哪條路由。`route_review_report` 的當前語意是：**只要任何一條 finding 的 `needs_user_input=True`，整個 report 立刻走 `ask_user`，連 reviser loop 都不會跑。** 因此 finding 的形狀直接決定下游行為。

Reviewer 產出 finding 時，必須照下列對應表選擇形狀：

| 情境 | severity | needs_user_input | decision | revision_instruction 寫給誰 |
|---|---|---|---|---|
| writer 漏查可用工具（evidence trace 沒有預期的 retrieval call，且該 tool 在 `available_tools` 內） | `major` | `False` | `revise` | reviser / writer：明確指示要 call 哪個 tool、用什麼 query |
| writer 已嘗試 retrieval 但結果為空 | `minor` 或 `note` | `False` | `revise` 或 `pass` | reviser：要求 draft 誠實說明「已查詢但未找到」，不要編造 |
| `recall_history` 因 active skill policy 被排除（在 `excluded_tools` 內） | `blocker` | `True` | `block` | 使用者：說明是 skill policy / 工具設定問題，建議切 skill 或聯絡管理者 |
| 使用者真的沒提供必要資料，且現有工具都救不了 | `blocker` 或 `major` | `True` | `block` 或 `revise` | 使用者：可讀的具體問題（不是內部 reviser 指令） |
| draft 引入未由 evidence 支撐的研究成果、數據、方法、citation | `blocker` | `True` 或 `False`（看是否能由再次 retrieval 救回） | `block` 或 `revise` | 視情況：reviser 修掉，或使用者澄清來源 |

**關鍵原則**：`needs_user_input=True` 是逃生開關，只在「靠 reviser 多跑一輪也救不回來」時用。能靠 reviser 補一次 tool call 救回的，一律走 `needs_user_input=False` + `decision=revise` + `severity=major`，讓 reviser loop 處理。

#### Reviewer 必須遵守的具體規則

- 若使用者要求查先前紀錄，而 `recall_history` 可用但 evidence trace 沒有 `recall_history` call：
  - 不可直接 `needs_user_input=True`
  - 應要求 reviser / writer 先使用 `recall_history`
- 若 `recall_history` 已使用但結果為空：
  - 不可要求使用者重述所有研究內容
  - 應允許 draft 明確說明「已查詢但未找到足夠紀錄」
  - 可詢問更小的下一步問題，例如「這段是否在 plan mode logs」
- 若 `recall_history` 因 active skill policy 不可用：
  - stop message 必須指出這是工具設定 / skill policy 問題
  - 不可包裝成一般 academic intake checklist
- 若 draft 引入未由 evidence 支撐的研究成果、數據、方法：
  - 仍必須阻擋或要求修正
- `needs_user_input=True` 的 `revision_instruction` 必須是使用者可讀的問題，不可像內部 reviser 指令。

---

### P3 — Meta-conversation escape hatch

這一階段不一定與 P0–P2 同 PR 完成，但 spec 先定義方向。

當使用者明確從 domain task 切到 agent behavior / memory troubleshooting，例如：

- 「你為什麼一直問我」
- 「你應該能看見我的紀錄」
- 「忘了前面的任務」
- 「是不是工具沒接上」
- 「你剛剛沒有照我意思查」

agent 不應繼續套用 academic-paper-writing intake checklist。

最低需求：

- 回答應轉成 troubleshooting frame。
- 應說明：
  - 目前 active skill 是什麼
  - 可用 / 不可用工具有哪些
  - 是否查過 `recall_history`
  - 查詢結果是否為空
  - 若資料可能在 plan logs，明確指出 plan logs 不會被 `recall_history` 搜到
- 不要求自動 deactivate skill。
- 不要求第一版做複雜 classifier；可先用明確 phrase-based heuristic 或 reviewer prompt 規則處理。

---

### P4 — 文件更新

更新文件要讓後續新增 skill 的人不再踩同一個坑。

至少補充：

- `rag.search` 與 `history.search` 是不同 capability。
- 若 skill 可能需要使用者過去對話脈絡，manifest 應宣告 `history.search`。
- base `SYSTEM_PROMPT` 裡的「always available」在 active skill policy 下是條件式可用。
- active skill 啟用後，實際工具集合以 `allowed_tools` / `denied_tools` / `tool_policy_active` 為準。
- plan mode logs 不進 chroma，因此 `recall_history` 查不到 plan-mode-only 的紀錄。

---

## 介面 / 資料結構

### Tool availability context

可用 string block，不要求新增 dataclass。重點是 rewriter、writer、reviewer 看到同一份事實。

建議內容：

```text
active_skill: academic-paper-writing
task_mode: none
tool_policy_active: true
available_tools: [...]
denied_tools: [...]
unavailable_base_tools: [...]
```

### Function signature 調整

只描述介面，不指定實作。

```python
def rewrite_messages(
    *,
    skill_text: str,
    user_input: str,
    visible_context: str,
    skill_context: str,
    tool_availability: str = "",
) -> list:
    ...
```

```python
def review_messages(
    *,
    raw_user_input: str,
    rewritten_prompt: str,
    draft: str,
    skill_context: str,
    evidence_trace_summary: str,
    previous_rebuttal: str,
    tool_availability: str = "",
) -> list:
    ...
```

`rewrite_prompt(...)` 與 `review_draft(...)` 也要接受對應參數並往下傳。

---

## 驗收條件

### P0 驗收

- [ ] 啟用 `academic-paper-writing` 不會 fail。
- [ ] 啟用後 `allowed_tools` 包含 `recall_history`。
- [ ] 啟用後 `bash` 不在可用工具集合中。
- [ ] graph binding 後 writer LLM 可看到 `recall_history` schema。
- [ ] `problem.md` 類型 prompt 不應在未嘗試 history retrieval 前直接要求使用者補完整研究背景。

### P0.5 驗收

- [ ] 至少跑過上述四種查詢風格，結果寫入 PR/handoff note。
- [ ] PR/handoff note 註明：結論是 hit / empty / inconsistent。
- [ ] 若 empty，P3 escape hatch 的 prompt 規則必須包含「明確區分三種 empty 來源」的話術。
- [ ] Integration test **必須同時包含 hit case 和 empty case**，不只測 hit。
- [ ] 不在 code 裡寫死「local data 一定有 / 一定沒有」這類前提。

### P1 驗收

- [ ] `rewrite_messages` wrapper 不再 hardcode 全量工具清單。
- [ ] active skill 下 rewriter 看到的工具清單與 writer 實際 allowed tools 一致。
- [ ] writer prompt 中有 ephemeral tool availability hint。
- [ ] 沒有 active skill 時，正常模式工具說明維持既有行為。
- [ ] tool availability hint 不會被存進 `recent_turns` 或 `chat_history`。
- [ ] grep `agent/thinking.py`，**不應該找到** hardcoded 的 `rag_explore` / `rag_search` / `recall_history` / `read_file` / `bash` / `web_search` / `github` 工具名稱字串。所有工具名稱都從 runtime / graph 取得。
- [ ] 無 active skill 時，rewriter 看到的工具清單應該是 base 全量（包含 MCP family，列 family name 不列每個 tool 全名以避免 prompt 膨脹），但這份清單是動態組裝的，不是字面 hardcoded。
- [ ] 共用 helper（`render_tool_availability_block` 或等效物）有自己的 unit test，至少測：active skill / no active skill / skill with `tool_policy.disallow` 三種情境的輸出。

### P2 驗收

- [ ] reviewer 能在 evidence trace 沒有 `recall_history` call 時要求先查，而不是直接問使用者補研究內容。
- [ ] reviewer 能在 `recall_history` empty 時允許 user-facing answer 說明查無足夠紀錄。
- [ ] reviewer 不會把 retrieval empty 包裝成一般 academic intake checklist。
- [ ] `needs_user_input=True` 時輸出給使用者的是可理解問題，不是內部 reviser 指令。
- [ ] scholarly integrity 仍然有效：沒有 evidence 時不得編造成果、數據、方法、citation。
- [ ] Reviewer 在 evidence trace 沒有預期 `recall_history` call 時，產出的 finding 形狀必須是 `severity=major, needs_user_input=False, decision=revise`，而不是 `needs_user_input=True`。可用單元測試驗證：餵特定的 evidence trace + tool availability，斷言 reviewer 輸出的 `route_review_report` 結果是 `"revise"` 而非 `"ask_user"`。

### P3 驗收

- [ ] 使用者明確質疑 agent 記憶或工具行為時，agent 不再繼續輸出 academic intake checklist。
- [ ] 回答會說明目前 active skill 與工具可用性。
- [ ] 回答會說明是否查過 history，以及查詢結果狀態。
- [ ] 不會自動清空 session 或 deactivate skill，除非使用者明確要求。

### P4 驗收

- [ ] `info.md` 補充 skill policy 與 tool availability 的關係。
- [ ] skill authoring 文件或相關段落補充 `history.search` 的使用時機。
- [ ] 文件明確區分 RAG KB、chat history、plan logs。
- [ ] 文件沒有承諾 `recall_history` 能查到 plan mode logs。

---

## 測試建議

### Unit tests

- skill capability resolution：
  - `academic-paper-writing` resolved tools contains `recall_history`
  - `bash` remains denied
- rewrite prompt construction：
  - no hardcoded full tool list
  - injected available tools appear
  - unavailable tools do not appear as available
- review prompt construction：
  - tool availability block appears
  - retrieval failure rules appear
- prompt history：
  - active skill tool availability hint appears after system prompt
  - no active skill 時不插入該 hint

### Integration-ish tests

- 使用 fake history store 注入一筆「一月上半成果」turn。
- 啟用 extended + academic-paper-writing。
- 發送「我一月上半的成果如果要寫成論文，abstract 重點是什麼？我不記得了，你自行看一下。」
- 驗證至少發生一次 `recall_history` tool call。
- 驗證回答不能在未檢索前直接要求使用者補完整研究背景。
- 若 fake history store 回傳 empty，驗證回答明確說明查無足夠紀錄，而不是編造 abstract。

---

## 給實作 agent 的提示

### 已知 edge case

- `recent_turns` 尚未 eviction 時，內容在 prompt 裡，不一定需要 `recall_history`。
- `plan_mode` turn 不進 chroma，`recall_history` 查不到。
- 使用者說「你之前看過」可能指：
  - recent visible context
  - persisted chat_history
  - plan_logs
  - indexed KB
  - repo file
- `rag_search` 不應被拿來替代 `recall_history` 查舊聊天。
- reviewer 的 stop message 目前可能直接暴露內部 `revision_instruction`，修 prompt 時要避免生成內部語氣。

### 現有 code 的依賴

- skill capability resolution 依賴 `agent/skills/capability_map.yaml`。
- `academic-paper-writing` 的 tool policy 來自 `skills/academic-paper-writing/manifest.yaml`。
- graph binding 依賴 active skill state 的 `allowed_tools` / `denied_tools`。
- extended mode rewrite / review pipeline 在 `agent/session.py` 與 `agent/thinking.py`。
- chat history store 與 RAG KB 是不同 collection / persist path。
- plan mode logs 與 chat_history 是不同儲存路徑。

### 容易踩的坑

- 不要只改 reviewer prompt，卻忘了 writer 根本沒 bind 到 `recall_history`。
- 不要只改 manifest，卻讓 rewriter / reviewer 繼續看到錯誤工具清單。
- 不要把 `recall_history` empty 當成使用者一定沒提供資料；可能資料在 plan logs。
- 不要為了讓 academic skill 查資料而重新允許 `bash`。
- 不要讓 reviewer 因為 scholarly integrity 過度保守，阻止「我查不到」這種誠實回答。
- 不要把工具可用性 hint 存入長期記憶。
- 不要承諾本地 `chat_history` 一定有一月上半資料；需要實測。
- 不要把「要求 reviser 先查」的 finding 標成 `needs_user_input=True`——這會直接走 `ask_user`，reviser loop 不會被觸發。

---

## 建議實作順序

1. P0：manifest 加 `history.search`，補對應測試。
2. P0.5：在本機驗 `recall_history` 對目標 query 是否命中，記錄結果。
3. P1：建立 shared tool availability block，接到 rewriter、writer。
4. P2：把 same block 接到 reviewer，補 retrieval failure 規則與 finding routing contract。
5. P3：處理 meta-conversation escape hatch。
6. P4：更新文件。
7. 用 `problem.md` 的對話流程做 regression test / manual replay。

---

## 完成定義

這次任務完成時，coding agent 應能證明：

- `academic-paper-writing` active skill 下，writer 可使用 `recall_history`。
- extended mode 的 rewriter、writer、reviewer 對工具可用性的認知一致。
- 使用者要求「去看我之前紀錄」時，agent 會先嘗試正確 retrieval path。
- retrieval empty 時，agent 會誠實說明查不到，而不是編造成果或把責任推回使用者填完整 intake checklist。
- 文件已更新，後續新增 skill 不會再誤以為 `rag.search` 等於 `history.search`。

---

## 結語

本 spec 已整合三項關鍵修正：P2 的 finding routing contract（修「P0 之後 reviser loop 仍可能跑不到」的 routing 漏寫）、P0.5 的後續處置分支（修「結論為 empty 時 spec 沒給後續路徑」的分支缺口）、P1 的 `tool_availability=""` fallback 規則（修「留 hardcoded 後門會讓 P4 防新 skill 踩坑的目的失效」的設計倒退）。P0 / P1 / P2 / P3 / P4 形成一條沒有縫隙的修正鏈。
