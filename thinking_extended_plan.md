# `/thinking extended` 暫定設計計劃 v3

## 背景

目前 `Minervamuses/app` 已經有 LangGraph agent、slash command、skills runtime、tool policy、context window 與 `academic-paper-writing` skill；v2 計劃實作了 `/thinking extended`，但 Compiler 走的是「結構化 TaskSpec」路線，與使用者腦中構想的「prompt 重寫 + 同一個 agent 改稿」差異較大。

v3 把 Extended mode 重新校準回原始構想：

- 使用者輸入先被攔截，由 prompt-master 重寫成更明確的 prompt
- 同一個 agent 拿重寫後的 prompt 去做事
- 結果交給第二個 reviewer agent 審查
- 原 agent 根據意見**修改或駁斥**，兩輪後輸出

```text
/thinking normal
/thinking extended
```

- `normal`：預設模式，維持目前 agent 行為。
- `extended`：高精度模式，先重寫 prompt，再走 Writer / Reviewer / Reviser 短流程。

本版相對 v2 的主要修正：

- 拿掉結構化 `TaskSpec`，Compiler 改為呼叫 `prompt-master` skill 重寫 prompt。
- Reviser 從「graph 外裸 LLM 呼叫」改為「既有 graph 帶 reviewer 意見再跑一次」，保留工具能力。
- 新增「駁斥」語意：Reviser 可以選擇修改 draft，也可以對 finding 提出反對意見。
- 澄清機制改由 prompt-master 透過 `<<CLARIFY>>` sentinel 主動提出。

---

## 核心結論

Extended mode 是：

```text
先用 prompt-master 重寫 user prompt
→ 同一個 agent 用重寫後的 prompt 跑既有 graph，產出 draft
→ Reviewer agent 看 (raw user input + rewritten prompt + draft)，給出結構化意見
→ 原 agent 帶著 reviewer 意見再跑一次 graph：可修改、可駁斥
→ 最多兩輪，硬停
→ Final skill validation
→ 回 user
```

---

## 計劃

### 1. `/thinking` slash command（保留 v2）

`/thinking normal|extended` 與 `ChatSession.thinking_mode` 行為不變：

- 預設 `thinking_mode = "normal"`
- `/thinking normal|extended` 切換
- `/status` 顯示目前 `thinking_mode`
- 無效參數回覆可用選項，不呼叫 LLM
- session-scoped 狀態，不要求跨 process persistence

### 2. Normal 模式維持現狀（保留 v2）

```text
user
→ agent
→ tools if needed
→ agent
→ skill_validator
→ final answer
```

切回 `/thinking normal` 後，既有 graph node、tool loop、skill validator 與回覆格式不應因 Extended mode 變更而改變。

### 3. Vendor prompt-master skill

從 `nidhinjs/prompt-master`（MIT，v1.6.0）一次性拷貝至：

```text
skills/_prompt-master/
  SKILL.md
  references/
    patterns.md
    templates.md
  manifest.yaml          # 套上本 repo 的 manifest schema
  UPSTREAM_VERSION.md    # 記錄上游 SHA / tag / 拷貝日期
```

說明：

- 同步策略：一次性 vendor，commit message 註明上游來源與版本。未來上游更新時手動 re-vendor。
- `_` 前綴是內部 helper 慣例；是否從 `/skill` 選單隱藏暫不處理，可日後再加 `discover_skills` 過濾。
- manifest.yaml 可以給最小設定（`tool_policy.disallow: [bash, rag_*, read_file, recall_history, ...]`），因為 prompt-master 不需要工具，但這只在使用者手動 `/skill` 啟動時才有意義；Extended controller 本身直接讀 SKILL.md，不走 skill loader。

### 4. Extended workflow：graph 外部 controller（保留 v2 §3 思路）

第一版仍不要直接重拆 LangGraph nodes，在現有 chat/session invoke 外層加 controller：

```text
if thinking_mode == "normal":
    invoke existing graph

if thinking_mode == "extended":
    rewrite prompt via prompt-master
    if <<CLARIFY>>:
        return clarifying questions to user
    else:
        run Writer (existing graph) with rewritten prompt
        loop up to 2 review/revise rounds:
            run Reviewer (naked LLM)
            route ReviewReport
            if revise:
                run Reviser (existing graph) with reviewer feedback
            else:
                break
        run final skill validation
        return final answer
```

controller 邊界（沿用 v2）：

- 不改變 normal mode 的 graph。
- 不繞過 active skill context 與 tool policy。
- 不因 Extended mode 自動允許被 skill policy 禁止的工具或輸出。
- Writer 與 Reviser 都復用既有 graph，讓需要工具的任務仍能走既有 tool loop 與 skill loader。
- 若 controller 無法安全組合必要 context，應停止並要求補資料，而不是降級成自由發揮。

### 5. Prompt rewrite step

呼叫一次裸 LLM，組 messages：

```text
system = SKILL.md 全文
       + wrapper:
         "你是內部 pipeline 的一環。
          target tool 是一個 LangGraph research agent，
          可用工具：rag_explore / rag_search / rag_get_context /
                    recall_history / read_file / bash / MCP web_search /
                    MCP github / 使用者目前啟用的 active skill。
          重寫後的 prompt 應該是給這個 agent 看的自然語言指令。

          若你需要使用者補充資訊：
            回應第一行寫 <<CLARIFY>>，之後列出最多 3 個澄清問題。
          若你判斷資訊足夠：
            直接輸出重寫後的 prompt，不要前綴、不要解釋、不要 code fence。"
user = 原始 user input
```

控制層解析：

- `response.lstrip().startswith("<<CLARIFY>>")` → 視為 clarification，把後續內容回傳給使用者，turn 結束。
- 其他情況 → 視為 `rewritten_prompt: str`，進 Writer。

失敗處理：LLM 呼叫失敗 / 超時，回覆固定錯誤訊息並停止本 turn。

### 6. Writer：既有 graph + rewritten prompt

```python
writer_result = await self._run_graph_turn(
    user_input=rewritten_prompt,                          # 主要 human message
    extra_system_messages=[
        SystemMessage("[Original user input]\n" + raw_user_input),
        SystemMessage("[Rewritten by prompt-master]\n" + rewritten_prompt),
    ],
)
draft = writer_result.answer
```

原因：

- 用 rewritten prompt 當主要 human message，讓 agent 把重寫後的版本當作真正要解的問題。
- raw user input 仍以 system hint 保留，防止 prompt-master 編譯偏掉時完全覆蓋原意。
- 透過既有 graph，仍享有 RAG、read_file、bash gating、MCP、active skill loader 與 skill_validator。

Writer 的任務只是產生 draft，不做自我審查。

### 7. Reviewer：裸 LLM + 結構化 ReviewReport（保留 v2 schema）

Reviewer 不直接改稿，只負責審查。輸入：

- raw user input
- rewritten prompt
- draft

主要檢查維度（沿用 v2）：

- instruction following
- background logic
- method logic
- claim-evidence alignment
- citation integrity
- section coherence

額外的 v3 規則：

- 若 raw user input 與 rewritten prompt 之間有重大語意偏移，視為 finding。
- 若無法判定 draft 是否符合 raw user input 因為原始需求過於模糊，可標 `needs_user_input = true`。

schema 不變：

```python
class ReviewFinding(BaseModel):
    severity: Literal["blocker", "major", "minor", "note"]
    dimension: str
    location: str
    problem: str
    evidence_from_draft: str
    revision_instruction: str
    needs_user_input: bool

class ReviewReport(BaseModel):
    decision: Literal["pass", "revise", "block"]
    findings: list[ReviewFinding]
    summary_for_reviser: str
```

### 8. Reviewer routing（保留 v2 §8）

```text
任何 finding needs_user_input = true
→ ask_user，停，列出需使用者補的資訊

decision = block 或任何 finding severity = blocker
→ ask_user，停

decision = pass
→ 直接輸出

attempts >= 2
→ stop，輸出當下 draft + 「仍需確認處」

任何 major finding
→ revise（進 Reviser）

其他（只有 minor / note）
→ 直接輸出
```

明確規則：

- `blocker` 不進 Reviser。
- 需要使用者補資料的 finding 不進 Reviser。
- `needs_user_input = true` 優先級高於 `major`。

### 9. Reviser：既有 graph + reviewer 意見（v3 重點變更）

Reviser 不再是 graph 外的裸 LLM，改成跑既有 graph，等同「同一個 agent 帶著 reviewer 意見再做一次」：

```python
reviser_result = await self._run_graph_turn(
    user_input=rewritten_prompt,
    extra_system_messages=[
        SystemMessage("[Original user input]\n" + raw_user_input),
        SystemMessage("[Rewritten by prompt-master]\n" + rewritten_prompt),
        SystemMessage("[Previous draft]\n" + draft),
        SystemMessage("[Reviewer feedback]\n" + report.model_dump_json(indent=2)),
        SystemMessage(
            "你可以對每一個 finding 做以下其中之一：\n"
            "(a) 修改 draft 以處理該 finding；\n"
            "(b) 駁斥該 finding 並在回應中說明你不同意的理由。\n"
            "不要新增無法佐證的 citation / DOI / 數據 / 樣本數 / 方法細節 / 研究發現。\n"
            "回應就是新版 draft 全文（如有駁斥也寫在裡面）。"
        ),
    ],
)
draft = reviser_result.answer
attempts += 1
```

Reviser 的限制：

- 不新增 forbidden 內容（透過 system message 約束 + 後續 skill validation 兜底）。
- 若 finding 需要使用者補資料（`needs_user_input = true`），不會被分派到 Reviser（routing 攔截）。
- ReviewReport 包含 `blocker` 時，不會被分派到 Reviser（routing 攔截）。

### 10. 駁斥機制（v3 新增）

採用最簡實作：

- Reviser 的回應就是新版 draft 全文，可以包含對 finding 的反對說明。
- 不要求結構化 per-finding decision，避免逼 agent 跳出 graph 寫 JSON。
- 下一輪 Reviewer 自然會看到新 draft（包含駁斥論述），由 Reviewer 判斷接受或繼續挑。
- Reviewer system prompt 加一句：「若 Reviser 對某 finding 提出合理的反對說明，應接受其判斷，不要為改而改。」
- 兩輪硬上限作為終極兜底，避免無限拉扯。

### 11. Final skill validation（保留 v2）

Reviser 走的是既有 graph，本身就會經過 skill_validator node。

但保留 controller 層的 `_apply_final_skill_validation`，原因：

- 若 active skill 啟用且回覆仍有違規（例如 retry 上限耗盡），controller 層可以再做一次 graph-based revision 嘗試補救。
- 提供「Reviser 後的最後一道防線」這個明確語意。

實作維持 v2 現狀。

### 12. 測試與審核驗收條件

最低測試範圍：

- `/thinking normal|extended` slash command parsing、錯誤參數與 status 顯示。
- `ChatSession.thinking_mode` 預設值與切換行為。
- Normal mode 不改變既有 graph flow。
- prompt-master rewrite step：成功路徑（純自然語言 rewritten prompt）。
- prompt-master rewrite step：`<<CLARIFY>>` sentinel 觸發澄清回應，不進 Writer。
- prompt-master rewrite step：LLM 失敗時 controller 安全停止。
- Reviewer 對 `minor/note` 不觸發重寫。
- Reviewer 對可修正 `major` 觸發 Reviser，且最多兩輪。
- `blocker` 或 `needs_user_input = true` 不觸發 Reviser。
- Reviser 走的是既有 graph（測試確認 graph 被呼叫，且 reviewer feedback 出現在 prompt 中）。
- 駁斥場景：Reviser 回應含駁斥論述、Reviewer 接受後判 pass。
- 小型 eval set 應至少覆蓋 Reviewer 是否能抓出 major issue 與 academic integrity issue（沿用現有 `agent/evaluation/thinking.py`）。

---

## 為什麼這樣設計

### 1. 使用者心理預期清楚

Extended mode 是使用者主動開啟的高精度模式，因此等待時間增加是可接受的。  
Normal mode 則維持快速回應。

### 2. 避免無限審查 loop

審查最多兩輪，並明確定義：

- pass
- revise
- block
- ask_user
- max attempts

避免模型在沒有新資訊的情況下反覆改寫、成本失控。

### 3. 同一個 agent 改稿，保留工具能力

v2 把 Reviser 設成裸 LLM，導致 Reviser 想補資料、查 RAG、讀檔時無計可施。v3 讓 Reviser 走既有 graph：

- Reviser 仍有 RAG / read_file / bash / MCP / active skill
- 修改意見涉及「查證再答」時可以實際查
- 保持與 Writer 同一個 agent 的人格與工具集

### 4. prompt-master 取代 TaskSpec

v2 的 TaskSpec 是一份 14 欄位的結構化規格，原意是「Writer / Reviewer / Reviser 共用任務規格」，但實務上：

- 編譯成本不低，且 LLM 自由發揮的欄位（confidence、success_criteria）品質不穩
- Writer 仍需同時看 raw input + TaskSpec，TaskSpec 並未真正「定錨」任務
- Reviewer 的審查標準仍由它自己定義，TaskSpec 影響有限

v3 改用 prompt-master 重寫 prompt：

- 重寫後的 prompt 本身就是「更清晰的任務描述」
- prompt-master 是專門為「寫好的 prompt」設計的 skill，已涵蓋目標、限制、成功條件
- 不增加額外的 schema parsing 成本

### 5. 駁斥讓 Reviewer 不再霸權

審查流程的常見問題是 Reviewer 提出邊際 finding、Reviser 為改而改、品質反而下降。v3 允許 Reviser 駁斥：

- 合理駁斥 → Reviewer 下輪接受 → 收斂
- 不合理駁斥 → Reviewer 繼續挑 → 兩輪硬停
- 永遠保留「不改」的選項，讓 Reviser 有自主判斷空間

### 6. 保留學術誠信底線

即使在 Extended mode，也不能為了讓稿件看起來完整而捏造：

- citation
- DOI
- page number
- sample size
- dataset
- statistics
- method details
- research findings

防線：

- prompt-master 重寫時不會自己加事實
- Reviewer 把 citation integrity 列為必檢維度
- Reviser 的 system message 明列禁止項
- Reviser 走 graph 仍會經過 skill_validator
- Controller 層 final skill validation 再兜一層

---

## 暫定實作優先順序

1. Vendor prompt-master 至 `skills/_prompt-master/`，加 manifest.yaml 與 UPSTREAM_VERSION.md。
2. 在 `agent/thinking.py` 拿掉 `TaskSpec`、`compile_task_spec`、`route_task_spec`、`render_task_spec_stop_message`、`append_assumption_note`、`task_spec_messages`、`revise_draft`、`reviser_messages`。
3. 新增 `rewrite_prompt(model, *, skill_text, user_input) -> RewriteResult`，回傳 `Clarify(questions=str)` 或 `Rewrite(prompt=str)`；包含 `<<CLARIFY>>` sentinel 偵測。
4. 改寫 `agent/session.py` 的 `_run_extended_turn`：
   - 第一步呼叫 `rewrite_prompt`
   - clarify → `_record_turn` 收尾
   - rewrite → Writer (`_run_graph_turn`)
   - Reviewer loop 用 `route_review_report`
   - Reviser 改成 `_run_graph_turn` 帶 reviewer feedback
5. 更新 `_task_spec_hint` 對應的 helpers，改成 `_rewrite_hints` / `_reviser_hints`。
6. 更新 / 刪除 `tests/test_thinking.py` 與 `tests/test_thinking_session.py` 中 TaskSpec 相關測試。
7. 新增測試覆蓋 prompt-master 兩個分支、Reviser 走 graph、駁斥場景。
8. `agent/evaluation/thinking.py` 不動（Reviewer eval cases 不受影響）。
9. 更新 `agent/cli/eval.py`（不動）與 README / SKILLS_GUIDE.md 中 Extended mode 的描述（如有提及）。

---

## 暫定一句話總結

`/thinking extended` 的目標是把高精度任務變成一個短流程：**用 prompt-master 把 user prompt 變清楚，同一個 agent 拿去做事，第二個 agent 審查，原 agent 可修可駁，兩輪後輸出**。
