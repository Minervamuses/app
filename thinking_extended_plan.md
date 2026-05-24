# `/thinking extended` 暫定設計計劃 v3.4

## 背景

目前 `Minervamuses/app` 已經有 LangGraph agent、slash command、skills runtime、tool policy、context window 與 `academic-paper-writing` skill；v2 計劃實作了 `/thinking extended`，但 Compiler 走的是「結構化 TaskSpec」路線，與使用者腦中構想的「prompt 重寫 + 同一個 agent 改稿」差異較大。

v3 把 Extended mode 重新校準回原始構想：

- 使用者輸入先被攔截，由 prompt-master 重寫成更明確的 prompt
- 同一個 agent 拿重寫後的 prompt 去做事
- 結果交給第二個 reviewer agent 審查
- 原 agent 根據意見**修改或駁斥**，兩輪後輸出

v3.1 根據設計審核修正：

- Reviewer 輸入加 active skill context 與 Writer tool trace summary。
- prompt-master rewrite wrapper 明確禁止新增原始輸入沒有的事實。
- Reviser 駁斥透過 `DRAFT:` / `REBUTTAL:` marker 分段，controller 只把 DRAFT 段回給使用者。
- manifest tool_policy 範例用 broker 真正支援的明確工具名。
- 「最多兩輪」明確只指 Reviewer/Reviser loop；final skill validation 是 loop 外的單次安全網。

v3.2 修五個 v3.1 殘留問題：

- rewrite step messages 補上 `visible_context` 與 `active_skill_context`。
- Reviser 輸出缺 marker 時加 repair retry → 啟發式剝尾 → warning log 三段 fallback。
- review/revise loop pseudo-code 改寫為顯式 while-true。
- 暫定實作優先順序新增 `AgentConfig.thinking_tool_trace_chars` 與可選 `thinking_tool_trace_total_chars`。
- manifest 範例移除多餘的 `capabilities: required: [] optional: []`。

v3.3 修四個 v3.2 殘留問題：

- `tool_trace_summary` 改名 `evidence_trace_summary`，每次 Writer / Reviser graph 後 append 並帶輪次標籤。
- 新增 `rebuttal_history`，每輪 Reviewer 收到 `previous_rebuttal`。
- 修正 attempts cap 文案：pass 優先於 attempts cap，仍需 revise 才 stop。
- 新增 `thinking_rewrite_visible_chars` 與 `thinking_rewrite_skill_chars` 兩個 cap。

v3.4 補一個 v2 → v3.3 始終沒人問的盲點：

- **各個角色該用哪個 LLM 一直沒被指定。** 現況是 Writer / Reviewer / Reviser / format-repair / prompt-master rewrite 全部用 `config.llm_model`（GLM-5），等於同 model 自己審自己，blind spot 重疊。`AgentConfig.judge_llm_model = openai/gpt-5.2` 已存在但 Extended mode 沒用到。
- v3.4 在 `AgentConfig` 新增三個角色 model 欄位，**預設皆為空字串**，作為 forcing function：沒填 → Extended mode 啟動時 raise，逼使用者在開啟前明確選定。
- 同時新增 Reviewer 專用 max_tokens 上限，避免 ReviewReport JSON 被現有預設 1024 token 截斷。

```text
/thinking normal
/thinking extended
```

- `normal`：預設模式，維持目前 agent 行為。
- `extended`：高精度模式，先重寫 prompt，再走 Writer / Reviewer / Reviser 短流程。

---

## 核心結論

Extended mode 是：

```text
先用 prompt-master 重寫 user prompt（看得到 raw + visible + skill 三層 context，皆受 char cap）
→ 同一個 agent（Writer = graph）用重寫後的 prompt 跑既有 graph，產出 draft
→ Reviewer agent（獨立 model，跨 family）看 (raw + rewritten + draft + skill ctx
                    + 累積 evidence_trace_summary + previous_rebuttal)
  給出結構化 ReviewReport
→ Writer/Reviser agent 帶 reviewer 意見再跑一次 graph
  回應分 DRAFT: 與 REBUTTAL: 兩段，可修可駁
  controller 累積 evidence trace 與 rebuttal_history
→ Reviewer/Reviser loop 最多兩輪
  第二次 revise 後仍跑最後一次 reviewer：pass 則 pass，仍需 revise 才 stop
→ Final skill validation（loop 外的單次安全網）
→ 只把 DRAFT 段回給 user
```

五個角色 / 五個 model 槽：

| 角色 | 用的 model | config 欄位 | v3.4 預設 |
|---|---|---|---|
| Writer | 既有 graph，繼承 main agent model | `config.llm_model`（已存在） | `z-ai/glm-5` |
| prompt-master rewrite | 裸 LLM，獨立 model | `config.thinking_rewrite_model` | `""` (空，必填) |
| Reviewer | 裸 LLM，獨立 model（建議跨 family） | `config.thinking_reviewer_model` | `""` (空，必填) |
| Reviser | 既有 graph，跟 Writer 同 model（設計如此） | `config.llm_model` | `z-ai/glm-5` |
| Format-repair | 裸 LLM，獨立 model（便宜的即可） | `config.thinking_repair_model` | `""` (空，必填) |

成本上限：

```text
1 × prompt-master rewrite (裸 LLM, thinking_rewrite_model)
+ 1 × Writer graph (llm_model)
+ 最多 2 × (Reviewer 裸 LLM (thinking_reviewer_model) + Reviser graph (llm_model))
+ 最多 1 × Reviewer 裸 LLM (loop 退出前的最後一次審, thinking_reviewer_model)
+ 最多 1 × format-repair 裸 LLM (thinking_repair_model)
+ 最多 1 × validation graph (llm_model)
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

v3.4 補：切到 `/thinking extended` 時，handler 應檢查三個必填 model config 是否為空，若有任一為空，回覆說明「需要在 AgentConfig 設定 thinking_reviewer_model / thinking_rewrite_model / thinking_repair_model 才能使用 Extended mode」，**不切換** `thinking_mode`。避免使用者切完之後第一個 turn 才爆掉。

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

Normal mode **不受 v3.4 model config 缺值影響**——只有 Extended mode 會檢查 thinking_* model 欄位。

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
- manifest.yaml 可以給最小設定，但只在使用者手動 `/skill _prompt-master` 啟動時才有意義；Extended controller 本身直接讀 SKILL.md，不走 skill loader。

manifest.yaml 的 `tool_policy.disallow` 必須使用 broker 真正支援的語法：exact 工具名，或 `mcp_family.*` glob（見 [agent/skills/broker.py:210-221](agent/skills/broker.py#L210-L221)）。不支援 `rag_*` 這種前綴 glob。

```yaml
# skills/_prompt-master/manifest.yaml 建議內容
tool_policy:
  disallow:
    - bash
    - rag_explore
    - rag_search
    - rag_get_context
    - recall_history
    - read_file
    # 如要禁某 MCP family，使用 family.* 格式，如 web_search.*

resources: []
task_modes: []
```

（schema 允許省略 `capabilities` 區塊，當 `tool_policy.disallow` 已存在時不會被拒絕 — 見 [agent/skills/manifest_schema.py:58-70](agent/skills/manifest_schema.py#L58-L70)。）

### 4. `AgentConfig` 新增的 model 與 cap 欄位（v3.4 重點）

集中列出 Extended mode 需要的所有新欄位：

```python
# agent/config.py
@dataclass
class AgentConfig:
    # 既有欄位（不動）
    llm_model: str = "z-ai/glm-5"
    gen_llm_model: str = "google/gemini-3.1-pro-preview"
    judge_llm_model: str = "openai/gpt-5.2"
    filter_llm_model: str = "llama3.1:8b"
    ...

    # v3.4 新增 — Extended mode 角色 model（皆預設空字串，必填）
    thinking_reviewer_model: str = ""
    thinking_reviewer_max_tokens: int = 4096     # ReviewReport JSON 不被截斷
    thinking_rewrite_model: str = ""
    thinking_repair_model: str = ""

    # v3.2 / v3.3 新增 — char cap
    thinking_tool_trace_chars: int = 500         # 每 tool result excerpt
    thinking_tool_trace_total_chars: int = 4000  # evidence trace 總長度
    thinking_rewrite_visible_chars: int = 2000   # rewrite step visible context
    thinking_rewrite_skill_chars: int = 4000     # rewrite step active skill context
```

**為什麼三個 model 留空白：**

- 強迫使用者在開啟 Extended mode 前明確選定，避免不知不覺都用同一個 model 自己審自己。
- 沒有預設 fallback 到 `llm_model`，因為那等於回到 v3.3 以前的 self-review 反模式。
- 第一版只支援在 `agent/config.py` 的 `AgentConfig` 直接填入這三個欄位；不從 `.env` 讀取，也不做啟動參數覆蓋。
- 使用者之後手動修改 `agent/config.py`，例如：

```python
thinking_reviewer_model: str = "openai/gpt-5.2"
thinking_rewrite_model: str = "anthropic/claude-haiku-5"
thinking_repair_model: str = "meta-llama/llama-3.1-8b-instruct"
```

（建議：Reviewer 用跨 family 高階模型；rewrite 用中等強 instruction-following；repair 用最便宜的。）

**啟動檢查**：`ChatSession` 或 `/thinking extended` handler 載入時，若 `thinking_mode == "extended"` 或使用者要求切換，必須驗證三個欄位皆非空；任一為空則 raise `ExtendedModeNotConfigured`（新例外類）並附訊息列出缺哪幾個。

### 5. Extended workflow：graph 外部 controller（v3.4 多 model）

```python
if thinking_mode == "normal":
    invoke existing graph
    return

# extended
_require_thinking_models(config)   # raise ExtendedModeNotConfigured if any empty

rewrite_model = get_chat_model_for_role(config, role="rewrite")
reviewer_model = get_chat_model_for_role(config, role="reviewer")
repair_model = get_chat_model_for_role(config, role="repair")
# Writer / Reviser 不需獨立 model getter — 直接用既有 self.graph，
# graph 內部已綁 config.llm_model。

rewrite_result = rewrite_prompt(
    rewrite_model,
    skill_text=PROMPT_MASTER_SKILL_TEXT,
    user_input=raw_user_input,
    visible_context=trim_tail(
        session._visible_context_text(),
        config.thinking_rewrite_visible_chars,
    ),
    skill_context=trim_head(
        session._active_skill_context_block(),
        config.thinking_rewrite_skill_chars,
    ),
)
if isinstance(rewrite_result, Clarify):
    return_questions_to_user(rewrite_result.text)
    record_turn(...)
    return

rewritten_prompt = rewrite_result.prompt

writer_result = run_graph_turn(
    user_input=rewritten_prompt,
    extra_system_messages=[raw_hint, rewritten_hint],
)
draft = writer_result.answer

evidence_trace_summary = summarize_tool_trace(
    writer_result.tool_calls,
    writer_result.new_messages,
    source_label="[Writer]",
)
rebuttal_history: list[str] = []

attempts = 0
final_route = None
while True:
    report = review_draft(
        reviewer_model,                                # ← 跨 family Reviewer model
        raw_user_input=raw_user_input,
        rewritten_prompt=rewritten_prompt,
        draft=draft,
        skill_context=session._active_skill_context_block(),
        evidence_trace_summary=evidence_trace_summary,
        previous_rebuttal=rebuttal_history[-1] if rebuttal_history else "",
    )
    route = route_review_report(report, attempts=attempts)
    if route in {"pass", "ask_user", "stop"}:
        final_route = route
        break

    # route == 'revise'
    reviser_graph_result = run_graph_turn(
        user_input=rewritten_prompt,
        extra_system_messages=[
            raw_hint, rewritten_hint,
            previous_draft_hint(draft),
            reviewer_feedback_hint(report),
            reviser_instruction_hint(),
        ],
    )
    parsed = parse_reviser_output(
        reviser_graph_result.answer,
        repair_model=repair_model,                     # ← 便宜的 repair model
    )
    draft = parsed.draft

    evidence_trace_summary = append_tool_trace(
        evidence_trace_summary,
        reviser_graph_result.tool_calls,
        reviser_graph_result.new_messages,
        source_label=f"[Reviser round {attempts + 1}]",
        total_chars_cap=config.thinking_tool_trace_total_chars,
    )
    rebuttal_history.append(parsed.rebuttal)
    attempts += 1

final_answer = render_route_message(final_route, draft, report)
final_answer = apply_final_skill_validation(final_answer)
record_turn(final_answer)
return final_answer
```

新增的 helper：

```python
# agent/llm/openrouter.py (或新檔 agent/llm/thinking.py)

def get_chat_model_for_role(
    config: AgentConfig,
    *,
    role: Literal["reviewer", "rewrite", "repair"],
) -> ChatOpenAI:
    """Return a ChatOpenAI bound to the model assigned to a thinking-mode role.

    Raises ExtendedModeNotConfigured if the role's model field is empty.
    """
    model_attr = {
        "reviewer": "thinking_reviewer_model",
        "rewrite":  "thinking_rewrite_model",
        "repair":   "thinking_repair_model",
    }[role]
    model_name = getattr(config, model_attr)
    if not model_name:
        raise ExtendedModeNotConfigured(
            f"{model_attr} is empty; set it in agent/config.py AgentConfig "
            "before using /thinking extended."
        )
    max_tokens = (
        config.thinking_reviewer_max_tokens
        if role == "reviewer"
        else 1024
    )
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=model_name,
        temperature=0.3,
        max_tokens=max_tokens,
        max_retries=10,
    )


def _require_thinking_models(config: AgentConfig) -> None:
    missing = [
        attr
        for attr in ("thinking_reviewer_model", "thinking_rewrite_model", "thinking_repair_model")
        if not getattr(config, attr)
    ]
    if missing:
        raise ExtendedModeNotConfigured(
            "Extended mode requires these AgentConfig fields to be set: "
            + ", ".join(missing)
        )


class ExtendedModeNotConfigured(RuntimeError):
    pass
```

controller 邊界（沿用 v2）：

- 不改變 normal mode 的 graph。
- 不繞過 active skill context 與 tool policy。
- 不因 Extended mode 自動允許被 skill policy 禁止的工具或輸出。
- Writer 與 Reviser 都復用既有 graph，讓需要工具的任務仍能走既有 tool loop 與 skill loader。
- Reviewer / rewrite / repair 是 graph 外的裸 LLM，用獨立 model client。
- 若 controller 無法安全組合必要 context，應停止並要求補資料，而不是降級成自由發揮。

### 6. Prompt rewrite step（v3.2 補 context + v3.3 補 cap + v3.4 用獨立 model）

```text
model = rewrite_model  (from config.thinking_rewrite_model)

system = SKILL.md 全文
       + wrapper:
         "你是內部 pipeline 的一環。
          target tool 是一個 LangGraph research agent，
          可用工具：rag_explore / rag_search / rag_get_context /
                    recall_history / read_file / bash / MCP web_search /
                    MCP github / 使用者目前啟用的 active skill。
          重寫後的 prompt 應該是給這個 agent 看的自然語言指令。

          硬性禁令：你不得新增以下「原始輸入、visible context 與
          active skill context」三者都未提供的內容：
          - citation、DOI、page number、quote
          - 數據、樣本數、dataset 名稱、統計結果
          - 研究方法細節、實驗條件、研究發現
          - 對使用者意圖的擴張詮釋（例如把「改一下」自行擴張成
            『修改第三章的方法論段落』）
          若必要事實缺失，使用 <<CLARIFY>> 詢問使用者，不要自行補齊。

          若你需要使用者補充資訊：
            回應第一行寫 <<CLARIFY>>，之後列出最多 3 個澄清問題。
          若你判斷資訊足夠：
            直接輸出重寫後的 prompt，不要前綴、不要解釋、不要 code fence。"
user = "原始 user input:\n" + raw_user_input
       + "\n\nVisible context (recent turns, tail-truncated to N chars):\n"
       + (visible_context or "(none)")
       + "\n\nActive skill context (head-truncated to M chars):\n"
       + (skill_context or "(none)")
```

`visible_context` 與 `active_skill_context` 的取得與截斷：

- visible_context 來自 `session._visible_context_text()`，用 `trim_tail(text, config.thinking_rewrite_visible_chars)` 從頭部截斷，保留最近對話。預設 2000 字元。
- skill_context 來自 `session._active_skill_context_block()`，用 `trim_head(text, config.thinking_rewrite_skill_chars)` 從尾部截斷，保留 SKILL.md 開頭核心規則。預設 4000 字元。
- 截斷處附 `... [truncated]` 標記。

控制層解析：

- `response.lstrip().startswith("<<CLARIFY>>")` → 視為 clarification，把後續內容回傳給使用者，turn 結束。
- 其他情況 → 視為 `rewritten_prompt: str`，進 Writer。

失敗處理：LLM 呼叫失敗 / 超時，回覆固定錯誤訊息並停止本 turn。

### 7. Writer：既有 graph + rewritten prompt + evidence trace 初始化

```python
writer_result = await self._run_graph_turn(
    user_input=rewritten_prompt,                  # 主要 human message
    extra_system_messages=[
        SystemMessage("[Original user input]\n" + raw_user_input),
        SystemMessage("[Rewritten by prompt-master]\n" + rewritten_prompt),
    ],
)
draft = writer_result.answer
evidence_trace_summary = summarize_tool_trace(
    writer_result.tool_calls,
    writer_result.new_messages,
    source_label="[Writer]",
)
```

Writer 走的是既有 graph，因此使用 `config.llm_model`，不另設。

`summarize_tool_trace(tool_calls, new_messages, *, source_label)` 規格：

- 開頭一行 `=== {source_label} ===`。
- 對每個 tool_call 取出 `(name, args, result_excerpt)`，result_excerpt 截至前 `config.thinking_tool_trace_chars` 字（預設 500）。
- 同名工具多次呼叫合併或去重。
- 已有的 `format_tool_counts(tool_calls)` 可作為極端 fallback。

`append_tool_trace(existing, tool_calls, new_messages, *, source_label, total_chars_cap)` 規格：

- 呼叫 `summarize_tool_trace(...)` 取新段。
- 用 `\n\n` 接在 `existing` 後面。
- 若總長度超過 `total_chars_cap`（預設 4000），從頭部截斷舊內容，保留最近的證據，截斷處標 "... [older evidence truncated]"。

### 8. Reviewer：獨立 model + 結構化 ReviewReport（v3.4 多 model）

Reviewer 不直接改稿，只負責審查。**Model：`config.thinking_reviewer_model`**（必填，建議跨 family）。

Reviewer 用獨立 ChatOpenAI client，`max_tokens = config.thinking_reviewer_max_tokens`（預設 4096），避免 ReviewReport JSON 被截斷。

輸入：

- raw user input
- rewritten prompt
- draft
- active skill context（active skill runtime 的 `context_block()`，若無 active skill 留 "(none)"）
- evidence_trace_summary（累積至此的所有 Writer / Reviser tool 痕跡，含輪次標籤）
- previous_rebuttal（上一輪 Reviser 的 REBUTTAL 段，第一輪為空字串）

主要檢查維度（沿用 v2）：

- instruction following
- background logic
- method logic
- claim-evidence alignment（用 evidence trace 比對）
- citation integrity（用 evidence trace 驗證 citation 是否來自真正取得的來源；可區分 Writer 與 Reviser 哪一步加進來的）
- section coherence

額外的 v3 規則：

- 若 raw user input 與 rewritten prompt 之間有重大語意偏移，視為 finding。
- 若 draft 內出現 active skill context 禁止的內容，視為 finding。
- 若無法判定 draft 是否符合 raw user input 因為原始需求過於模糊，可標 `needs_user_input = true`。
- 若 `previous_rebuttal` 不為空且該輪 Reviser 對某 finding 提出合理的反對說明，應接受其判斷，不要為改而改。

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

### 9. Reviewer routing（v3.3 修正 attempts cap 描述）

```text
任何 finding needs_user_input = true
→ ask_user，停

decision = block 或任何 finding severity = blocker
→ ask_user，停

decision = pass
→ 直接輸出（只取 DRAFT 段）

attempts >= 2
→ stop，輸出當下 DRAFT + 「仍需確認處」

任何 major finding
→ revise（進 Reviser）

其他（只有 minor / note）
→ 直接輸出（只取 DRAFT 段）
```

關於 attempts cap：route_review_report 內順序是 `needs_user_input → blocker → pass → attempts cap → major → 其他`（見 [agent/thinking.py:95-114](agent/thinking.py#L95-L114)），**pass 優先於 attempts cap**。第二次 revise 完成後仍跑下一輪 reviewer；若 pass 則 pass，仍需 revise 才 stop。

### 10. Reviser：既有 graph + reviewer 意見 + DRAFT/REBUTTAL marker（v3.1 變更）

Reviser 跑既有 graph，等同「同一個 agent 帶著 reviewer 意見再做一次」，使用 `config.llm_model`：

```python
reviser_graph_result = await self._run_graph_turn(
    user_input=rewritten_prompt,
    extra_system_messages=[
        SystemMessage("[Original user input]\n" + raw_user_input),
        SystemMessage("[Rewritten by prompt-master]\n" + rewritten_prompt),
        SystemMessage("[Previous draft]\n" + draft),
        SystemMessage("[Reviewer feedback]\n" + report.model_dump_json(indent=2)),
        SystemMessage(_REVISER_INSTRUCTION),
    ],
)
parsed = parse_reviser_output(reviser_graph_result.answer, repair_model=repair_model)
draft = parsed.draft
rebuttal_history.append(parsed.rebuttal)
evidence_trace_summary = append_tool_trace(
    evidence_trace_summary,
    reviser_graph_result.tool_calls,
    reviser_graph_result.new_messages,
    source_label=f"[Reviser round {attempts + 1}]",
    total_chars_cap=config.thinking_tool_trace_total_chars,
)
attempts += 1
```

`_REVISER_INSTRUCTION`：

```text
你可以對每一個 reviewer finding 做以下其中之一：
(a) 修改 draft 以處理該 finding；
(b) 駁斥該 finding 並在 REBUTTAL 段說明你不同意的理由。

硬性禁令：
- 不要新增無法佐證的 citation / DOI / 數據 / 樣本數 / 方法細節 / 研究發現。
- 不要新增原始 user input 與可見 context 未提供的事實。

回應格式（必須嚴格遵守，使用兩個區段標記）：

DRAFT:
<新版 draft 全文。這段會被回給使用者，所以保持乾淨、不要包含內部審稿討論。>

REBUTTAL:
<對 reviewer findings 的反對說明；若無，寫 (none)。這段只給下輪 Reviewer 看，不會回給使用者。>
```

Reviser 的限制：

- 不新增 forbidden 內容（透過 system message 約束 + 後續 skill validation 兜底）。
- 若 finding 需要使用者補資料（`needs_user_input = true`），不會被分派到 Reviser（routing 攔截）。
- ReviewReport 包含 `blocker` 時，不會被分派到 Reviser（routing 攔截）。

### 11. Reviser 輸出解析與駁斥機制（v3.2 加固 fallback）

採用三段式 fallback：先正則解析，失敗則 retry 格式修復（使用 **`thinking_repair_model`**），再失敗則啟發式剝尾，最終才整段視為 DRAFT。

`parse_reviser_output(text, *, repair_model) -> RevisedDraft`：

```text
1. 正常路徑：regex 切 'DRAFT:' 與 'REBUTTAL:' marker
   - 兩個都在 → draft = DRAFT 段, rebuttal = REBUTTAL 段。完成。
   - 只有 DRAFT: 沒有 REBUTTAL: → draft = DRAFT: 之後到結尾, rebuttal = ""。完成。
   - 兩個都缺 → 進入 step 2。

2. Repair 路徑（最多一次，使用 thinking_repair_model）：
   呼叫 repair_model 一次，messages:
     system = "將下列文字嚴格拆成 DRAFT: 與 REBUTTAL: 兩段。..."
     user = 原始 reviser 輸出
   再做一次 step 1 regex 切。
   - 成功 → 完成。
   - 仍失敗 → 進入 step 3，並 log warning "reviser output marker repair failed"。

3. 啟發式剝尾 fallback：
   - 從文字尾端往前掃，把連續包含以下任一關鍵字的段落（用空行或 markdown header
     切段）視為內部審稿論述並剝離：
       "REBUTTAL", "rebuttal", "駁斥", "我不同意", "I disagree",
       "Reviewer feedback", "Internal note", "(none)"
   - 剝完後剩下的視為 draft。
   - 若剝離超過原文 50%，放棄剝尾，整段視為 draft，rebuttal = ""，
     並 log error。
   - 若剝完後 draft 為空字串，同樣放棄剝尾，整段視為 draft。

4. 最終 fallback：
   - draft = 整段原文, rebuttal = ""。
   - log error。
   - controller 在 final_answer 開頭附加一行 user-visible 註記：
     "（注意：本次回應的 reviser 輸出格式異常，可能混入內部審稿討論，請斟酌使用。）"
```

### 12. Final skill validation（v3.1 釐清語意）

Final skill validation 是 **review loop 外的一次性安全網**，不重複進入 Reviewer/Reviser loop：

- 觸發條件：active skill 啟用且 `_apply_final_skill_validation` 在 review loop 結束後仍偵測到違規。
- 行為：至多再跑一次 graph-based revision（透過既有 graph，使用 `config.llm_model`；graph 內部的 skill_validator node 本身已有 retry cap）。
- 不重新走 Reviewer / Reviser。
- validation 後的輸出直接回給使用者（仍只取 DRAFT 段；若 validation graph 的輸出沒有 marker，整段視為 DRAFT）。

成本上限（v3.4 完整版）：

```text
1 × prompt-master rewrite  (裸 LLM, thinking_rewrite_model)
+ 1 × Writer graph         (llm_model)
+ 最多 2 × (
      Reviewer 裸 LLM     (thinking_reviewer_model, max_tokens=4096)
    + Reviser graph        (llm_model)
  )
+ 最多 1 × Reviewer 裸 LLM (loop 退出前的最後一次審, thinking_reviewer_model)
+ 最多 1 × format-repair   (裸 LLM, thinking_repair_model)
+ 最多 1 × validation graph (llm_model)
```

「最多兩輪」明確指 Reviewer/Reviser loop，不包含 prompt-master rewrite、Writer 首次、loop 退出前的最後審、format-repair、與 final validation。

### 13. 測試與審核驗收條件

最低測試範圍：

- `/thinking normal|extended` slash command parsing、錯誤參數與 status 顯示。
- `/thinking extended` 當 `thinking_*_model` 任一為空時，**handler 拒絕切換並提示哪些欄位需要填**（不切到 extended，下個 turn 還是 normal）。
- `ChatSession.thinking_mode` 預設值與切換行為。
- Normal mode 不改變既有 graph flow，且不因 `thinking_*_model` 為空受影響。
- `get_chat_model_for_role(config, role=...)` 在欄位為空時 raise `ExtendedModeNotConfigured`，列出缺哪個欄位。
- `get_chat_model_for_role(config, role="reviewer")` 套用 `thinking_reviewer_max_tokens`（structural assertion）。
- prompt-master rewrite step：成功路徑（純自然語言 rewritten prompt）。
- prompt-master rewrite step：messages 含 raw user input、visible context、active skill context。
- prompt-master rewrite step：visible / skill context 超過 cap 時正確截斷並附 `[truncated]` 標記。
- prompt-master rewrite step：`<<CLARIFY>>` sentinel 觸發澄清回應，不進 Writer。
- prompt-master rewrite step：使用的是 `thinking_rewrite_model` 而非 `llm_model`。
- prompt-master rewrite step：LLM 失敗時 controller 安全停止。
- prompt-master wrapper 含「不得新增事實」禁令。
- Reviewer 輸入含 active skill context、evidence_trace_summary、previous_rebuttal。
- Reviewer 使用的是 `thinking_reviewer_model`，且 max_tokens 為 `thinking_reviewer_max_tokens`。
- Reviewer 對 `minor/note` 不觸發重寫。
- Reviewer 對可修正 `major` 觸發 Reviser，且最多兩輪。
- attempts cap 行為：第二次 revise 完成後仍跑一次 reviewer；若 reviewer 判 pass，最終 route 為 pass；只有當 reviewer 仍判 revise 才返回 stop。
- `blocker` 或 `needs_user_input = true` 不觸發 Reviser。
- Reviser 走的是既有 graph（使用 `llm_model`），且 reviewer feedback 出現在 prompt 中。
- Reviser 輸出含 `DRAFT:` / `REBUTTAL:` marker；parser 正確切段。
- Reviser 輸出缺 marker 時 repair retry 一次（使用 `thinking_repair_model`）；repair 成功則 draft 正確切出；repair 失敗則啟發式剝尾；啟發式剝離超過 50% 則整段視為 draft 並加 user-visible 警告。
- 回給使用者的 final answer 不含 `REBUTTAL:` 段內容（除非走到 step 4 fallback，此時應有 user-visible 警告）。
- Evidence trace 累加：Reviser 跑完後 `evidence_trace_summary` 含 `[Reviser round N]` 標籤段，且第二輪 Reviewer 的 prompt 含此段。
- Evidence trace 超過 `thinking_tool_trace_total_chars` 時從頭部截斷並標 "older evidence truncated"。
- Rebuttal history：第一輪 Reviewer prompt 的 `previous_rebuttal` 為空字串；第二輪含上一輪解析出來的 REBUTTAL 內容。
- 駁斥場景：Reviser 在 REBUTTAL 段提出論述，下一輪 Reviewer 接受後判 pass，且 final answer 為乾淨 DRAFT。
- Final skill validation 在 review loop 結束後最多跑一次，不會反覆觸發 Reviewer/Reviser。
- 所有 char cap config（`thinking_tool_trace_chars`、`thinking_tool_trace_total_chars`、`thinking_rewrite_visible_chars`、`thinking_rewrite_skill_chars`）截斷行為。
- 小型 eval set 應至少覆蓋 Reviewer 是否能抓出 major issue 與 academic integrity issue（沿用現有 `agent/evaluation/thinking.py`，但需指定 `thinking_reviewer_model` 才能跑）。

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

v3.3 進一步把 Reviser 的 tool trace 累積到 `evidence_trace_summary`，下一輪 Reviewer 才能驗證 Reviser 新增的 citation / 改寫所依據的來源。

### 4. prompt-master 取代 TaskSpec

v2 的 TaskSpec 是一份 14 欄位的結構化規格，原意是「Writer / Reviewer / Reviser 共用任務規格」，但實務上：

- 編譯成本不低，且 LLM 自由發揮的欄位（confidence、success_criteria）品質不穩
- Writer 仍需同時看 raw input + TaskSpec，TaskSpec 並未真正「定錨」任務
- Reviewer 的審查標準仍由它自己定義，TaskSpec 影響有限

v3 改用 prompt-master 重寫 prompt：

- 重寫後的 prompt 本身就是「更清晰的任務描述」
- prompt-master 是專門為「寫好的 prompt」設計的 skill，已涵蓋目標、限制、成功條件
- 不增加額外的 schema parsing 成本

### 5. 駁斥讓 Reviewer 不再霸權，但不污染輸出，也真的傳得到

v3 允許 Reviser 駁斥；v3.1 隔離到 `REBUTTAL:` 段；v3.2 補三段式 fallback；v3.3 補 `rebuttal_history` 真的傳給下一輪 Reviewer。

### 6. Reviewer 必須看到證據與規則才能可靠審查

v3.1 把 active skill context 與 Writer tool trace 加入 Reviewer 輸入；v3.3 把 trace 改成累積式 `evidence_trace_summary`，每輪 Reviewer 都看到截至當下所有 graph 的證據來源。

### 7. prompt-master 也必須看到 context 才能不亂編

v3.2 修正 v3.1 的自相矛盾；v3.3 為兩段 context 加 char cap，避免長 SKILL.md 把 rewrite prompt 撐爆。

### 8. 各角色用不同 model，避免 self-review 反模式（v3.4 新增）

v2 → v3.3 的隱性 default 是：所有 Extended mode 角色都用 `config.llm_model`（同一個 model）。這違反 critique-style 系統的基本原則：

- 同 family model 的 blind spots 重疊。Writer 沒看出來的問題，同一個 model 當 Reviewer 也很可能看不出來。
- Reviewer 缺乏「外部視角」，等於請寫稿的人改自己的稿。
- 高精度模式因此沒有真的更高精度——只是同個 model 跑了五次。

v3.4 強制三個角色（Reviewer / rewrite / repair）的 model 必須由使用者**明確選定**：

- Reviewer 用跨 family 模型（推薦復用既有 `judge_llm_model = openai/gpt-5.2`）
- rewrite 用中等強 instruction-following 模型
- repair 用最便宜的模型
- Writer / Reviser 維持 `llm_model`（兩者按設計就是「同一個 agent」）

預設留空 + 啟動時 raise，是刻意設計的 forcing function：

- 確保使用者在啟用 Extended mode 前已經想過 model 選擇與成本
- 避免不知不覺退化回 self-review
- 不提供 silent fallback 到 `llm_model`，因為那等於回到 v3.3 以前的反模式

### 9. 保留學術誠信底線

防線（v3.4 強化）：

- prompt-master 重寫時看得到 raw / visible / skill 三層 context（皆受 char cap）
- Reviewer 把 citation integrity 列為必檢維度，且擁有累積式 evidence trace
- Reviewer 用跨 family 模型，blind spots 不易重疊
- Reviser 的 system message 明列禁止項
- Reviser 走 graph 仍會經過 skill_validator
- DRAFT/REBUTTAL fallback 確保 Reviser 不守格式也不會悄悄漏內部討論給使用者
- Controller 層 final skill validation 再兜一層（loop 外的單次）

---

## 暫定實作優先順序

1. Vendor prompt-master 至 `skills/_prompt-master/`，加 manifest.yaml（使用 §3 的明確工具名格式）與 UPSTREAM_VERSION.md。
2. 在 `AgentConfig` 新增 **三個 model 欄位（預設空字串）+ 一個 max_tokens 欄位 + 四個 char cap**（見 §4 完整列表）。
3. 新增 `ExtendedModeNotConfigured` 例外類別與 `_require_thinking_models(config)` 輔助函式。
4. 新增 `get_chat_model_for_role(config, role)` helper（reviewer / rewrite / repair），缺值時 raise `ExtendedModeNotConfigured`，reviewer 套用 `thinking_reviewer_max_tokens`。
5. `/thinking extended` handler 增加 model 完備性檢查；缺值時拒絕切換並列出缺哪幾個。
6. 在 `agent/thinking.py` 拿掉 `TaskSpec`、`compile_task_spec`、`route_task_spec`、`render_task_spec_stop_message`、`append_assumption_note`、`task_spec_messages`、`revise_draft`、`reviser_messages`。
7. 新增純文字 helper：`trim_tail(text, max_chars)`、`trim_head(text, max_chars)`，截斷處附 `[truncated]` 標記。
8. 新增 `rewrite_prompt(model, *, skill_text, user_input, visible_context, skill_context) -> RewriteResult`，回傳 `Clarify(text=str)` 或 `Rewrite(prompt=str)`；包含 `<<CLARIFY>>` sentinel 偵測與「禁止新增事實」wrapper。
9. 新增 `summarize_tool_trace(tool_calls, new_messages, *, source_label, per_result_chars) -> str`。
10. 新增 `append_tool_trace(existing, tool_calls, new_messages, *, source_label, per_result_chars, total_chars_cap) -> str`。
11. 新增 `parse_reviser_output(text, *, repair_model) -> RevisedDraft`（含 draft / rebuttal 兩欄位 + §11 三段式 fallback）。
12. 改寫 `agent/session.py` 的 `_run_extended_turn`：
    - 啟動時呼叫 `_require_thinking_models(config)`
    - 三個獨立 model client：`get_chat_model_for_role(..., role="rewrite"/"reviewer"/"repair")`
    - 第一步呼叫 `rewrite_prompt`（含 trimmed visible_context + skill_context）
    - clarify → `_record_turn` 收尾
    - rewrite → Writer (`_run_graph_turn`)，初始化 `evidence_trace_summary` 與空的 `rebuttal_history`
    - Reviewer loop 用 `route_review_report`，Reviewer 輸入含 active skill context + evidence_trace_summary + previous_rebuttal
    - Reviser 改成 `_run_graph_turn` 帶 reviewer feedback + DRAFT/REBUTTAL instruction
    - 每次 Reviser 後：解析輸出（用 repair_model）、`evidence_trace_summary = append_tool_trace(...)`、`rebuttal_history.append(parsed.rebuttal)`
    - 任何回給使用者的分支都只回傳 draft 段；走到 step 4 fallback 時加 user-visible 警告
13. 更新 `_task_spec_hint` 對應的 helpers，改成 `_rewrite_hints` / `_reviser_hints` / `_reviewer_inputs`。
14. 更新 / 刪除 `tests/test_thinking.py` 與 `tests/test_thinking_session.py` 中 TaskSpec 相關測試；新增 §13 列出的所有新測試。
15. `agent/evaluation/thinking.py` 不動（但要在 README 註明跑 thinking suite 前需設 `thinking_reviewer_model`）。
16. 更新 README / SKILLS_GUIDE.md：列出 Extended mode 啟用所需的 `agent/config.py` 欄位，並建議分配。

---

## 暫定一句話總結

`/thinking extended` 的目標是把高精度任務變成一個短流程：**用 prompt-master (獨立 model) 把 user prompt 變清楚，同一個 agent (Writer = graph，用 llm_model) 拿去做事，跨 family 的 Reviewer (獨立 model，max_tokens 加大) 看著 draft + 累積式 evidence trace + active skill context + 上一輪 rebuttal 審查，原 agent (Reviser = graph，用 llm_model) 可修可駁（駁斥放在 REBUTTAL: 段，缺 marker 時有 repair model + 剝尾 + 警告三段 fallback），review loop 兩輪後若 reviewer pass 就 pass、仍需 revise 才 stop，外加一次 final skill validation 兜底；三個獨立 model 欄位預設空字串，沒填 Extended mode 拒絕啟動，逼使用者明確選擇**。
