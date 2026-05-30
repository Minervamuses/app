# Fix Plan — Agent 工具呼叫爆走(tool-call runaway)

日期:2026-05-30
狀態:Phase 0(instrument)已完成並跑出證據;優先序已依證據重排(見下)。
偵錯全紀錄:`note/20260530/tool_call_runaway_debug.md`

## 背景:這個問題是怎麼浮出來的

1. 做 evaluator survey 時跑 c1(工具路由評測),發現某些 case 的 agent 對一個簡單問題叫了 **15-16 次工具**。
2. 一度以為是「逃去 bash」的工具選擇問題,但比對兩次完成的 trace 後確認:**agent 是 RAG 優先**,狂試 RAG 12-13 次,bash/read_file 只是最後迫不得已的逃生。
3. 使用者回報:**換 GLM5 也一樣**(簡單問題叫十幾次工具)→ 跨模型共通 → 嫌疑指向 harness/context,不是模型。
4. 期間還踩到兩個 side issue(已釐清,非本 plan 重點):
   - eval 在終端機跑時,runaway agent 逃去 bash 會卡在核准提示;之後又卡在**靜默等 reasoning model 回合**(無 timeout、無進度輸出)。
   - 沙箱會用 /dev/null 覆蓋家目錄 dotfile,導致 coding agent 誤報「Ollama 沒開 / repo 有 dotfile」。

## 真正的 root cause:兩層裁切 + 上限與視窗不對齊

系統有**兩層獨立的歷史裁切**:

- **Layer 1(turn 級,設計如預期)**:`ChatSession.recent_turns` 保留最近 `agent_recent_turns_window=10` 個 turn;更舊的由 `_evict_overflow` 踢進 Chroma,之後靠 `recall_history` 取回。prior turn 以乾淨 `[Human, AI(answer)]` 形式保存(tool 訊息已收斂)。
- **Layer 2(turn 內 tool 結果級,bug 所在)**:`prepare_messages_for_agent`([agent/history.py:50](agent/history.py#L50))每個 agent step 只在 prompt 裡保留**最近 `agent_max_tool_interactions=4` 次 tool 結果**,更早的裁掉並留一句 truncation note。

### 兩個互相加成的瑕疵

**瑕疵 A — Context 失憶(Layer 2)**
- 一個 turn 內 tool call 超過 4 次後,prompt 只剩最近 4 筆結果,**更早的搜尋結果(同一個 turn 的)被藏起來**。
- 後果:agent 看不到自己稍早搜到什麼 → 以為還沒找到 → 重搜 → 失憶迴圈。**模型無關**(GLM5 也中)。
- `4` 這個數字**從未經實驗驗證**,當初是賭「一個 turn 頂多 4 次 call」,所以 Layer 2 原以為是 no-op;call 爆開後它就開始即時吃掉自己的結果。

**瑕疵 B — 工具預算沒咬住(graph.py)**
- [agent/graph.py:126-128](agent/graph.py#L126) 有 per-turn 預算:`tool_count >= agent_max_tool_interactions(4)` → 改用不綁工具的模型 → 應該硬停。
- 但實測 2 個 turn 共 15-16 次,遠超 `4×2=8`。代表預算**沒有有效咬住**。候選機制(需 instrument 確認):
  - (a) **單回合溢出**:預算只擋「這回合要不要動手」,不限制一個 AIMessage 裡的 parallel tool_calls 數;count=3 時模型一口氣發多個就衝破。
  - (b) **計數未登錄**:`_tool_interaction_count` 認 `ToolMessage` 型別,若某路徑回的不是該型別則永遠數不到 → 永不 exhausted → 跑到模型自停或撞 recursion_limit(32)。
  - 兩者都導致「**真正執行的 call 次數 ≫ 看得到的視窗(4)**」。

### 一句話

> **真正的 call 次數上限(實際 15-16)和「看得到的歷史視窗」(4)嚴重不對齊。Layer 2 把超出視窗的早期結果藏起來製造失憶迴圈;預算(瑕疵 B)又沒及時掐斷。兩者疊加 → 跨模型爆走。**

## Phase 0 證據(2026-05-30 跑 run `...095849Z`)— root cause 重新定性

instrument 後跑一次,三個決定性證據:

1. **預算為何沒咬住 = (a) parallel 溢出**:`max_emitted_per_step = 2`(一個 AIMessage 塞 2 個 tool_call,衝破回合前的預算檢查)。不是計數未登錄(否則會撞 recursion_limit=32,但停在 9)。
2. **失憶確實存在**:embedding case 的 query 第 5≈第 2、第 6 又 explore,且 step_log 顯示第 5 筆=第 3 筆、第 8 筆=第 2 筆是重複的同一個結果 → 看不到自己拿過 → 重發。
3. **(關鍵)檢索回的全是垃圾,因為答案不在語料庫**:embedding case 七次檢索回 `JmolWidgetset cache.html`、`3dmol.d.ts`、`poetry.toml`、`SaveToLink.java` 等,**零個相關**。原因:`rag.cli.ingest` 自動跳過 this workspace,而「embedding 模組」(bge-m3/OllamaEmbedder)正住在被排除的 workspace → **目標內容不在索引裡,檢索注定回垃圾**。

**重新定性(主因換人):**

| 層 | 角色 | |
|---|---|---|
| **語料缺口 / 不可答查詢** | **主因** | 目標內容被 ingest 排除在索引外,沒有滿意結果可收手 |
| **agent 無「放棄紀律」** | 次因 | prompt 沒教「搜幾次都是垃圾就下結論:KB 沒有」,它不認輸一直重搜 |
| **Context 失憶(Layer 2)** | 共犯 | 看不到自己拿過的垃圾 → 重發;修了能讓它更快放棄 |
| **預算溢出(瑕疵 B,(a) parallel)** | 放大器 | 把「找不到」放大成 9-16 次 |

> 原本以為「對齊 cap/window」是核心。證據顯示它只是**止血**——就算記憶完美,答案不在語料裡,模型還是會卡。真正高槓桿的是**放棄紀律**和**修評測題**。

## 修復原則(使用者拍板)

**讓「真正的 call 次數硬上限」== 「prompt 裡看得到的 tool 結果數量」。**
只要兩者對齊,就**永遠不會裁掉一個還被允許繼續使用的結果** → 失憶從根消失。
(此原則仍要做,定位為「止血」;主軸見下方重排後的計畫。)

## 修復計畫(分階段,優先序已依 Phase 0 證據重排)

### Phase 0 — instrument 找根因 ✅ 已完成
- 在 `c1_routing.py` 用 `progress_cb` 記錄 emitted tool_calls 數 + tool 結果預覽,detail 加 `actual_args`/`max_emitted_per_step`/`step_log`(不改計分,commit `72ff55f`)。
- 結論:見上方「Phase 0 證據」。(a) parallel 溢出已確認;主因是語料缺口 + 無放棄紀律,失憶為共犯。

### Phase 1 — 放棄紀律(新核心,最高槓桿)
- 給 agent 明確規則:**連續 N 次檢索都不相關 / 命中空 → 停止搜尋,結論「KB 裡沒有此內容」並照實回答**,而不是換句話重搜。
- 落點:system prompt 加明確 give-up 指示;或在 graph 層偵測「連續低品質/重複命中」後注入強制收斂訊息。
- 驗收:對「答案不在語料庫」的查詢(如 embedding case),agent 在少數幾次搜尋後收手並誠實說「找不到」,不再 9-16 次。

### Phase 2 — 修評測題(新核心)
- c1 embedding case 的 gold 期望 `rag_get_context`,但目標內容不在索引裡 → 近乎不可通過。
- 兩條路擇一:(i) 改寫成語料裡真有的主題;(ii) 重新分類成「優雅放棄」測試,gold 改為「少數搜尋後得出 KB 沒有」。
- 驗收:該 case 的 gold 與「這份語料實際可達成的正確行為」一致。

### Phase 3 — 對齊上限與視窗(止血,非核心)
- 讓預算成為**真正的硬上限 N**:修 (a) 溢出 —— 把 parallel tool_calls 納入扣額(發之前先看剩餘額度),確保實際執行數 ≤ N;決定 scope(per-turn vs per-conversation)。
- 讓 Layer 2 視窗 == N,確保到上限前**所有 tool 結果都看得到**(失憶從根消失,放棄判斷也更準)。
- 驗收:單 turn 內實際 call 數 ≤ N,且 prompt 看得到全部 N 筆。

### Phase 4 — 決定 N 的平衡點(實驗,非拍腦袋)
- N 太小 → turn 做不完;N 太大 → 成本/爆走風險。用 c1/c4 在不同 N 下跑,挑平衡點。
- 驗收:N 有數據支撐,不再是「賭一個 4」。

### Phase 5 — 防呆
- chat model 設 `timeout`(避免靜默掛 ~10 分鐘)。
- eval loop 加進度輸出(讓「靜默」和「當掉」可區分)。
- eval 場景 bash 自動 deny(中和執行、保留違規計分)。

## 待決問題(需使用者/實驗回答)

1. **放棄紀律的觸發條件**(Phase 1):「不相關」怎麼判定?連續幾次空命中/低品質就收手?要靠 prompt 規則還是 graph 層偵測?
2. **評測題處置**(Phase 2):embedding case 改寫成語料裡真有的主題,還是改成「優雅放棄」測試?
3. 硬上限 scope(Phase 3):per-turn 還是 per-conversation?
4. N 的初始值與調法(Phase 4 實驗設計)。
5. 超過上限時的行為:保留「改用不綁工具的模型強制 synthesize」,還是給更明確的「列出缺什麼」指示?

### 已解(Phase 0)
- ~~觸發層是檢索品質差還是失憶~~ → **主因是語料缺口**(目標內容被 ingest 排除在索引外),失憶為共犯。
- ~~預算沒咬住是 (a) 還是 (b)~~ → **(a) parallel 溢出**。

## 受影響檔案(預估)
- `agent/graph.py`(agent_node 預算邏輯)
- `agent/history.py`(`prepare_messages_for_agent` 視窗)
- `agent/config.py`(`agent_max_tool_interactions` 語意/值)
- `agent/evaluation/claims/c1_routing.py`(Phase 0 instrument:存 args/results)
