# Fix Plan — Agent 工具呼叫爆走(tool-call runaway)

日期:2026-05-30
狀態:規劃中(尚未動手改 code)

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

## 修復原則(使用者拍板)

**讓「真正的 call 次數硬上限」== 「prompt 裡看得到的 tool 結果數量」。**
只要兩者對齊,就**永遠不會裁掉一個還被允許繼續使用的結果** → 失憶從根消失。

## 修復計畫(分階段)

### Phase 0 — 先 instrument,用證據決定(不盲修)
- 在 eval 的 failed case details 存下每次 tool call 的 **args + 截斷 results**,以及每個 AIMessage 的 **tool_calls 數量**。
- 跑一次 c1 embedding case,釘死:
  - 瑕疵 B 是 (a) 溢出 還是 (b) 計數未登錄?
  - 觸發層是「檢索品質差」還是「Layer 2 失憶造成的假性沒找到」?
- 驗收:能明確指出 15-16 次是怎麼累積出來的。

### Phase 1 — 對齊上限與視窗(核心修復)
- 讓預算成為**真正的硬上限 N**:
  - 修瑕疵 B:把 parallel tool_calls 也納入扣額(發之前先看剩餘額度),或修正計數,確保實際執行數 ≤ N。
  - 決定 scope:per-turn(現況)是否足夠,或需要 per-conversation/per-case 上限。
- 讓 Layer 2 視窗 == N(同一個 config 值),確保到上限前**所有 tool 結果都看得到**。
- 驗收:單 turn 內實際 call 數 ≤ N,且 prompt 裡看得到全部 N 筆;c1 embedding case 不再出現 15-16。

### Phase 2 — 決定 N 的平衡點(實驗,非拍腦袋)
- N 太小 → turn 做不完、答案被截斷;N 太大 → 成本/爆走風險。
- 用 c1/c4 在不同 N 下跑,看 routing/完成度 vs call 數,挑平衡點。
- 驗收:N 有數據支撐,不再是「賭一個 4」。

### Phase 3 — 防呆(可選,但建議)
- chat model 設 `timeout`(避免靜默掛 ~10 分鐘)。
- eval loop 加進度輸出(讓「靜默」和「當掉」可區分)。
- eval 場景 bash 自動 deny(中和執行、保留違規計分;見 evaluator 既有 forbidden 設計)。

## 待決問題(需使用者/實驗回答)

1. 硬上限 scope:per-turn 還是 per-conversation?
2. N 的初始值與調法(Phase 2 實驗設計)。
3. 超過上限時的行為:目前是「改用不綁工具的模型強制 synthesize」,是否保留?還是給更明確的「列出缺什麼」指示?
4. 觸發層若證實是檢索品質差(非失憶),要不要連帶處理(這會繞回 c2/rag)。

## 受影響檔案(預估)
- `agent/graph.py`(agent_node 預算邏輯)
- `agent/history.py`(`prepare_messages_for_agent` 視窗)
- `agent/config.py`(`agent_max_tool_interactions` 語意/值)
- `agent/evaluation/claims/c1_routing.py`(Phase 0 instrument:存 args/results)
