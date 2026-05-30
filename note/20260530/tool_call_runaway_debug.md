# 偵錯紀錄:Agent 工具呼叫爆走(tool-call runaway)

日期:2026-05-30
相關 run:`c1-20260530T080146Z-c36d63e0`、`c1-20260530T083927Z-dda5336a`、`c1-20260530T095849Z-2f9cbf36`

## 症狀

做 evaluator survey 時跑 c1(工具路由評測),發現某些 case 對一個簡單問題叫了 **15-16 次工具**。
換不同模型(deepseek-v4-pro、先前的 GLM5)都會,簡單問題照樣叫十幾次工具 → 懷疑是 harness/context 層的問題,不是模型。

## 偵錯過程(含走過的死路,留作教訓)

過程中有幾個一開始的判斷是錯的,後來被證據推翻。記下來,因為這些「排除掉的假設」本身就是結論的一部分。

### 死路 1:以為是「選錯工具、偏好 grep」
- 初判:agent 面對知識庫問題不用 RAG,跑去 bash grep。
- 推翻:比對兩次完成的 trace,agent **每次都 RAG 優先**,狂試 RAG 12-13 次,bash/read_file 是最後迫不得已的逃生。
- 之所以會誤判,是因為手動跑時 **rag_search 不會跳核准提示**,畫面上只看得到最後爆出來的 bash,前面一大串 RAG 是靜默的。

### 死路 2:以為程式卡在「等使用者核准 bash」
- 初判:eval 批次跑卻停在 bash 的 y/N 核准。
- 推翻:使用者其實一直在按 y/Enter,真正的凍結發生在**按完最後一個 bash 之後、程式一片靜默**。
- 真相:卡在**靜默等待 reasoning model 的下一個回合**。主模型 deepseek-v4-pro 是推理模型(本來就慢),且 `ChatOpenAI` **沒設 request timeout**(預設約 10 分鐘),又有 `max_retries=10` 靜默重試,而 eval 跑單 case 時 console 完全不印進度 → 「正在等模型」跟「當掉」無法區分。

### 死路 3(環境陷阱):沙箱假警報
- coding agent 在沙箱裡跑指令時,網路白名單不含 `localhost:11434`,所以探測 Ollama 會誤報「沒開」;沙箱又用 `/dev/null` 覆蓋家目錄 dotfile,導致 `git status` 誤列一堆不存在的 dotfile。
- 教訓:**以使用者自己終端機的結果為準**,agent 沙箱裡的探測不算數。

### 正路:harness survey
讀 graph/context 程式碼,找到系統有**兩層獨立的歷史裁切**:
- **Layer 1(turn 級,如設計)**:`recent_turns` 保留最近 `agent_recent_turns_window=10` 個 turn,更舊的踢進 Chroma(`recall_history` 取回)。prior turn 以乾淨 `[Human, AI(answer)]` 保存。
- **Layer 2(turn 內 tool 結果級,bug 所在)**:`prepare_messages_for_agent` 每個 agent step 只在 prompt 保留**最近 `agent_max_tool_interactions=4` 次 tool 結果**,更早的裁掉。

同時 `graph.py` 有個 per-turn 工具預算(同樣是 4),但實測 15-16 次遠超 `4×2 turn=8`,代表預算沒咬住。

## Phase 0 instrumentation(用證據,不盲修)

在 `c1_routing.py` 用 `progress_cb` 記錄每個 graph step 的 emitted tool_calls 數 + tool 結果預覽,case detail 加 `actual_args`、`max_emitted_per_step`、`step_log`(不改計分)。跑一次(run `...095849Z`)後得到三個決定性證據:

### 證據 1 — 預算為何沒咬住:(a) parallel 溢出
`max_emitted_per_step = 2` → 模型一個 AIMessage 塞 2 個 tool_call。預算在回合「之前」檢查,回合內一次發 2 個就衝破。**不是計數沒登錄**(否則會跑到 recursion_limit=32,但它停在 9)。配上每個 turn 重置一次預算,embedding case(2 turn)這次 n=9,先前 15-16 是同機制的高變異。

### 證據 2 — query 序列露出「失憶」指紋
embedding case 9 次 call 的 query:第 5 次「embedding module」幾乎是第 2 次的重做、第 6 次又 explore 一次(第 1 次已做過)。在繞圈、重發已試過的查詢。對照 scoring case(passed,n=4)的 query 是漸進、不重複的,找到就收手。

### 證據 3(關鍵)— 檢索回的全是垃圾,因為答案不在語料庫
撈每次 ToolMessage 的結果預覽,embedding case 七次檢索回的是:`JmolWidgetset cache.html`(GWT 亂碼)、`3dmol.d.ts`(前端型別)、`MASTER_PLAN.md`、`poetry.toml`、`SaveToLink.java`——**零個跟 embedding 模組相關**,而且第 5=第 3、第 8=第 2 是重複的同一筆。

原因:`rag.cli.ingest` **會自動跳過 this workspace**(agent/rag 框架自己的 code),而「embedding 模組」(bge-m3 / OllamaEmbedder)正住在這個被排除的 workspace 裡。**所以這個查詢的目標內容根本不在索引裡,檢索注定回 PiDNA2 語料裡的垃圾。**

## Root cause(三層,主因被重新定性)

1. **主因 — 語料缺口 / 不可答查詢**:目標內容被 ingest 排除在索引外,檢索注定回垃圾,沒有滿意結果可收手。
2. **次因 — agent 沒有「放棄紀律」**:system prompt 說「1-3 次後 synthesize、不要瞎搜、不要捏造」,但沒說「搜了幾次都是垃圾就該下結論:KB 裡沒有」。它不肯認輸,一直換句話重搜。
3. **共犯 — Context 失憶(真的存在)**:Layer 2 把超過 4 筆的早期 tool 結果藏起來,模型看不到自己已經拿過這些垃圾(重複的 #5=#3、#8=#2 為證),所以重發。若看得到,至少會更快放棄。
4. **放大器 — 預算溢出(瑕疵 B)**:parallel 呼叫衝破 per-turn 預算 + 每 turn 重置,讓「找不到」被放大成 9-16 次。

## 附帶發現

- **評測題目可能設錯**:c1 embedding case 的 gold 期望 `rag_search` + `rag_get_context`,但目標內容不在索引裡,**根本不可能撈到好 hit 去 get_context** → 這題對這份語料近乎不可通過。應改寫,或重新分類成「優雅放棄」測試。
- **缺 request timeout**:主 chat model 沒設 timeout,stalled 請求會靜默掛約 10 分鐘。
- **eval 無進度輸出**:跑單 case 時 console 全靜默,無法區分「在等模型」和「當掉」。

## 這次改了什麼 / 接下來

- 已做:Phase 0 instrumentation(`c1_routing.py`,commit `72ff55f`)。
- 待改寫 fix_plan:優先序要調整——
  - 新增高槓桿項:**「放棄紀律」**(連續 N 次檢索不相關 → 結論 KB 沒有,停)、**修評測題**。
  - 原本的「對齊 cap/window」降為**止血**(止住次數,但單獨解不掉「答案不在語料裡」)。
  - 失憶修法變輔助(讓它放棄更快)。
  - 防呆:chat model 加 timeout、eval 加進度輸出、eval bash 自動 deny。
