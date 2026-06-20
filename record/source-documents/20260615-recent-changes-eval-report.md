# 2026-06-15 — 近期變更與 dev 評測報告

## 摘要

6/14–6/15 的主要工作集中在三條線：第一，整理 agent 的基礎能力邊界，讓工具清單、工具可用性、skill 狀態與 LLM 存取方式都有單一來源；第二，把 C1 的 embedding 失敗案例重新定義成「找不到資料時要優雅放棄」，並補上可評分的答案規格；第三，建立可以一次跑完 C1–C4 的 dev 評測流程，並記錄今天的全量結果。

今天 C1–C4 dev 都已跑完。C1 這次 8 題全評、沒有 skip，7 題通過；C2 和 C3 分數與 5/31 相同；C4 的答案內容分數改善到滿分，但 history 題仍因沒有呼叫 `recall_history` 而失敗。

## 近期變更

### 1. 任務與文件整理

`to_be_solved/` 被整理成目前仍待解的問題清單與 archive，舊的 `agent.md` 也改名成 `AGENTS.md`。完成的問題卡陸續移除或標記關閉，包括：

- `frontmatter-parser-pyyaml`
- `base-tool-inventory-single-source`
- `agent-history-tool-availability`
- `agent-history-recall-user-facing-failure`
- `openrouter-retry-cleanup`
- `agent-tool-call-runaway-followups`
- `llm-access-contract`

這批整理讓最近的修正線索比較清楚：工具 inventory、history recall、tool-call runaway、OpenRouter retry、LLM access contract 都已經有對應修正與測試，不再只是待辦描述。

### 2. Skill metadata 改用 PyYAML

skill frontmatter 的解析改成使用 PyYAML，而不是手寫字串 parsing。這修掉了 YAML frontmatter 在格式稍微複雜時容易解析錯的問題，也讓 skill metadata 的處理方式更接近標準 YAML 行為。

相關變更：

- `agent/skills/metadata.py`
- `tests/test_skills.py`

### 3. Base tool inventory 改成單一來源

原本 base tools 的定義散在 graph 建構、session system prompt、skill policy、eval scoring 等位置，容易出現「prompt 說有工具，但 runtime 沒綁」或「eval 用另一套工具宇宙評分」的漂移。現在改成由 `agent/tools/inventory.py` 統一管理。

新的 inventory 負責：

- 宣告 base tools 的靜態 metadata，包括 `rag_explore`、`rag_search`、`rag_get_context`、`recall_history`、`read_file`、`bash`。
- 建立 graph 實際綁定的 tool instances。
- 提供 `base_tool_names()`、`behavior_tool_names()` 等 ordered tool name list。
- 產生 system prompt 裡的工具說明、工具選擇政策與基本 workflow。
- 讓 evaluation 的 tool taxonomy 與 runtime 使用同一份工具清單。

相關變更：

- `agent/tools/inventory.py` 新增
- `agent/graph.py` 改由 inventory 建立 base tools
- `agent/session.py` 的 system prompt 改嵌入 inventory render 出來的工具說明
- `agent/evaluation/base.py`、`agent/evaluation/metrics/tool_routing.py` 改從 inventory 取得工具分類
- `tests/test_tool_inventory.py` 新增大量覆蓋

這是近期最重要的結構性整理之一，後續工具行為與 eval rubric 比較不容易各走各的版本。

### 4. Tool availability fallback 與 history 能力修正

extended thinking 的 writer / rewriter / reviewer 在沒有傳入 `base_tool_names` 時，原本可能 render 出空的 available tools，導致角色以為 base tools 都不可用。現在 `render_tool_availability_block()` 在參數是 `None` 時會 fallback 到 base inventory；但明確傳入空 list 仍代表「沒有工具」。

同時，`capability_map.yaml` 補上 `rag.search` 與 `history.search` 的差異說明。這點很重要，因為 `recall_history` 查的是持久化聊天歷史，不是 indexed knowledge base；兩者不能混成同一種檢索能力。

相關測試覆蓋：

- `tests/test_skill_runtime.py`
- `tests/test_thinking.py`
- `tests/test_academic_skill_tool_policy.py`

### 5. History recall 舊失敗情境被固定成 regression test

新增 `tests/test_history_recall_scenario.py`，把 archive 裡的 history recall 失敗案例轉成可重跑的 deterministic regression。測試重點有三個：

- 當 `recall_history` 可用但 writer 沒有先查時，reviewer 會標出 `retrieval_not_attempted`，並導向 reviser 補查。
- 當 `recall_history` 查不到資料時，答案要誠實說歷史不足，且要區分 chat history、plan logs、indexed KB。
- 當 active skill 禁用 history tool 時，使用者看到的是 tool policy/settings 的說明，而不是內部 reviewer 指令或 intake checklist。

這表示 history recall 這條線已經從「人工觀察到的失敗」變成「可回歸驗證的測試」。

### 6. Skill state serialization 集中到 `agent/state.py`

active skill runtime 轉成 LangGraph agent state 的邏輯原本在 `graph.py` 和 `session.py` 各有一份。現在集中成 `skill_runtime_to_agent_state()`。

集中後的好處：

- active skill name、root、instructions、loaded references、task mode、tool policy 都由同一段程式序列化。
- validation 狀態的初始值也統一設定。
- graph 與 session 不再各自維護一份容易漂移的 state shape。

相關變更：

- `agent/state.py`
- `agent/graph.py`
- `agent/session.py`
- `tests/test_state.py`
- `tests/test_skill_adherence.py`

### 7. OpenRouter retry 改交給官方 client

OpenRouter 呼叫原本有本地手寫 retry/backoff loop。現在改成把 retry 交給官方 OpenAI / LangChain client 的 `max_retries`。

具體變更：

- 新增 `AgentConfig.llm_max_retries`，作為 retry 次數的單一設定來源。
- `get_chat_model()` 會把 `llm_max_retries` 傳給 `ChatOpenAI`。
- OpenRouter prompt-to-text 呼叫也改成依賴 client 的 retry，不再維護 `_call_with_retry`、`MAX_RETRIES`、`INITIAL_DELAY`。
- thinking role models 也補上 `llm_max_retries`，避免 extended thinking 角色還在用硬編碼 retry 值。

相關測試：

- `tests/test_openrouter_model.py`
- `tests/test_thinking_models.py`

### 8. LLM access contract 標準化

`agent.llm` 從舊的 `BaseLLM` / provider class 風格，改成更一致的 LangChain chat model factory。

主要變更：

- 刪除 `agent/llm/base.py`。
- `agent/llm/openrouter.py` 提供 `get_openrouter_chat_model()` 與 `get_chat_model()`。
- `agent/llm/ollama.py` 提供 `get_ollama_chat_model()`。
- 新增 `agent/llm/text.py` 的 `invoke_text()`，用來把單一 prompt 丟給 chat model 並取回文字。
- e2e eval、OpenRouter tests、thinking model tests 都跟著調整。

這讓主 agent loop、eval、thinking roles 在模型建構方式上更一致，也減少「同樣是 LLM 呼叫但走不同抽象」的維護成本。

### 9. Tool-call runaway 與 graceful give-up

C1 embedding 題原本被當成 `rag_get_context` 類型：問 embedding module，再要求看最相關結果的上下文。但實際問題是 indexed KB 裡沒有 embedding module 的資料；正確行為不該是一直 search 或對無關結果叫 context，而應該是 bounded search 後誠實說資料不足。

因此這題被重新分類成 `rag_graceful_give_up`：

- 單輪問題：「How does the embedding module work?」
- 允許第一個工具是 `rag_search` 或 `rag_explore`
- 必須至少使用 `rag_search`
- 禁止 `rag_get_context`、history、web、file、bash
- 工具次數限制為 1–3
- 最終答案必須同時提到 KB/indexed knowledge base，以及 not-found / insufficient evidence 之類的訊號

同時 base workflow prompt 補上 give-up discipline：搜尋結果空、重複或無關時，不要無限搜尋，不要對無關結果叫 `rag_get_context`，也不要編答案。

### 10. C1 與 BehaviorEvaluator 規格同步

C1 frozen dataset 已改成 graceful give-up，但 legacy `BehaviorEvaluator` 一開始仍保留舊的 `rag_get_context` 期待，造成同一個 case id 在兩套 eval 裡語意不同。

近期修正把 BehaviorEvaluator 也同步成新規格，並新增 semantic parity test，檢查 C1 frozen case 與 legacy behavior case 的 category、question sequence、gold expectations 是否一致。這避免未來只靠 id 對齊、但實際 rubric 已漂移。

相關變更：

- `agent/evaluation/behavior.py`
- `tests/test_behavior_eval.py`
- `tests/test_c1_routing_eval.py`

### 11. C1 evaluator 新增答案評分與 live progress

C1 evaluator 現在支援 `expected_answer_regex`。對有這個欄位的 case，除了工具路由外，也會檢查 final answer 是否符合預期文字訊號，並輸出 `answer_accuracy`。

另外，C1 evaluator 與 CLI 加上 progress callback。跑慢模型或 RAG 時，會輸出 case start、turn start、tool call、tool result、case done 等進度，方便分辨是真的卡住，還是在等待外部呼叫。

相關變更：

- `agent/evaluation/claims/c1_routing.py`
- `agent/evaluation/metrics/tool_routing.py`
- `agent/cli/eval.py`
- `tests/test_c1_routing_eval.py`
- `tests/test_eval_runtime.py`

### 12. Tool-call cap 的測試與設定說明

新增測試確認多輪平行 tool calls 不會突破 `agent_max_tool_interactions`。測試模型會每輪都要求多個 parallel `rag_search`，驗證 graph 仍會在 cap 用完後停止工具互動並強制產生 final answer。

`AgentConfig` 裡也補上 `agent_max_tool_interactions=4` 的設定說明：這個預設是根據 C1 dev routing run 得到的資料，正常 eligible cases 都落在 0–4 次工具呼叫內；embedding runaway 是因為缺少 give-up discipline，不是因為 cap 太小。

### 13. Full eval runner 與今日 run 記錄

新增 `scripts/run_full_eval.sh`，用來一次跑完 C1、C2、C3、C4。

runner 行為：

- 預設跑 dev split。
- 正式跑前拒絕 dirty worktree。
- 可用 `ALLOW_SKIPS=1` 控制 C1 是否允許 skip。
- 可用 `NO_MCP=1` 關閉 MCP。
- 預設只寫入既有 ledger，不再產生額外旁路輸出。
- 可選擇額外跑 legacy suites。

今天的 C1–C4 dev run 已寫入：

- `eval/runs/c1.jsonl`
- `eval/runs/c2.jsonl`
- `eval/runs/c3.jsonl`
- `eval/runs/c4.jsonl`
- `eval/runs/details/c1-20260615T064950Z-6d27a65d.json`
- `eval/runs/details/c2-20260615T064955Z-ef938234.json`
- `eval/runs/details/c3-20260615T065007Z-c039bf54.json`
- `eval/runs/details/c4-20260615T065042Z-49804861.json`

## 今日 dev 評測結果

四個 run 都是 dev split，時間是 UTC 06:49–06:50，台北時間約 14:49–14:50。

| Claim | 主題 | 今日結果 | 和 5/31 相比 |
| --- | --- | --- | --- |
| C1 | 工具路由 | 7 / 8 通過，`routing_accuracy=0.875` | 5/31 有 3 題 web skip；今天 8 題全評 |
| C2 | 檢索排序 | `recall@5=1.0`、`MRR=0.833`、`nDCG@5=0.877` | 分數相同 |
| C3 | validator / reviewer / session | validator 與 session 滿分，reviewer 一題誤擋 | 分數相同 |
| C4 | 端到端 checklist | `task_success_rate=0.5`、`answer_requirements_accuracy=1.0` | answer requirements 從 0.5 升到 1.0 |

### C1 Routing

本次 C1 是全量 dev run：`eligible=8`、`evaluated=8`、`skipped=0`、`baseline_eligible=true`。這比 5/31 的 `allow_skips=true` run 更適合作為正式對照，因為三個 web 題也實際跑進來了。

通過的題目：

- `rag_explore_inventory`
- `rag_search_scoring`
- `history_codename`
- `web_summary_openai_models`
- `web_full_langgraph_docs`
- `web_single_example`
- `no_tool_thanks`

唯一失敗是 `rag_context_embedding_followup`，現在分類是 `rag_graceful_give_up`。

- 實際工具：`rag_search`、`rag_explore`、`rag_search`、`rag_search`，共 4 次。
- 已改善：第一個工具、工具家族、forbidden tools、coverage 都過；沒有再碰 `bash`，也沒有對無關結果叫 `rag_get_context`。
- 仍失敗：rubric 要 1–3 次工具，但實際 4 次，所以 `count_ok=false`；最後回答語意上接近「找不到 embedding module」，但沒有命中 expected not-found regex，所以 `answer_ok=false`。

判斷：C1 embedding 問題已從 5/31 的 runaway / wrong tool 降級成停止條件與答案格式問題，仍需要修。

### C2 Retrieval

C2 結果和 5/31 完全一致。

- `c2-score-java-wrapper`：正解 `Score.java` 排第 1。
- `c2-score-container-columns`：正解 `ScoreContainer.java` 排第 1。
- `c2-pidna-readme-web-interface`：正解 `pidna2/README.md` 有進前 5，但排第 2；第 1 名是 `web/backend/pyproject.toml`。

判斷：召回沒有問題，排序仍可改善。若要拉高 C2，重點不是讓正解進前 5，而是讓正解更常排第 1。

### C3 Validator / Reviewer / Session

C3 和 5/31 一樣。

- Validator：3 / 3 正確，`violation_f1=1.0`，false positive rate 是 0。
- Session：2 / 2 正確，normal 與 extended 的 validation retry 都有觸發，最終答案乾淨。
- Reviewer：3 題中 2 題正確，`decision_macro_f1=0.556`、`route_macro_f1=0.556`。

固定失敗點仍是 `c3b-clean-draft-pass`：gold 是 `pass`，prediction 卻是 `block`，failure mode 是 `user_input_missing`，route 變成 `ask_user`。也就是 reviewer 對乾淨稿過度保守，把本來該放行的草稿誤判成需要補資料。

### C4 Checklist

C4 兩題一過一敗，`task_success_rate=0.5`。

- `c4-local-file-summary` 通過：有叫 `read_file`，答案包含 Phase 1/2/3 與 C1–C4，也沒有碰 forbidden web tools。
- `c4-history-codename-answer` 失敗：答案有 `Blue Lantern`，也命中 `codename|deployment`，所以 answer requirements 已通過；唯一失敗是沒有呼叫 required tool `recall_history`。

和 5/31 相比，C4 的答案內容已改善，`answer_requirements_accuracy` 從 0.5 升到 1.0。不過 task success 仍是 0.5，因為 checklist 要求「答案內容」和「必叫工具」都通過。

這題也暴露出 rubric 設計問題：dataset 先給「Remember that the deployment codename is Blue Lantern.」，下一輪立刻問剛才的 codename。現行工具政策寫明「當內容已在目前對話可見時，不要呼叫 `recall_history`」；但 C4 rubric 又要求一定要叫 `recall_history`。因此這題需要先決定評測目標是「測 persisted history」還是「測 visible context」。如果是前者，dataset 應該把記憶內容放到 setup history 或不可見歷史；如果是後者，required tool 應該放寬。

## 結論

- 工具系統已完成一輪重要整理：base tool inventory、tool availability fallback、tool taxonomy、prompt tool descriptions 現在較一致。
- History recall 的舊失敗情境已被固定成 regression test，後續比較不容易退回「沒查歷史就要求使用者補資料」的問題。
- LLM access 與 retry 已經比之前乾淨：retry 有單一 config，OpenRouter/Ollama 都走 LangChain chat model factory。
- C1 graceful give-up 的規格已建立，但實作行為還沒有完全達標。
- C2 召回穩定，下一個瓶頸是排序。
- C3 reviewer 仍有 false block。
- C4 的內容回答變好，但 history 工具 rubric 與現行工具政策互相衝突。

## 建議下一步

- 先處理 C1 graceful give-up：工具呼叫到第 3 次仍無關時強制收斂，並讓 not-found 回答穩定包含 rubric 要的 KB / insufficient evidence 訊號。
- 釐清 C4 history 題的測試目標：若要測 persisted history，就改 dataset；若要測可見上下文，就改 required tool。
- 修 C3 reviewer 的 clean draft false block，補一個明確的 regression。
- 針對 C2 排序做診斷，查 `pidna2/README.md` 為何輸給 `web/backend/pyproject.toml`。
