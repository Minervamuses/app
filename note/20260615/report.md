# 2026-06-15 — 最近一週修正與 dev 評測報告

## 一句話

最近一週主要在收斂 agent 的工具使用規則、LLM 存取介面、skill 狀態序列化與 eval 基礎建設；今天跑完 C1–C4 dev 全量評測後，整體比 5/31 更完整，尤其 C1 已能不跳過 web 題、C4 的答案內容也更準，但還有三個明顯缺口：C1 graceful give-up 還不夠乾淨、C3 reviewer 仍會誤擋乾淨稿、C4 history 題仍沒有照 rubric 呼叫 `recall_history`。

## 我讀了哪些東西

- Git commit：`2026-06-14` 到 `2026-06-15` 的一週內提交，共 28 個 commit。
- Eval ledger：`eval/runs/c1.jsonl`、`c2.jsonl`、`c3.jsonl`、`c4.jsonl`。
- 今日明細：`eval/runs/details/*20260615*.json` 四份 detail 檔。
- 對照基線：ledger 裡 2026-05-31 的 dev run。

## 最近一週改了什麼

### 1. 工具清單與工具選擇規則

- 把 base tool inventory 收斂到 `agent/tools/inventory.py`，讓 graph 綁工具、session prompt、skill policy、eval scoring 都讀同一份工具清單。
- 修掉 tool availability fallback 來源漂移問題，fallback 現在從 base inventory 推導。
- 在 base workflow prompt 裡新增 graceful give-up 規則：搜尋結果空、重複或無關時，最多 1–3 次 `rag_search` 後要停，不要對無關結果叫 `rag_get_context`，也不要硬編答案。
- 補了跨多輪平行 tool call 的 cap 測試，確認 `agent_max_tool_interactions` 不會被多輪工具呼叫繞過。

### 2. Skill / state / metadata

- skill frontmatter 改用 PyYAML 解析，取代脆弱的手寫 parsing。
- active skill state 的序列化集中到 `agent/state.py`，graph 和 session 不再各自維護一份相似邏輯。
- 加了 history recall 舊失敗情境的 regression test，避免歷史查詢相關行為回退。

### 3. LLM 存取與 retry

- OpenRouter retry 改交給官方 OpenAI / LangChain client 的 `max_retries`，移除本地手寫 sleep/backoff loop。
- 新增 `AgentConfig.llm_max_retries` 作為 retry 次數的單一設定來源，thinking role models 也會正確吃到這個值。
- `agent.llm` 改成統一回傳 LangChain chat model factory，刪掉舊的 `BaseLLM` / `OpenRouterLLM` 風格抽象，新增 `invoke_text()` 作為 prompt-to-text helper。

### 4. Eval 基礎建設

- C1 embedding 題從「應該叫 `rag_get_context`」重新定義為「應該 bounded search 後誠實說知識庫沒有足夠資訊」，也就是 `rag_graceful_give_up`。
- C1 evaluator 新增 `expected_answer_regex` / `answer_accuracy`，讓 graceful give-up 不只看工具，也看最後回答是不是明確 not-found。
- C1 CLI 加上 progress callback，慢模型或 RAG 呼叫時可以看到 case / turn / tool-call / tool-result 進度。
- 新增 `scripts/run_full_eval.sh`，一次跑 C1–C4 dev，且正式 run 前會拒絕 dirty worktree；後續修成 ledger-only，不再額外產生旁路輸出。

## 今天的評測結果

今天的四個 run 都是 dev split，時間是 UTC 06:49–06:50，換算台北時間約 14:49–14:50。

| Claim | 主題 | 今日結果 | 重點 |
| --- | --- | --- | --- |
| C1 | 工具路由 | 7 / 8 通過，`routing_accuracy=0.875` | 全 8 題都評到、沒有 skip；只剩 graceful give-up 題失敗 |
| C2 | 檢索排序 | `recall@5=1.0`、`MRR=0.833`、`nDCG@5=0.877` | 和 5/31 一樣，正解都進前 5，但有一題排第 2 |
| C3 | validator / reviewer / session | validator 與 session 滿分，reviewer 一題誤擋 | 和 5/31 一樣，弱點仍是 reviewer 過度保守 |
| C4 | 端到端 checklist | `task_success_rate=0.5`、`answer_requirements_accuracy=1.0` | 答案內容都符合，但 history 題沒叫 required tool |

### C1 Routing

本次 C1 是正式全量 dev run：`eligible=8`、`evaluated=8`、`skipped=0`、`baseline_eligible=true`。這點比 5/31 的 `allow_skips=true` run 更可用，因為 web 工具三題這次都真的跑進來了。

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
- 好消息：第一個工具、工具家族、forbidden tools、coverage 都過了；沒有再碰 `bash`，也沒有對無關結果叫 `rag_get_context`。
- 失敗點：`count_ok=false`，因為 rubric 要 1–3 次工具，它叫了 4 次；`answer_ok=false`，因為最後回答雖然語意上接近「找不到 embedding module」，但沒有命中 expected not-found regex。
- 解讀：5/31 的問題是爆走與用錯工具；今天已經收斂很多，但 graceful give-up 的停止條件和答案格式還要再壓穩。

### C2 Retrieval

C2 結果和 5/31 完全一致。

- `c2-score-java-wrapper`：正解 `Score.java` 排第 1。
- `c2-score-container-columns`：正解 `ScoreContainer.java` 排第 1。
- `c2-pidna-readme-web-interface`：正解 `pidna2/README.md` 有進前 5，但排第 2，第一名是 `web/backend/pyproject.toml`。

結論：目前檢索「找得到」沒問題，`recall@5` 是滿分；下一步若要提升 C2，重點是排序品質，不是召回。

### C3 Validator / Reviewer / Session

C3 也和 5/31 一樣。

- Validator：3 / 3 正確，`violation_f1=1.0`，false positive rate 是 0。
- Session：2 / 2 正確，normal 與 extended 的 validation retry 都有觸發，最終答案乾淨。
- Reviewer：3 題中 2 題正確，`decision_macro_f1=0.556`、`route_macro_f1=0.556`。

Reviewer 的固定失敗點仍是 `c3b-clean-draft-pass`：gold 是 `pass`，但 prediction 是 `block`，failure mode 是 `user_input_missing`，route 變成 `ask_user`。也就是 reviewer 對乾淨稿過度挑剔，把本來該放行的草稿誤判成需要使用者補資料。

### C4 Checklist

C4 兩題一過一敗，總體 `task_success_rate=0.5`。

- `c4-local-file-summary` 通過：有叫 `read_file`，答案包含 Phase 1/2/3 與 C1–C4，也沒有碰 forbidden web tools。
- `c4-history-codename-answer` 失敗：答案有 `Blue Lantern`，也命中 `codename|deployment`，所以 answer requirements 已通過；唯一失敗是 `required_tools_ok=false`，沒有呼叫 `recall_history`。

和 5/31 相比，C4 的答案內容已有改善：`answer_requirements_accuracy` 從 0.5 變成 1.0。不過 task success 還是 0.5，因為 checklist 是「答案內容 + 必叫工具」都要過。

這題還有一個設計問題值得確認：dataset 的兩輪訊息是「先叫 agent 記住 Blue Lantern，再問剛才的 deployment codename」。目前 agent 直接從可見對話脈絡回答；但 rubric 要它呼叫 `recall_history`。如果 C4 想測的是 persisted history，就要讓測試情境更像「先前對話已不可見」；如果想測的是當前上下文，required tool 可能應該放寬。

## 目前判斷

- 工具 inventory 和 LLM access 這兩塊已經比一週前穩很多：重複定義少了，retry 行為也收斂到 config。
- C1 embedding 問題已從「爆走、亂叫工具」降級成「多叫一次工具、答案文字沒完全符合 rubric」。這是進展，但還不是解完。
- C2 沒退步，檢索召回穩定；排序仍是下一個改善點。
- C3 reviewer 的 clean draft false block 是舊問題，今天沒有改善。
- C4 的答案生成變好了，但工具路由仍沒符合 checklist；而且這題可能需要先釐清 rubric 是要測 history tool，還是測可見上下文回答。

## 建議下一步

- 先修 C1 graceful give-up：工具呼叫超過 3 次時強制停，並把 not-found 回答模板調到能穩定命中 rubric。
- 再修 C4 history 題：決定是改 agent 讓 C4 場景必走 `recall_history`，還是調整 dataset/rubric，避免和「可見上下文不用 history tool」的現行政策打架。
- C3 reviewer 針對 clean draft pass 加回歸測試或 prompt/rubric 調整，降低誤擋。
- C2 若要進一步拉分，新增排序診斷，專門看為什麼 `pidna2/README.md` 被 backend `pyproject.toml` 壓過。
