# 2026-05-31 — 評估 claim 套件跑測記錄

## 一句話
今天 dev 模式下把四個 claim 套件（C1 路由、C2 檢索、C3 審稿/驗證、C4 端到端清單）各跑了一次；C2、C3 主體都不錯，C1 與 C4 各有一題沒過，問題剛好都出在「該叫的工具沒叫」。

## 這次怎麼跑的
- 全部是 dev split、`baseline_eligible: false` 的開發測試，結果連續寫在今天 UTC 14:02~14:03。
- 檔案位置：每個套件一個摘要 `eval/runs/c{1-4}.jsonl`，逐題明細在 `eval/runs/details/`。
- C1 的 web 工具是**故意不載入**的，所以那三題被標 skip，不算失敗。

## 總成績一覽
| 套件 | 主題 | 結果 | 關鍵分數 |
|------|------|------|----------|
| C1 Routing | 該用哪個工具 | 4 過 / 1 失敗 / 3 略過（共 8） | routing 0.80、tool_count 0.80、tools_coverage 0.75 |
| C2 Retrieval | 純檢索品質 | 3 題全有命中 | recall@5 = 1.0、MRR 0.83、nDCG@5 0.88 |
| C3（三個子項） | 驗證 / 審稿 / 重試 | 大多正確，審稿錯 1 題 | 見下方 |
| C4 Checklist | 端到端任務清單 | 1 過 / 1 失敗（共 2） | task_success 0.50 |

## 各套件重點

### C1 Routing（8 題：4 過、1 失敗、3 略過）
- 過的：列知識庫分類（rag_explore）、scoring 模組怎麼運作（rag_search）、回憶部署代號（recall_history）、單純道謝不動工具（no_tool）。
- **失敗：`rag_context_embedding_followup`**（「How does the embedding module work?」）
  - 連續呼叫 8 次工具，rag_search / rag_explore 來回換，**就是沒切去用該用的 `rag_get_context`**。
  - 因此 `tool_count`（叫太多次）和 `tools_coverage`（該用的沒用到）兩項拉低分數。
  - 一句話：這題「搜尋上癮」，繞圈而沒拿上下文。
- 略過的三題全是 web 工具相關（web 摘要、完整 web 搜尋、讀單一網頁），因為這次沒載入 web 工具。

### C2 Retrieval（3 題檢索，recall@5 全中）
這套不經過 agent，直接測檢索引擎的排序好壞。
- `c2-score-java-wrapper`、`c2-score-container-columns`：目標檔都排第 1，MRR / nDCG 都 1.0。
- `c2-pidna-readme-web-interface`（「PiDNA2 web interface three step wizard」）：目標 `pidna2/README.md` 有進前 5，但被 `web/backend/pyproject.toml` 擠到第 2 名，MRR 掉到 0.5。
- 結論：**該找的都找得到（recall 滿分），但排序還能更準**，平均 MRR 0.83 就是被這題拉下來的。

### C3（8 題，分三個子項）
- **Validator（3/3 完美）**：violation_f1 1.0、誤判率 0。能抓到「沒附來源的百分比數字」要標違規，也能正確放行「已附引用的百分比」和「未註冊 skill 沒規則」兩個不該報的情況。
- **Reviewer（3 題對 2 題）**：把審稿當分類器測。
  - 對的：「沒嘗試檢索就給答案」判 revise/major；「工具被禁用」判 block/ask_user。
  - **錯的：`c3b-clean-draft-pass`** —— 一份乾淨、該直接 pass 的草稿，卻被誤判成 block，理由是「沒有提供要潤飾的句子（user_input_missing）」。也就是**對乾淨稿過度挑剔、誤擋**。
  - 影響：decision / route 的 macro-F1 掉到 0.56；但 severity 判斷仍 1.0。
- **Session（2/2）**：normal 與 extended 兩條路徑的 skill 驗證重試都如預期觸發，最終答案也都乾淨（retry 與 final-clean 皆 1.0）。

### C4 Checklist（2 題：1 過、1 失敗）
端到端任務，用確定性清單（必叫工具、必含關鍵字/檔名）評分。
- **過：`c4-local-file-summary`** —— 摘要 `EVALUATOR_PLAN.md` 的七個 Phase，正確用了 `read_file`，內容涵蓋 Phase 1/2/3 與 C1~C4，也沒去碰被禁的 web 工具。
- **失敗：`c4-history-codename-answer`** —— 答案內容其實對（講出「Blue Lantern」），但兩個地方扣分：
  1. **沒呼叫該用的 `recall_history`**，直接憑上下文回答（required_tools 不過）。
  2. 答案是中文「部署代號」，沒命中英文 regex `codename|deployment`（answer_regex 不過）。
  - 結果 task_success 只有 0.5。

## 今天的共同教訓
- **三個失敗有兩個是同一個毛病：該叫的工具沒叫。** C1 該用 `rag_get_context` 卻一直 search；C4 該用 `recall_history` 卻憑記憶答。要加強「了解某模組／回憶歷史」這類追問時的工具選擇。
- C3 的 Reviewer 有**過度挑剔**傾向，會把乾淨稿誤判成需要使用者補資料；要調整對「沒問題就放行」的判斷。
- C2 檢索 recall 已經滿分，下一步是**改善排序**（讓正解更常排第 1）。
- C4 的中文答案撞到英文 regex，這也提醒清單 rubric 的關鍵字檢查要考慮語言。

## 資料集題數盤點（不只今天跑的結果）
順手清點 `eval/datasets/` 裡每個 claim 實際有幾題。結論：**整體都偏薄，C2 和 C4 最少。**

| Claim | 主題 | dev | test | 狀況 |
|-------|------|-----|------|------|
| C1 | Routing 工具路由 | 8 | 8 | 最完整，但每個 category 只 1 題 |
| C2 | Retrieval 檢索 | 3 | 1 | ⚠️ 很少 |
| C3 | 驗證/審稿/重試 | 8 | 4 | 結構最均衡 |
| C4 | Checklist 端到端 | 2 | 1 | ⚠️ 最少 |

- 沒有 `manifest.json`，也沒有 `holdout.jsonl`；資料集只有 `dev.jsonl` + `test.jsonl` 兩個分割。
- **C4 最薄**：dev 2 題、test 1 題。dev 那 2 題還分屬完全不同類型（讀本地檔摘要 / 回憶歷史代號），等於每種任務只有 1 個樣本，沒有重複驗證餘地。
- **C2 次之**：dev 3 題、test 1 題。而且 test 那題（`c2-score-container-columns-frozen`）只是 dev 同一題的 frozen 翻版，獨立鑑別力幾乎為零。
- **C1 看似 8 題最多，但每個 category 剛好只有 1 題**（rag_explore / rag_search / rag_get_context / recall_history / 三個 web 工具 / no_tool）。這就是今天 embedding（rag_get_context 類）一失敗、該類通過率直接掉到 0% 的原因——單點失分沒有緩衝。
- **C3 相對健康**：dev 分 3a validator 3 題、3b reviewer 3 題、3c session 2 題；test 4 題各類都有。

共通問題：**每個子類別只有 1~3 個樣本**，任一題失敗就會讓該類別分數劇烈跳動，統計上不穩；C2/C4 的 test split 又大多是 dev 的凍結翻版，獨立性不足。

## 待辦
- 修 C1 embedding 題：追問模組細節時切去 `rag_get_context`，別狂 search。
- 修 C4 codename 題：強制走 `recall_history`；或放寬 regex 接受中文用語。
- 調 C3 Reviewer：降低對乾淨稿的誤擋。
- 之後載入 web 工具補跑 C1 那三題，並換真實 LLM 再跑一輪對照。
- **補題**：C4（最優先，每類任務只有 1 題）、C2 各補幾題；C1 每個 category 至少加到 2~3 題，讓分數不會被單題左右。
