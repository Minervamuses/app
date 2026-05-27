# 長期可維護 Evaluator — 實作計畫

## Context(為什麼做這個)

開發者(非 end user)在 `agent/`(LangGraph agent，消費 sibling repo `../rag`)上做了多輪大規模開發後，
意識到**現有評估已過時、不可信**：`agent/evaluation/` 三個 suite 誕生於數個大改之前，
上次 run 的結果 schema 無版本標記、behavior 只有 8 個 case(現 code 16 個)、e2e 72% 判全錯。

痛點(錨點，依與使用者三題確認)：
- **Q1 評什麼**：四個 claim 全要，且**可單獨跑、可一次全測**(開發者經 CLI / slash command 觸發)。
- **Q2 要得到什麼**：**可信的絕對數字** + **append-only、帶版本標記、不覆蓋**的結果檔；regression / 版本對比由開發者讀數字得出。
- **Q3 最怕什麼**：**自製評估跟主流學界/業界對不上 → 發 paper / 對外說明不具說服力**。
  → 決議：**主流方法論/指標跑在私有資料上 + 至少一個公開 benchmark 做外部對標**(兩者都要)。

硬約束：**核心數字必須來自確定性 code**；LLM/agent 可做 error analysis、建議，**不可自己給分**。

目標：一套**半年後還在用**的 evaluator，可重現、可比較版本、可偵測 regression、test set 不被自己無意 overfit。

---

## 四個 Claim

- **C1 工具路由正確性** — 面對問題選對工具家族並正確編排檢索。
- **C2 答案忠實度 / 檢索品質** — 跨多 chunk 取證、有根據、不幻覺。
- **C3 Skill 遵循 + extended-thinking 把關** — skill tool policy、不捏造、reviewer 攔截高風險草稿(含 normal 與 extended 兩路徑)。
- **C4 端到端任務完成** — 真實任務自主走完、產出可用結果(**獨立 rubric，不混進 C2**)。

---

## 架構決定：方案 A(純確定性 class harness)+ 預留 B(agent 探索層)接口

- 核心全為 deterministic class/函式；error analysis 先靠開發者讀明細。
- 在 results / dataset 的讀取介面上**預留 B 要用的 read API**(`read_run`, `read_details`, `read_dataset`)，
  未來疊 agent 探索層(做 error taxonomy / 建議 case / 建議 ablation，**永不算分**)為純疊加、零返工。

### Metric 分兩層(永遠分開報，不合成單一總分)

| 層 | 數字來源 | 代表 metric |
|---|---|---|
| **Tier 1 確定性(核心數字)** | 純 code：集合比對 / rank 計算 / regex / 給定 report 的路由 | retrieval `recall@k`,`MRR`,`nDCG@k`,`context-recall`；tool-routing accuracy；citation/groundedness 字串比對；skill-validator pass rate；reviewer 當分類器的 P/R/F1；public benchmark exact-match |
| **Tier 2 LLM-judged(次要)** | LLM-as-judge，**附 judge model/版本/prompt hash + 與人工標註子集的一致度(judge–human agreement)** | RAGAS 式 faithfulness / answer-relevancy；holistic 任務完成度 |

### 每個 Claim 的 metric 對應(誰是 system-under-test、誰是 deterministic scorer)

- **C1** → 全 Tier 1。prediction = `turn_with_trace` 的 tool trace;gold = 凍結 case 的期望工具集。
  scorer:`_score_tool_expectations` 目前是 `BehaviorEvaluator` 的 **instance method**(`behavior.py:337`),不是 free function →
  **先把純比對邏輯抽成 module-level pure function**(輸入 case+actual_tools/args,輸出 scores dict),C1 runner 與舊 `BehaviorEvaluator` 共用。
  **工具宇宙明確納入 `bash`/`read_file`**(現 `tool_inventory` 漏了),forbidden universe 一併涵蓋。
- **C2 檢索** → Tier 1(`recall@k/MRR/nDCG@k`，gold = 人工確認的 relevant `(pid,chunk_id)` 集，rank-based 因 `Hit.score` 恆為 None)。
  **C2 答案忠實度** → Tier 2。
- **C3** → 全 Tier 1，**拆成三個彼此獨立的子評估**(repo 裡這是三條不同路徑，不可混為一談)：
  - **C3a — validator 確定性檢查**：對標好的最終回答集，跑 `agent/skills/validator.py::validate_skill_output`，
    算 violation 命中/誤判率。純函式,無 LLM。
  - **C3b — reviewer 當分類器(P/R/F1)**：**reviewer LLM 是 system-under-test,scorer 確定性**。
    對人工標好正確 decision/severity/`failure_mode`/route 的草稿集,直呼 `agent/thinking.py::review_draft`
    取 `ReviewReport`,再用 `route_review_report` 導出 route,比對 gold。**不過 ChatSession。**
    **P/R/F1 計算法須固定(否則各算各的)**:把問題定義成**多個獨立二元判定**,各算 P/R/F1 後報 **macro 平均 + 每類細項**:
    (i) `decision`(pass/revise/block,multi-class → one-vs-rest);(ii) 每個 `failure_mode` label(positive = 該 mode 出現於任一 finding);
    (iii) `route`(pass/ask_user/stop/revise,one-vs-rest);(iv) `needs_user_input`(二元)。
    severity 用 `_SEVERITY_RANK` 門檻轉二元(`>= 標好的 min_severity` 視為 positive)。gold/positive 定義寫進 dataset schema doc,不留給實作者臆測。
  - **C3c — normal/extended ChatSession 整合**:走完整對話路徑,驗證 skill validation retry 真的觸發。
    **observability 注意**:`turn_with_trace` 的 `trace_events`/`turn_logs` **只記 tool call**(`session.py:572,638`),
    **skill validation retry 不是 tool call → 看不到**。故 C3c harness **不能只靠 `turn_with_trace`**,採既有測試證實可行的手法:
    - normal 路徑:**直接 `build_graph(cfg).invoke(state)` 後檢查 graph state 的 `validation_attempts` / model invoke 次數**
      (參 `tests/test_skill_validator.py::test_skill_validator_retries_once...`);或用 `progress_cb`(`session.py:201,570`)捕捉 `skill_validator` node 觸發。
    - extended 路徑:直接測 `agent/session.py::_apply_final_skill_validation`(參 `tests/test_thinking_session.py`)。
    - 註:`skill_validator_node` 是 `build_graph()` 內的 **nested function**,不可 module-level import → 只能經 graph/session integration 觀測。
    metric = 確定性檢查(該 retry 有沒有 retry、retry 次數、最終輸出是否仍違規)。
- **C4** → 獨立 rubric。核心 = **確定性 checklist**(該叫的工具有沒有叫、最終答案是否含必備事實/檔名，用 regex 或 embedding 門檻)= Tier 1；holistic 完成度 = Tier 2 選配。

---

## Test set 設計(直接打「可重現 / 不被自己 overfit」)

- **資料與 code 分離(釘死,消除歧義)**：
  - **凍結 JSONL 資料**放 **repo root** `eval/datasets/<claim>/{dev,test}.jsonl`，commit 進 git(跟 code 版本走)。
  - **loader / schema / provenance 驗證 code**放 **package** `agent/evaluation/datasets/`。
  - 兩者不要混:`agent/evaluation/datasets/` 裡**不放** JSONL。
- 每筆 JSONL 一行一物件:`id`(穩定)、`inputs`、`gold`、`provenance`(來源/標註者/日期)。**固定 schema 範例(C1)**:
  ```json
  {"id": "c1-rag_search_scoring", "claim": "c1", "split": "dev",
   "inputs": {"messages": ["How does the scoring module work?"]},
   "gold": {"expected_tool_family": "rag", "expected_first_tool_in": ["rag_search", "rag_explore"],
            "expected_tools_include": ["rag_search"], "expected_tool_count": {"min": 1, "max": 4}},
   "provenance": {"source": "migrated:behavior.py", "labeler": "gary", "date": "2026-05-27"}}
  ```
  C2 的 `gold` = `{"relevant": [{"pid": "...", "chunk_id": 0}, ...], "k": 10}`;
  C3b 的 `gold` = `{"decision": "revise", "min_severity": "major", "failure_mode": "retrieval_not_attempted", "route": "revise"}`。
- **dev/test 切分 + sealed test**：平時只在 **dev** 迭代；**test 封存**，只看 aggregate、不看 per-case，僅里程碑跑；
  紀錄 test 被跑次數(overfit 風險指標)。README 鐵律：**永不在 test 上調參**。
- **新 failure case 納入協定(sustained 必備)**：新失敗先進 **dev**(`provenance: regression/<date>`)，
  每個 release 批次 **promote 到 test**；test 只增不改不刪，保住歷史可比。

### C2 語料 = 固定 eval fixture(右尺寸,非 high risk)

`../rag/store/` 在開發期就是**借來的固定測試 fixture**(PiDNA2 + notes),開發者不會邊開發邊改,且 evaluator 不上線、無 user-facing store lifecycle。
→ 計畫**明確不支援**：兩次正式 eval 之間的 store 變動、migration、動態重建。store 對 evaluator 而言是唯讀常數。

1. gold 綁定該 store 的 `(pid,chunk_id)`(已驗:raw.json 3229 筆,每筆 metadata 有 `pid,chunk_id`)。
2. **fingerprint 必須蓋到「實際被檢索的 Chroma index」,不只 `raw.json`**。
   ⚠ C2 semantic retrieval 走 `rag.api.search()` → `ChromaStore` + `VectorRetriever`(`vector.py::retrieve` → `as_retriever().invoke()`),**查的是 Chroma collection,不是 `raw.json`**。
   所以 fingerprint 只 hash `raw.json` 不夠:raw.json 相同、但 Chroma index 壞掉/被重嵌/非同一份時,C2 數字仍會漂。fingerprint 分兩部分:
   - **(a) 內容指紋(robust)**:透過 Chroma collection 的 `get(include=["documents","metadatas","embeddings"])` 取出
     `ids + documents + metadatas(含 filter 用的 category/file_type/folder/date/tags)+ embeddings`,排序後算 hash。
     這直接涵蓋「驅動檢索的東西」,且避開 HNSW 二進位檔逐位元比對的脆弱性(同資料不保證 byte 相同)。
   - **(b) artifact 清單**:檢查 store 目錄存在且齊全(`chroma.sqlite3`、HNSW index 目錄、`raw.json`、`folder_meta.json`),作為 presence guard。
   - **原因(metadata 漂移)**:filter 吃 `category/file_type/folder/date`(`api.py::build_where`),而 `category/tags` 由 **LLM tagger 產生**
     (`ingest.py::_tag_folders` → `LLMTagger`,`temp=0` 仍非保證 bit-identical、換 model 會變)。(a) 已涵蓋此漂移。
   run 啟動時與凍結記錄比對;**不一致(含 rag sha / `embed_model` mismatch)直接報錯中止**,不自動修復。
3. **新環境如何取得同一份 fixture**(因 `../rag/.gitignore` 有 `/store/*`、`*.sqlite3`,store **未進版控**,clone 不會有):
   - **主路徑 = snapshot**:把**整個 store 目錄**(`chroma.sqlite3` + HNSW index 目錄(`<uuid>/*.bin`+`index_metadata.pickle`)+ `raw.json` + `folder_meta.json`)
     另存成 eval 快照(repo 外固定位置 / 外部儲存,實作期定),新環境直接還原。整包還原才保住實際被查的 Chroma index。**因為 tagger 非決定 → 重跑 ingest 無法保證重現同一 fixture。**
   - 備援(不建議當主路徑):pinned rag sha + `bge-m3` 重跑 `python -m rag.cli.ingest`,**且必須通過 fingerprint 才採用**;
     fingerprint 不過(極可能因 tagger metadata 變動)→ 退回 snapshot。
   - missing / fingerprint mismatch → **硬報錯並印出取得步驟**,不靜默繼續。
4. 真要刷新 fixture → 當成**刻意的版本升級**(新 snapshot + 新 fingerprint + 新 dataset 版本),歷史數字各自可比。

---

## 重現性(處理四個獨立隨機源)

1. 主 LLM `temperature=0.3`、無 seed(`agent/llm/openrouter.py::get_chat_model`) → eval 模式**強制 `temp=0`**(config override)，
   並試接 `extra_body={"seed":...}`；**承認 temp=0 仍非完全決定** → 支援 **n-sample 多跑，報 mean ± std**。
2. Ollama 嵌入 → pin `embed_model`，寫進 metadata。
3. case 生成的 Python `random`(`endtoend.py:141-144`，無 seed)→ 凍結資料集後不再即時生成；殘留隨機固定 seed。
4. judge LLM → pin model + prompt，記 hash。
5. **版本 metadata 必含**：`agent_git_sha` + `rag_git_sha` + dirty flag + dataset id/hash + store 指紋 + model ids + embed_model + seed + n_samples + timestamp。

---

## 結果儲存 + 版本對比

- **append-only ledger**：`eval/runs/<claim>.jsonl`，**一行一次 run，永不覆蓋**，含完整 metadata + Tier1/Tier2 分數。
- per-run 明細：`eval/runs/details/<run_id>.json`(每 case 的 prediction / gold / 通過與否)。
- **regression / 版本 diff = 薄確定性 helper**(非核心)：讀 ledger 兩行 → 印「哪些 case pass→fail」「每 metric delta」。

---

## 公開 benchmark 對標(Q3「兩者都要」)—— 定位為 **Spike,非 MVP 必成**

為避免 coding agent 卡在「先接哪個、做到哪算完成」,把外部對標**限縮成一個明確的 spike**:

- **MVP 只要求一個最小可跑 adapter**:選 **C2 檢索 / BEIR 的一個小子集**(理由:rank-based、確定性、且**不依賴 agent 的 tool-calling 格式**,耦合最低、最可控),
  產出**一個 deterministic metric**(`nDCG@10`),能跟已發表 baseline 並列即算過關。
- **明確 out of MVP**:C1 的 **BFCL / τ-bench / ToolBench** 需把 agent tool-calling 映射到其 function-call 格式,耦合與不確定性高 → **列為後續、非必成**。
- **C3 / C4**:**無公開 benchmark 直接對得上** → 只做主流方法論(labeled set + P/R/F1 + judge–human agreement),不硬掰排行榜。
- ⚠ 依賴:**目前 `pyproject.toml` 無任何 BEIR / IR-eval 套件**(`poetry add` 待實作期);BEIR 子集需下載(網路);
  實作期須先**指定 exact subset + 套件**(例如 `beir` 套件的單一小 dataset),知識截止 2026-01,確切 API/格式**實作期實證**。spike 失敗不阻擋 C1–C4 的內部數字交付。

---

## 接點 / 模組結構

沿用 `agent/evaluation/` package,重構並退役舊三 suite,reusable scorer 搬進 `metrics/`。結構:

```
agent/evaluation/
  harness.py        # BaseEvaluator(沿用) + n-sample 執行 + 收斂成 EvalResult
  repro.py          # 版本 metadata 蒐集(雙 git sha / store 指紋 / seed / model ids)
  ledger.py         # append-only 寫入 + 版本 diff helper
  datasets/         # loader / schema / provenance 驗證 code(JSONL 資料在 repo root eval/datasets/)
  metrics/          # Tier1 確定性 metric(rank metrics / tool-routing / citation / P-R-F1)
  judges/           # Tier2 LLM-judge + judge–human 一致度
  claims/           # c1_routing.py  c2_retrieval.py  c3a_validator.py c3b_reviewer.py c3c_session.py  c4_endtoend.py
  benchmarks/       # 公開 benchmark adapter(spike: BEIR 小子集)
```

**CLI 入口(釘死,不另開新檔)**:**改現有 `agent/cli/eval.py`**(目前 `--suite behavior|e2e|thinking`,見 `agent/cli/eval.py:28`):
- 新增 `--claim c1|c2|c3|c4 --all`(可單 claim 可全測);
- 遷移期保留舊 `--suite` 可跑,C1–C4 全數遷完後**退役舊 suite**。

**Slash command**:目前 registry(`agent/cli/slash_commands.py`)**無 eval 指令**。**列為獨立的後階段(Phase 7,非 MVP)**,屆時新增一個 `SlashCommand` 定義 + 對應測試,不在前幾個 phase 阻塞。

- 驅動:C1/C2 用 `ChatSession.turn_with_trace`(prediction 入口);C3b 直呼 `review_draft`+`route_review_report`(不過 session);
  **C3c 不可只靠 `turn_with_trace`**(retry 不在 trace_events/turn_logs)→ 用 `build_graph().invoke()`+state 檢查,或 `progress_cb` 捕 node,或直測 `_apply_final_skill_validation`(詳見上方 C3c)。
- config 共存:eval 專用覆寫(temp=0、pinned models、store 路徑)疊在 `AgentConfig` 上,不污染正常跑。

### 重用既有零件(注意:多為 instance method / nested fn,需先抽純函式或經 integration 測,不能直接 import)

- `agent/evaluation/base.py`:`EvalResult`、`BaseEvaluator`、`tool_inventory`(**補上 `bash`/`read_file`**)— 可直接 import。
- `agent/evaluation/behavior.py` 的 `_score_tool_expectations`(instance method)與 `_missing_required_tools`(`@staticmethod`)都**在 `BehaviorEvaluator` class 內、非 module-level** → **先抽成 module-level pure function** 再共用(staticmethod 抽離成本低)。
- `agent/thinking.py::route_review_report`、`ReviewReport.model_dump`(C3b)— module-level,可直接用。
- `agent/skills/validator.py::validate_skill_output`(C3a)— module-level,可直接用。
- `agent/graph.py` 的 `skill_validator_node` 是 `build_graph()` 內 **nested function**,**不可 import** → C3c 經 graph/session integration 觀測。
- 既有 JSON case 格式 → 演進成帶 `id`/`provenance` 的凍結 dataset。

---

## 實作 Roadmap(phased,每階段獨立可交付 + 驗收標準)

**不要一次實作整包。** 按階段做,每階段以 unit test 綠燈收尾。前 3 階段大多**不需 API key/Ollama** 即可驗。

- **Phase 1 — 地基(dataset schema + loader + ledger + repro/fingerprint)**
  - 做:`datasets/` loader+schema 驗證、`ledger.py` append-only、`repro.py`(雙 git sha + store 指紋 + seed + model ids)。
  - **驗收**:(a) loader 能驗一份 sample JSONL 並拒絕壞 schema;(b) 連寫兩次 ledger → **兩行,舊行不變**;
    (c) repro 對 **Chroma collection 內容(ids+documents+metadatas+embeddings)** 算出穩定 fingerprint(非只 `raw.json`),且**故意改動 → 啟動時硬報錯**;(d) 全 unit-tested(指紋計算可離線;取 embeddings 需本機 store/Ollama,視測試環境可 mock)。
- **Phase 2 — C1 遷移**
  - 做:`tool_inventory` 補 `bash`/`read_file`;把現有 16 個 behavior case 凍結成 `eval/datasets/c1/`(補 `id`/`provenance`);claim runner 重用 `_score_tool_expectations`。
  - **驗收**:同一組 inputs 下,新 runner 的 per-case pass/fail 與現 `behavior.py` scorer **逐項一致**(可離線比對 scorer);live 跑需 OpenRouter。
- **Phase 3 — C2 檢索(確定性,吃固定 fixture)**
  - 做:`recall@k / MRR / nDCG@k`(rank-based);C2 dev set(數筆 query + 人工標 `(pid,chunk_id)` gold);fingerprint guard 接上。
  - **驗收**:同一 fixture 連跑兩次 → **Tier1 數字完全一致**;fingerprint mismatch → 中止。需 Ollama 嵌入。
- **Phase 4 — C3 三子評估(a/b/c,不可只評 reviewer)**
  - 做:C3a validator P/R;C3b reviewer 分類器 P/R/F1(gold 含「該查紀錄卻沒查」case);C3c normal+extended 兩路徑整合。
  - **驗收**:C3b 對 gold 算得出 P/R/F1;C3c **在 normal 與 extended 兩條路徑各驗一次** retry 是否如預期觸發、最終輸出是否仍違規。需 live env。
- **Phase 5 — C4 checklist rubric**
  - 做:≥N 個任務,每個附**確定性 checklist**(必叫工具、必含事實/檔名);holistic Tier2 選配。
  - **驗收**:checklist 分數確定性可重現;Tier2 僅在 judge–human 一致度達標時納入正式報告。
- **Phase 6 — 公開 benchmark Spike(非阻塞)**
  - 做:接一個 **BEIR 小子集**,產出一個 `nDCG@10`。
  - **驗收**:跑出一個可重現、可跟已發表 baseline 並列的數字。**失敗不阻擋 C1–C4 交付。**
- **Phase 7 — slash command(非 MVP)**
  - 做:在 `agent/cli/slash_commands.py` 新增 eval 指令 + 測試。

### 跨階段端到端 sanity(實作期第一件事)
⚠ 跑 `turn_with_trace` 與 judge 需 **live env**(OPENROUTER_API_KEY + Ollama + 網路),plan mode 無法驗 →
實作期**第一步先跑通一個 smoke case**,看真實 sample output,再往 Phase 2+ 推。
版本化驗證:故意改一個 config 重跑 → 確認 ledger **append 不覆蓋**、diff helper 標出變動 case。

### 現況基線(Codex 已跑,17 passed)
`poetry run pytest tests/test_behavior_eval.py tests/test_eval_runtime.py tests/test_thinking_eval.py tests/test_e2e_eval.py` → **17 passed**。
→ 既有 eval code 目前綠燈,遷移有可對齊的工作基線(但其**數字**仍不可信,只當 scorer 對齊參照)。

---

## 已知風險 / 待實作期確認

- 公開 benchmark 與 agent tool-calling 格式的相容性(unknown，最大不確定)。
- temp=0 後殘餘非決定性程度 → 決定 n_samples 該設多少。
- judge–human 一致度若太低 → Tier 2 數字要降級為「僅供 error analysis，不入正式報告」。
- 目前在 branch `fix/review-routing-hardening`(領先 main 一個 commit，C3 routing 行為與 main 不同)；evaluator target 當前 working tree。
- 舊 eval 結果(8 case、72% zero)**不可當 baseline**；第一次新 run 即為 baseline。
