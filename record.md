# Minervamuses/app 編年史

本文是一份自足的 repo 演進紀錄。目標不是摘要，而是讓沒有參與過此專案的人，只靠這份檔案就能理解 `Minervamuses/app` 從 2026-03-28 到 2026-06-15 之間為什麼會長成現在的樣子：當時面臨什麼問題、用了什麼技術、做過哪些計畫、哪些設計真的落地、落地後產生什麼結果，以及後續還留下哪些待解問題。

掃描基準是 `record/` 產生時的 `fa1cd8d36cce81cdb74da2608e124b95eea9f2d5`，也就是新增歷史紀錄前可達的 279 個 commit。後續 `e5945a4 docs: add project history record` 是把歷史材料整理進 `record/` 的 commit，本檔則建立在那些材料上。主要素材包括完整 commit 表、`project-turning-points.md`、`relevant-commits.md`，以及 `record/source-documents/` 裡保存的計畫、規格、報告與除錯紀錄。

## 專案的核心脈絡

這個 repo 一開始不是一般聊天機器人，而是一個為 PiDNA2 研究與程式碼理解服務的知識管理系統。它最早叫 `kms`，重點是把資料切 chunk、嵌入、存進 ChromaDB，然後讓 CLI 可以查詢。很快地，單純的 retrieval 不夠用了，因為使用者需要一個能在本機專案、研究筆記、聊天歷史、網路與 GitHub 資訊之間切換的 agent。於是這個 repo 經歷了幾次大的轉向：

1. 從 `kms` 知識庫變成 tool-calling agent。
2. 從自寫 agent loop 轉成 LangGraph runtime。
3. 從一個 package 拆成 `rag` core 與 `agent` app 兩層。
4. 從短期對話記憶轉成可持久化查詢的 `history_rag`。
5. 從一般 agent 加上 skill runtime、tool policy、academic-writing guardrails。
6. 從 normal agent 加上 `/thinking extended` 的 prompt rewrite、reviewer、reviser 流程。
7. 從舊式 behavior/e2e eval 轉成 C1-C4 claim-based evaluation。
8. 從多個散落 source of truth 整理成 base tool inventory、skill state serializer、統一 LLM access contract。

這些變動背後的共同問題是：這個 repo 不只要「回答問題」，還要能可靠地知道自己有什麼工具、能不能用某個工具、什麼資料是真的存在、什麼答案有證據、什麼情況要停止搜尋而不是繼續繞圈。多數大改都是在處理這些邊界。

## 2026-03-28 至 2026-03-31：KMS 與 multi-layer RAG 的起點

最早的 commit `6097e03` 建立了 multi-layer RAG KMS 的 `kms/` package。當時的系統由幾個傳統 RAG 元件構成：

- `kms/chunker/`：負責把文件切成 chunk。
- `kms/embedder/`：負責嵌入，早期以 Ollama embedder 為主。
- `kms/retriever/`：負責 vector retrieval。
- `kms/store/`：後來加入 Chroma store、document store、JSON store。
- `kms/llm/`：OpenRouter LLM wrapper。
- `kms/cli/ingest.py`：把 repo 或資料目錄 ingest 進 store。

這個階段的問題很基礎，但也決定後來架構的方向。`38f1a71` 修正 `langchain_core` import path 與 `store/` gitignore pattern，`d63e8eb` 修正 store directory 需要使用 absolute path，代表系統一開始就在本機資料、向量庫持久化、套件 import 邊界上踩過坑。`67212dc` 把環境改成 conda + Poetry，並加入 batch repo ingestion，顯示這不是單次腳本，而是要長期跑在固定 Python 環境裡。

初期的 retrieval 也很快從「查相似 chunk」演化成帶 metadata 的知識庫。`bf2d065` 加入 multi-layer ingestion 與 LLM folder tagging，`28b8677` 抽取更豐富的自動 metadata，`ceeac43` 加入任意 metadata filtering。這些功能背後的需求是：PiDNA2 的資料不是平面文件，資料夾、日期、檔案類型、標籤都會影響正確檢索結果。單純 embedding similarity 不夠，需要 metadata filter 配合。

2026-03-29 到 2026-03-31，agent 的雛形開始出現。`b410078` 加入 tool-calling support，`a23f311` 加入 search tool schema 與 metadata filter executor，`44ae6a3` 把 `chat.py` 改寫成 tool-calling agent loop。這是第一個重要轉折：使用者不只是查 KB，而是要 agent 自己決定何時搜尋、怎麼過濾、怎麼把搜尋結果組成回答。

這段沒有找到單獨的 plan/spec 檔，但 commit 序列很清楚：搜尋工具先做成多 collection，再改成自動發現 collection，最後又收斂回 single collection 加 category/tags metadata。`082403a`、`3d2e20d`、`ce97fbd`、`571780f`、`95714ea` 這些 refactor 表示當時一度嘗試讓資料分 collection，但後來發現多 collection 常數與查詢邏輯太重，改成一個 `KNOWLEDGE_COLLECTION` 搭配 metadata filtering 比較可控。`ad56ec2` 的 `ExploreTool` 與 `50d8073` 的 `ContextTool` 則補上兩個 retrieval 工作流：先探索 KB 裡有什麼，再對命中的 chunk 取上下文。

這個階段的落地成果是：repo 有了可 ingest、可 query、可 chat、可用工具搜尋的 KMS 原型。遺留問題是 agent loop 仍是自寫，工具 abstraction 仍在 `kms/tool/`，還沒有成熟的 graph runtime，也沒有可靠 evaluation。

## 2026-04-02：從自寫 agent loop 轉向 LangGraph

2026-04-02 的一批 commit 把 agent 從自寫 loop 帶到 LangGraph。背景是：自寫 tool loop 可以跑，但很快會遇到 state、tool dispatch、message history、recursion control 與未來擴展問題。LangGraph 提供 StateGraph、node routing 與 LangChain tool binding，比自寫 while loop 更適合逐步長成正式 agent runtime。

具體技術變更如下：

- `a316c5f` 加入 `langgraph` 與 `langchain-openai` dependency。
- `36effd3` 新增 `get_chat_model()` factory，讓 LangGraph 可以取得 ChatOpenAI-compatible model。
- `1b0a8e9` 把原本工具轉成 LangChain `@tool` factories。
- `8013a1c` 新增 `kms/agent/`，裡面有 `graph.py` 與 `state.py`。
- `ca4fd5a` 把 `cli/chat.py` 改寫成跑 compiled graph。
- `991cf00` 刪掉 custom `BaseTool` ABC。
- `20b9b36` 刪掉死掉的 `ChatResponse` / `ToolCall` 與 `OpenRouterLLM.chat()`。

這段的關鍵不是「多了一個框架」，而是 agent 的權責從 CLI loop 移到 graph runtime。後續 skill loader、policy tool node、validator、extended thinking 都能接到 graph 上，是因為這裡先建立了 graph/state 的骨架。

當時沒有找到獨立 plan，但 `4f9365e docs: add info.md explaining langgraph branch changes` 與 `b36f65e docs: add pipeline SVG diagrams to info.md` 表示作者有用 `info.md` 記錄 LangGraph branch 的差異與流程圖。這一份 `info.md` 後來多次被改寫，沒有作為獨立 source document 保存，但它在 commit 歷史中扮演過架構說明文件。

落地後的成果是：工具不再只是自訂 class，而是 LangChain tool；agent runtime 有 StateGraph；CLI 只是 graph 的 host。後續問題則轉向：graph 裡的 state 怎麼控、工具結果怎麼保留、evaluation 怎麼測。

## 2026-04-05 至 2026-04-13：Evaluation 起步與第一次可靠性危機

2026-04-05，repo 加入第一版 evaluation module。`1955d15` 建立 `kms/evaluation/`，包含三類 evaluator：behavior、retrieval、end-to-end。接著 `2b33844` 加入 `kms/cli/eval.py`，讓 eval 可以一個命令跑。這代表作者開始意識到：agent 行為不能靠手感判斷，需要系統性跑 cases。

第一版 eval 很快暴露問題。4/5 的 commit 中有許多修補：

- behavior evaluator 要支援 multi-turn cases。
- `_extract_json` 要能處理 markdown code fences。
- end-to-end evaluator 不應直接依賴 `get_state`。
- multi-turn case 沒有 `question` key 不能 KeyError。
- recursion limit 要提高到 32 並捕捉 `GraphRecursionError`。
- e2e evaluator 要拆成 generation LLM 與 judge LLM。
- RetrievalEvaluator 被重新定位成 embedding quality unit test，後來又被移除。

這些修補指出第一版 evaluator 的痛點：它一開始混合了 retrieval、behavior、judge parsing、agent graph state，多個不穩定因素會互相干擾。當 agent 真的跑失敗時，很難知道是模型沒有答好、judge JSON 解析失敗、graph recursion、還是 retrieval case 本身設計錯。

2026-04-11 的 `note/20260411/report.md` 是第一份完整可重現的評估報告。當時的 setup 是：

- Agent LLM：`z-ai/glm-5`
- Generation LLM：`google/gemini-3.1-pro-preview`
- Judge LLM：`openai/gpt-5.2`
- Filter LLM：local Ollama `llama3.1:8b`
- Embedder：local Ollama `bge-m3`
- recursion limit：32

相較 4/5，這次換了 agent LLM，加入 local Ollama chunk quality filter，並移除 redundant 的 `chunk_hit_rate` metric。chunk quality filter 會先擋掉 compiled JS、lock files、SQL DDL 等不適合拿來生成問題的 chunk，這說明當時 evaluation 不是只在測 agent，也在修「測資生成」本身的品質。

結果很差但很有用。End-to-end 從 30 個嘗試 cases 中被 filter 跳過 5 個，剩 25 個計分；avg_score 只有 25.3%，avg_score_raw 是 0.76/3，score_3_pct 24%，score_2_pct 0%，score_1_pct 4%，score_0_pct 72%。Behavior suite 也只有 8 個 built-in cases，分項是 first_tool_accuracy 60%、tool_count_accuracy 75%、no_tool_accuracy 100%、tools_coverage 0%、filter_accuracy 100%。這些數字讓問題輪廓很清楚：agent 並非完全不會選工具，該不用工具時也能閉嘴，但它幾乎不會完整覆蓋應使用的工具集合，且工具次數控制不穩。

18 個 zero-score cases 被分成兩類：10 個 hit recursion limit，占 zero cases 56%；8 個 judge parsing failed，占 44%。報告裡的重要結論是雙峰分佈：agent 只要能乾淨完成，通常能拿高分，7 個非零 cases 中有 6 個是滿分；一旦失敗，就整個 turn 爆掉。這個觀察後來反覆出現：repo 的主要問題常常不是「答案品質平均偏低」，而是某些 runtime 邊界一破就變成 hard failure。

當時看到兩個關鍵問題：

1. `z-ai/glm-5` tool-happy。簡單問題可以叫 10 到 18 次工具。SYSTEM_PROMPT 說 1 到 3 次搜尋後要 synthesize，但模型沒有停。
2. Judge parsing failure 可能把有內容的答案直接打 0。報告指出有些 raw answer 看起來是實質回答，但因 judge JSON parse fail 被歸零。

計畫中的 next steps 是：先修 judge parsing，例如讓 OpenRouter judge 呼叫支援 JSON object response format 並記錄 raw response，預估可能救回約 4 個 cases，avg_score_raw 可提高 0.3 到 0.5；再收斂 agent over-looping，預估可能救回 5 到 8 個 cases，avg_score_raw 可提高 0.6 到 1.0；最後再判斷 GLM-5 是 prompt adherence 問題，還是模型本身 tool-use default 與這個 agent 不合。報告也保留了當時的結果檔路徑：`store/eval/e2e_cases_20260411_1915.json`、`store/eval/e2e_results_20260411_1915.json`、`store/eval/behavior_results_20260411_1908.json`。這些路徑本身後來成為「舊 eval 結果不可當 baseline」的背景，因為 schema 與 suite 都在 5 月重建。

2026-04-13 的 `note/20260413/agent_state_cleanup.md` 則處理另一個可靠性問題：LangGraph state growth。當時的任務是減少不必要的 state 成長，特別是：

- 不要把完整 `ToolMessage` 內容長期存進 history。
- 限制同一 turn 內的 tool-context growth。
- session 是 process-scoped，重開不延續舊 graph checkpoint memory。

使用者接受了幾個決策：A，移除完整 `ToolMessage` contents from long-term history；E，對 prompt-visible history 套 fixed message-count cap；F，session ends with process，下一次 launch 從乾淨 conversation 開始。使用者拒絕 B 與 C，也就是不接受只保留最近 1 turn 或最近 2-3 turns，因為記憶策略後續另有計畫。部分接受的是 D，bound same-turn tool interactions，window 設成 4。這組決策很重要，因為它不是單純技術 refactor，而是使用者對「短期對話可見性」和「長期可追溯記錄」之間的取捨：不能為了省 token 直接犧牲最近對話脈絡，但也不能讓 tool payload 無限制累積。

實作落地到：

- `kms/agent/graph.py`：移除 LangGraph checkpoint-based cross-turn memory，改成每次 agent step 前準備 bounded prompt。
- `kms/agent/history.py`：新增 deterministic history helpers，同 turn 只保留最近 4 個 tool interactions，舊工具活動用 truncation note 表示。
- `kms/cli/chat.py`：`ChatSession` 明確管理 session history；long-term prompt history 只保留 system prompt、user messages、final assistant answers。
- `kms/config.py`：新增 `agent_max_messages = 20` 與 `agent_max_tool_interactions = 4`。
- evaluation 改成走實際 compact-session chat behavior。

這個 cleanup 在當時是合理止血，但後來 5/30 runaway debug 會證明：同 turn 只保留最近 4 個 tool results 會在工具呼叫爆量時造成「失憶迴圈」。也就是說，4/13 的決策在「正常 turn 不超過 4 次工具」的假設下成立，但當 agent 真的爆走，它會把早期證據裁掉，反而讓 agent 不知道自己已經查過什麼。

## 2026-04-18：RAG 與 Agent 邊界拆分

2026-04-18 是第一次大型架構切分。當時的 `repo_split_plan.md` 明確寫出目標：這個 repository 要成為一個 neutral workspace，裡面有兩個 peer packages：

- RAG/core package：負責 indexing、storage、retrieval、public retrieval API。
- Agent/app package：負責 tool adapters、graph orchestration、chat CLI、evaluation。

這是往後續 MCP-style deployment boundary 前進的 staging plan。計畫中對邊界的定義很清楚：

`rag` 應該是 source of truth for：

- public API：`rag.api`、`rag.types`、`rag.filters`、`rag.config`
- data pipeline：chunker、embedder、tagger
- retrieval/storage：retriever、store
- RAG-side CLI：ingest、query

`agent` 應該是 source of truth for：

- graph/session/state/history
- framework adapters
- evaluation suites
- agent-side CLI

這個 plan 不是空話，commit 序列顯示它很快落地：

- `f6ce5c3` 先加 decoupling refactor smoke baseline。
- `80eafb3`、`72177bc`、`833f84d` 把 path、folder metadata、where-clause builder 等核心工具抽成比較 framework-neutral 的 helper。
- `e0b9eedd` 加入 `kms.api` public API。
- `2192c7b` 把 `kms/tool/` 移到 `kms/adapters/langchain`，讓工具變成薄 wrapper。
- `b94d55a` 把 `ChatSession` 移到 `kms/agent/session`，並把 evaluation 從 CLI 解耦。
- `968edf0` 把 agent layer 從 `kms` core 抽成 `kms_agent`。
- `8f676ca` 把 core packages rename 成 `rag` 與 `agent`。
- `6be7f94` 把 repo root 當作 rag-agent workspace。
- `0598ec3` 把 `rag` package nested 到自己的 project directory。
- `8c320a3` 把 workspace 拆成 agent 與 rag，各自有 conda env 與 Poetry。
- `1761983` 最後把 `rag` 移出 app，改成 `../rag` path dependency。

這段的技術意義很大：在此之前，agent、RAG、evaluation、CLI 都還像同一個 package 的功能。拆分後，agent 不再應該直接碰 `rag.store` 之類的內部實作，而是透過 public API 或 adapters。這也解釋後來為什麼 evaluation、tools、history、MCP 都在 `agent/` 內繼續成長，而 core retrieval 則交給 sibling `../rag`。

計畫裡還留下兩個未完成方向：Step 3 要把 core API 包成 server boundary，可以是 MCP tools、HTTP/JSON RPC 或 local subprocess tool server；Step 4 要把 physical repo split 到 `PiDNA2/` 下的 sibling path。後來 `rag` 已經變成 `../rag` path dependency，代表 physical split 在實務上已部分達成，但 server boundary 並沒有在這段完全完成。

## 2026-04-20：MCP、async session 與 stdio 防禦

2026-04-20 的主軸是讓 agent 能載入外部 MCP tools，尤其 Web Search 與 GitHub。這裡的背景是：本機 RAG 只能回答已 ingest 的資料，agent 若要查最新網路資料或 GitHub 遠端狀態，就需要 tool server。MCP 是合適的 protocol，但也帶來 operational 問題。

`note/20260420/mcp_setup.md` 把兩層 MCP 清楚分開：

- `PiDNA2/opencode.json` 是外部 `opencode` host 的 config，不影響 Python LangGraph agent。
- `app/agent/mcp.py` 是 Python runtime 的 MCP loader，由 `app/.env` 裡的 env vars 驅動。

這個區分很重要，因為使用者很容易以為在 opencode 設 MCP 後 Python agent 也會有同樣工具。實際上兩個 host 完全獨立。

當時 Web Search MCP 需要安裝 `mrkrsl/web-search-mcp` release zip，並用 Node 啟動；它不是 npm package，而是 GitHub release zip，且需要 `npx playwright install chromium` 下載約 300 MiB browser binaries。成功後預期工具是 `full-web-search`、`get-web-search-summaries`、`get-single-web-page-content`。GitHub MCP 則用 GitHub 官方 Go binary，啟動參數是 `stdio`，toolsets 限在 repos、pull_requests、issues、actions、context。文件也明確指出 GitHub MCP 不是 local git workflow 的替代品，local clone/pull/rebase/commit 仍應走 terminal。

這段落地了一批 runtime 變更：

- `78bc8bd` 加入 MCP stdio loader for Web Search and GitHub servers。
- `3357bd3` 加入 tool-policy prompt 與 async `ChatSession.create()`。
- `dcd8fc2` 讓 CLI async startup 可以載入 MCP，並加入 `.env.example` 和 tests。
- `aa4ba65` 把 agent-only settings 從 RAG 移到 `AgentConfig`。
- `e68db32` 讓 CLI startup 載入 `app/.env`。

但 MCP stdio 也暴露很多上游穩定性問題。某些 MCP server 會把 human-readable banner 或 notice 印到 stdout，而 stdio JSON-RPC client 只接受 JSON-RPC lines。於是有幾個 fix：

- `c0b8518` silence stdio parse-error spam。
- `33d011b` 用 `/bin/sh` 包裝 launch，silence server stderr。
- `9c39917` filter subprocess stdout，只有 JSON-RPC lines 送進 client。
- `1fc93bc` 讓 `turn()` 變 async，支援 MCP async-only tools。
- `613dbcb` 把 tool exceptions 轉成 tool messages，而不是 crash 整個 turn。

這段還有 `2a67618` 到 `4e62926` 的 memory work：turn-aware memory module 與 rolling compaction。當時長期記憶還不是後來的 history vector DB eviction，但已經開始從單純 prompt history 走向 turn-aware state。

落地成果是：Python agent 可以在 startup 時載入外部 MCP tools，且即使某個 MCP server 失敗，session 仍會啟動，只是少了那些外部工具。若完全不用 MCP，可在 env 裡不開 `AGENT_ENABLE_MCP_*`，或 CLI 用 `--no-mcp`；此時 agent 仍有 local KB tools。這種 failure behavior 對後來 evaluation 很重要，因為 C1 web cases 可以在 `--no-mcp` 時被 skip，而不是讓整個 eval 崩潰。文件還順手記錄了一個不直接相關但重要的記憶設定：long-term conversation memory 當時每 `config.agent_turns_per_compaction` 個 completed turns，預設 10 turn，會 compact 成 rolling summary；這是在 history vector store 完全成形前的過渡設計。

## 2026-04-25 至 2026-04-27：長期記憶從壓縮轉成 History RAG

2026-04-25，對話長期記憶換了策略。之前有 rolling compaction，但後來改成「recent turns window + vector DB eviction + recall_history tool」。背景是：LLM compaction 會壓縮資訊，但不可避免地丟失細節，而且使用者常常不是要摘要，而是要 agent 找到過去具體說過什麼。

主要 commit：

- `d795650` 新增 `agent_recent_turns_window`，為 upcoming history_rag eviction 做準備。
- `44cd3b3` 加入 `ChatHistoryStore` 與 module-level cache。
- `e07c0e2` 加入 `recall_history` StructuredTool factory。
- `07ebc6b` 把 `recall_history` wire 進 agent toolset。
- `a377ad1` 用 vector-DB eviction 取代 LLM compaction。
- `50bb2c4` 文件說明 `history_rag` long-term memory。

接著幾天修了不少邊界：

- `d7b5063` 修 tool pruning 時要保留 unevicted turns。
- `b0fc5dd` chat exit 時 flush recent turns。
- `37d2c6e` 加 test 覆蓋 chat exit flush。
- `10eed6f` normalize quit commands。
- `eed4cfa` 把 history store 注入 recall tool。
- `259ce19` 讓 end-to-end eval 與 real chat history store 隔離，避免 eval 汙染真實歷史。

這段沒有找到原始 plan 檔，但它後來在 5/24 至 5/27 引出重大失敗：active skill 啟用後，writer 看不到 `recall_history`，使用者明明要求 agent 查看先前紀錄，agent 卻一直問使用者補資料。也就是說，history_rag 本身落地了，但 tool policy 與 active skill 之間的整合後來才補齊。

## 2026-05-03 至 2026-05-10：互動式 CLI、slash commands、plan mode 與 bash

2026-05 初，使用者介面從簡單 chat CLI 變成更完整的操作環境。`97cd1c1` 先把 chat line reader 注入，`245edef` 使用 `prompt_toolkit`，`7a92f40` 加入 slash commands 與 completion。接著 `086cd58` 加入 `/ingest`、`/sync`、`/prune`，`51765b0` 加入 `/init`，讓 agent 可以 ingest parent repo minus this app。

這代表 CLI 不再只是輸入一句話拿回答，而是 agent runtime 的操作台。使用者可以切模式、載入資料、同步索引、控制記憶或工具。

同一時期討論模式被整理成 plan mode：

- `4d2620b` 加 discussion turn metadata。
- `7cdea45` 把 discussion turns 存成 markdown。
- `28b498e` 在 CLI 暴露 discussion mode。
- `71bbbff` 內部把 discussion mode rename 成 plan mode。
- `1ee453c` 用 `/mode` framework 取代 `/discuss`。
- `7920ab6` 把所有 tool results 記錄到 plan log markdown。
- `8241ab6` 把 plan-mode hint 注入 prompt history。
- `34c95a8` gitignore `plan_logs/`。

這段的重點是：plan mode logs 是 markdown logs，不會進 Chroma `chat_history`。這件事後來在 history recall spec 裡被明確寫成文件要求，因為使用者若說「你之前看過」，那段資訊可能在 recent visible context、persisted chat_history、plan_logs、indexed KB 或 repo file 中，不一定能用 `recall_history` 查到。

2026-05-10 還加入 bash tool：

- `2031752` 加 bash tool with mandatory user approval。
- `7193370` wire bash tool into chat graph and system prompt。

Bash tool 的加入讓 agent 能執行本機命令，但也提高風險。後續 skill policy、read_file confinement、C1 forbidden tool scoring 都和這個能力有關。C1 runaway debug 中也曾一度誤以為 agent 偏好 bash，其實後來證明 agent 是先 RAG 爆走，最後才逃去 bash。

## 2026-05-14 至 2026-05-18：Skill runtime、tool policy 與安全邊界

2026-05-14 起，repo 進入 skill runtime overhaul。背景是：agent 需要根據使用者任務切換不同能力與規則。`academic-paper-writing` 這種 skill 不只是一段 prompt，它需要限制工具、載入參考資料、檢查輸出是否違反學術誠信、甚至要求不同 task mode。

主要落地內容包括：

- `749494e` 加 capability broker。
- `b635c8b` 加 skill runtime loader。
- `e5f38ee` 在 state 裡加入 skill runtime fields。
- `037590a` 讓 session wire active skill runtime。
- `cda88c3` 加 skill slash command。
- `77b8b6d` 讓 `read_file` skill-root aware。
- `692763b` 加 skill loader graph node。
- `d03857a` 暴露 skill runtime toggles。
- `7866359` enforce skill tool policy。
- `473c010` bind tools per active skill。
- `e134d86` validate active skill responses。
- `a0dc7c6` 加 runtime adherence tests。

這段的設計不是單純「把 prompt 換成某個技能」。它有幾個實際 runtime 邊界：

1. Capability resolution：skill manifest 宣告需要哪些 capability，broker 把 capability 解析成具體 tool。
2. Tool policy：active skill 可以限制工具，例如 `academic-paper-writing` 禁用 `bash`。
3. PolicyToolNode：即使模型要求被禁止的工具，runtime 也要拒絕，而不是只靠 prompt。
4. Skill root aware read_file：skill references 可以被讀取，但不能任意讀敏感檔案或逃出 skill root。
5. Final validation：active skill 的輸出要經過 validator，必要時 retry。

5/18 的 fix 又補強幾個風險：

- `dadd58e` 讓 tool policy 更 explicit。
- `9d3e07e` 把 denied tools 標成 errors。
- `7856576` 讓 skill resources 保持 scoped。
- `7dc721d` block sensitive file reads。
- `df31826` validate manifests。
- `fd882f3` deterministic validators。

這一階段的落地成果是，agent 有了可擴充的 skill 系統，並且工具可用性不是只寫在 prompt 裡，而是 graph runtime 會真的執行。後續也因為這個機制，history recall 出現了「active skill policy 排掉 history tool」的問題。也就是說，skill runtime 本身是正確方向，但它使「base prompt 說工具 always available」與「active skill 實際可用工具集合」之間的矛盾變得不能忽略。

## 2026-05-24：Extended Thinking v3.4

2026-05-24 的 `thinking_extended_plan.md` 是本 repo 最完整的設計計畫之一。背景是：早先 v2 的 `/thinking extended` 走「結構化 TaskSpec」路線，但這和使用者真正想要的流程不一致。使用者腦中的 extended mode 是：

1. 先把 prompt 重寫得更明確。
2. 同一個 agent 拿重寫 prompt 去做事。
3. 第二個 reviewer agent 審查。
4. 原 agent 根據 reviewer 意見修改或駁斥。
5. 最多兩輪後輸出。

v3.4 把設計校準到這條路線，並加入一個之前忽略的關鍵：不同角色應該使用不同 LLM。若 Writer、Reviewer、Reviser、prompt rewrite、format repair 都用同一個 `config.llm_model`，那就是同一模型審自己，blind spot 會重疊。v3.4 因此新增角色 model 欄位：

- Writer：既有 graph，使用 `config.llm_model`。
- prompt-master rewrite：裸 LLM，使用 `config.thinking_rewrite_model`。
- Reviewer：裸 LLM，使用 `config.thinking_reviewer_model`，建議跨 family。
- Reviser：既有 graph，仍用 `config.llm_model`，因為設計上它就是同一個 agent 改稿。
- Format-repair：裸 LLM，使用 `config.thinking_repair_model`。

這三個 thinking role model 預設為空字串，不 silent fallback 到 main model。這是刻意的 forcing function：使用者必須在啟動 extended mode 前明確選擇模型與成本，避免不知不覺退回 self-review。

計畫中的 workflow 是：

1. `_require_thinking_models(config)` 檢查三個 role model 是否已設定。
2. prompt-master rewrite 看 raw user input、visible context、active skill context，並受 char cap 控制。
3. rewrite wrapper 明確禁止新增原始輸入、visible context、active skill context 都沒有提供的 citation、DOI、page number、quote、數據、樣本數、方法細節或研究發現。
4. 若資訊不足，rewrite 輸出 `<<CLARIFY>>`，controller 直接問使用者，不進 Writer。
5. Writer 使用既有 graph，拿 rewritten prompt 跑工具與回答。
6. controller 建立 `evidence_trace_summary`，記錄 Writer tool trace。
7. Reviewer 看到 raw input、rewritten prompt、draft、active skill context、evidence trace、previous rebuttal，產生 `ReviewReport`。
8. Routing 規則決定 pass、ask_user、stop 或 revise。`needs_user_input=True` 與 blocker 會停；major finding 會進 Reviser；minor/note 通常不重寫。
9. Reviser 仍跑既有 graph，但輸出必須分 `DRAFT:` 與 `REBUTTAL:`。DRAFT 回給使用者，REBUTTAL 只給下一輪 Reviewer。
10. 若 Reviser marker 缺失，先用 repair model 嘗試修，失敗再啟發式剝尾，最後才整段當 draft 並加 user-visible warning。
11. Review/Reviser loop 最多兩輪，loop 外再跑一次 final skill validation。

這個計畫落地成一連串 commit：`59aa8e0` 加 thinking mode slash command，`a5dc6ff` 加 schemas，`02a674f` wire controller，`41c6bed` 加 thinking reviewer eval suite，`cfc79fd` 用 v3 design 取代 v2 plan，`a326675` 加 thinking role model config，`1c18c0d` vendor prompt-master skill，`88fb116` implement v3.4 flow，`1b47138` configure extended thinking models，`7de10c1` 加 interactive picker，`c06d9fb` enforce per-turn tool budget，後續又修 Traditional Chinese prompts 與 max_tokens。

這段同時產生 `problem.md`，記錄一個真實失敗案例：使用者在 extended mode + academic-paper-writing 下要求 agent 查看自己一月上半的成果，agent 卻反覆要求使用者提供資料，而不是使用可用的歷史記憶。這個失敗後來成為 5/27 history/tool-availability spec 的背景。

## 2026-05-25 至 2026-05-27：History recall 與 tool availability 錯位修復

5/24 的 `problem.md` 暴露了一條具體失敗鏈：

1. 使用者啟用 `/thinking extended`。
2. 使用者啟用 `academic-paper-writing` skill。
3. 使用者要求 agent 自行查看一月上半成果紀錄。
4. `academic-paper-writing/manifest.yaml` 沒宣告 `history.search`。
5. active skill policy 啟用後，graph 根據 `allowed_tools` filter tools，再餵給 `model.bind_tools(...)`。
6. Writer LLM 看不到 `recall_history` schema，只能用 `rag_explore` / `rag_search` 或走 academic intake checklist。
7. `rag_search` 查的是 indexed KB，不是 `chat_history` collection。
8. Reviewer 又看不到 Writer 實際工具範圍，無法區分 tool unavailable、tool available but unused、retrieval empty、或使用者真的沒給資料。
9. Reviewer 最後把問題升級成 `needs_user_input=True`，使用者看到的像是內部審稿意見，而不是可理解的 troubleshooting。

`agent_history_tool_availability_spec .md` 把修復拆成 P0 到 P4。

P0 是讓 `academic-paper-writing` 可以使用 `recall_history`：manifest 的 `capabilities.required` 要包含 `history.search`，resolved allowed tools 必須包含 `read_file`、RAG tools 與 `recall_history`，同時 `bash` 仍要排除。required capability 如果解析不到，activation 必須 fail fast。

P0.5 是先驗證資料前提，不要只靠想像。診斷結果存在 `note/20260527/p0_5_history_query_diagnostic.md`。當時直接檢查本機 `chat_history` collection：

- persist dir：`/home/minervamuses/PiDNA2/rag/store`
- collection document count：108
- oldest timestamp：`2026-04-25T17:11:26.806479+00:00`
- newest timestamp：`2026-05-24T12:25:03.060958+00:00`

多種 query 風格都有 hit：

- `我一月上半做的研究成果`
- `early January research progress`
- `一月 人工智慧`
- `AI January experiments`
- role filter 到 assistant 也有 hit
- bulk inspection 也能看到歷史資料

診斷結論是 hit：本機 chat_history collection 可讀、非空，而且多種 query 風格都能返回一月相關結果。這表示 P0 不只是理論必要條件，它真的能 unblock 使用者問題。

P1 是讓 Rewriter / Writer 都知道實際工具可用性。spec 明確禁止 `thinking.py` hardcode 完整工具清單。應該有一個 shared tool availability block，例如：

```text
active_skill: academic-paper-writing
task_mode: none
tool_policy_active: true
available_tools: [...]
denied_tools: [...]
unavailable_base_tools: [...]
```

這段 context 要是 ephemeral system context，不可以存進 `recent_turns` 或 `chat_history`。

P2 是 Reviewer 要能區分 retrieval failure 類型。這裡最重要的規則是：`needs_user_input=True` 是逃生開關，只能用在 reviser 多跑一輪也救不回來時。若 `recall_history` 可用但 Writer 沒有用，Reviewer 應該產生 `severity=major`、`needs_user_input=False`、`decision=revise`，要求 Reviser 先 call `recall_history`，而不是直接問使用者補研究背景。若 `recall_history` 已查但 empty，答案應誠實說明「已查但未找到足夠紀錄」，不能把 empty 包成使用者沒提供資料。

P3 是 meta-conversation escape hatch。使用者若說「你為什麼一直問我」、「你應該能看見我的紀錄」、「是不是工具沒接上」，agent 不應繼續套 academic-paper-writing intake checklist，而要轉成 troubleshooting frame，說明 active skill、可用/不可用工具、是否查過 history、查詢結果是否 empty、plan logs 是否不在 chat_history。

P4 是文件更新：清楚區分 `rag.search` 與 `history.search`，說明 active skill 啟用後 base prompt 的「always available」其實是條件式可用，也說明 plan mode logs 不進 Chroma。

實作以 P0-P4 的 commit 形式落地：`60945d3` P0 加 academic history search capability，`fbfe550` P1a 加 shared tool availability helper，`d25e40d` P1b pass tool availability to rewrite prompts，`37371d4` P1c pass to review prompts，`352afa6` P1d inject into extended session，`7917b7e` P2 route retrieval gaps back to reviser，`4db22f3` P4 docs。這條線最後進了 PR #1，後續 `fdf906e` 又 harden review routing safeguards。

落地後的意義是：history recall 不再只是 base tool，而是 active skill 下可被 capability resolution 正確宣告、可被 prompt 認知、可被 reviewer routing 正確解讀的工具。這也奠定 6/14 regression tests 的基礎。

## 2026-05-27 至 2026-05-31：Evaluation 重建成 C1-C4 claims

`EVALUATOR_PLAN.md` 是另一份大型計畫。背景是：開發者做了多輪大規模 agent 改動後，意識到原有 evaluation 已過時且不可信。舊 `agent/evaluation/` 三個 suite 是幾個大改前的產物，上次 run schema 沒版本標記，behavior 只有 8 cases，e2e 又有 72% 被判全錯。這些數字無法支撐後續 paper 或對外說明。

使用者確認了三個核心問題：

1. 評什麼：四個 claim 全要，且可單獨跑、可一次全測。
2. 要得到什麼：可信的絕對數字，加 append-only、帶版本標記、不覆蓋的結果檔。
3. 最怕什麼：自製評估與主流學界/業界對不上，對外沒有說服力。

因此新 evaluator 的原則是：核心數字必須來自 deterministic code；LLM/agent 可以做 error analysis 或建議，但不能自己給分。metrics 分兩層：

- Tier 1 deterministic：集合比對、rank metrics、regex、route scoring、validator pass rate、P/R/F1。
- Tier 2 LLM-judged：RAGAS-like faithfulness 或 holistic task completion，只能附 judge model/version/prompt hash 與 human agreement。

四個 claims 定義如下：

- C1 工具路由正確性：面對問題選對工具家族並正確編排檢索。
- C2 答案忠實度 / 檢索品質：跨多 chunk 取證、有根據、不幻覺。C2 retrieval 用 Tier 1 rank metrics，faithfulness 可做 Tier 2。
- C3 Skill 遵循 + extended-thinking 把關：拆成 validator deterministic checks、reviewer classifier P/R/F1、normal/extended session integration。
- C4 端到端任務完成：用獨立 checklist，不混進 C2。

這份計畫還定義了資料集治理：

- JSONL 放在 repo root `eval/datasets/<claim>/{dev,test}.jsonl`。
- 每筆有 stable `id`、`inputs`、`gold`、`provenance`。
- dev/test 分割，平時只在 dev 迭代，test sealed。
- 新 failure case 先進 dev，每個 release 批次 promote to test。

C2 對固定 eval fixture 有非常嚴格的 reproducibility 要求。因為 semantic retrieval 走 Chroma collection，不是 `raw.json`，所以 fingerprint 不能只 hash `raw.json`；必須透過 Chroma `get(include=["documents","metadatas","embeddings"])` 對 ids、documents、metadatas、embeddings 排序後算 hash。若 store fingerprint mismatch，run 應硬報錯，不自動修復。

計畫分 phase：

1. dataset schema、loader、ledger、repro/fingerprint。
2. C1 遷移。
3. C2 retrieval。
4. C3 三子評估。
5. C4 checklist rubric。
6. BEIR public benchmark spike。
7. slash command。

實作在 2026-05-28 快速落地：

- `4b20d22` dataset loader schema validation。
- `e261583` append-only run ledger。
- `700a86b` reproducibility metadata fingerprints。
- `f2947f5` extract tool routing scorer。
- `015ac60` include local tools in routing universe。
- `75282a6` C1 routing runner。
- `dd4115b` ranked retrieval metrics。
- `28e996f` C2 retrieval runner。
- `2e36b7a` C3 validator evaluator。
- `955a5cb` C3 reviewer classifier。
- `f021510` C3 session validation。
- `3a254e2` C4 checklist evaluator。
- `4bbaff4` BEIR SciFact benchmark spike。
- `61be291` eval slash command。
- `3c0f9ea` evaluation package README。

2026-05-30 的 `note/20260530/c1_routing_findings.md` 是 C1 第一次跑況。當時命令是 `python -m agent.cli.eval --claim c1 --split dev --allow-skips --no-mcp`，run_id 是 `c1-20260530T080146Z-c36d63e0`。dev 共 8 題，評到 5、跳過 3 個 web 題。因為 `allow_skips=true` 且 skipped=3，這個 run 被標成 `baseline_eligible=false`，不能和後來評滿 8 題的 run 直接比較。routing_accuracy 0.6，但分項很有啟發：

- first_tool_accuracy 1.00
- tool_family_accuracy 1.00
- no_tool_accuracy 1.00
- tools_coverage 0.75
- forbidden_tool_accuracy 0.80
- tool_count_accuracy 0.60

兩個真實發現分別是：第一，`rag_search` 類的「scoring 模組怎麼運作」路由是對的，但實際叫了 5 次工具，超過 gold 的 1-4 次，只掛在 count；第二，`rag_get_context` 類的「embedding 模組」加「展開上下文」追問爆走，叫了 15 次工具、狂刷 `rag_search` 12 次、逃去 `bash`，而且全程沒用 `rag_get_context`。所以結論不是 agent 一開始就選錯工具家族，而是「選對門」的直覺很穩，first_tool / family / no_tool 都是 1.0；真正弱點是 count 最差，以及該用專門工具時一直用 search 土法硬找。這個發現直接引出 5/30 tool-call runaway debug。

2026-05-31 又跑了一次 C1-C4 dev claims。結果寫在 `note/20260531/20260531-eval-claim-run.md`。這些 run 都是 dev split，`baseline_eligible=false`，結果在 UTC 14:02 到 14:03 連續寫入 ledger；C1 的 web tools 是故意不載入，所以 3 個 web cases skip 不算失敗。

- C1：4 過 / 1 失敗 / 3 skip，routing 0.80。
- C2：3 題全有命中，recall@5 = 1.0、MRR = 0.83、nDCG@5 = 0.88。
- C3：validator 3/3，reviewer 3 題對 2 題，session 2/2。
- C4：1 過 / 1 失敗，task_success 0.50。

共同教訓是：三個失敗有兩個是「該叫的工具沒叫」。C1 embedding 該用 `rag_get_context` 卻一直 search；C4 history codename 答案內容其實講出 `Blue Lantern`，但沒呼叫 `recall_history`，而且當時中文「部署代號」沒有命中英文 regex `codename|deployment`。C3 reviewer 則有過度挑剔問題，把乾淨稿誤判成 block，failure mode 是 `user_input_missing`，使 decision / route macro-F1 掉到 0.56，但 severity 判斷仍是 1.0。資料集盤點也指出 C2/C4 太薄，C1 每個 category 只有一題，統計不穩：C1 dev/test 各 8 題但每類只有 1 題，C2 dev 3 / test 1，C3 dev 8 / test 4，C4 dev 2 / test 1，而且沒有 `manifest.json` 或 `holdout.jsonl`。C2 test 的 `c2-score-container-columns-frozen` 還只是 dev 同題 frozen 版，獨立鑑別力很低。

## 2026-05-30 至 2026-06-15：Tool-call runaway 的偵錯與修復

5/30 的 runaway 是 repo 重要轉折，因為它把「agent 會不會過度搜尋」從 prompt 問題提升成 graph/history/eval 共同問題。

症狀是 C1 evaluator 跑簡單問題時，agent 叫了 15-16 次工具。起初有幾個錯誤猜測，後來都被推翻：

1. 以為 agent 選錯工具、偏好 grep。實際上它每次都是 RAG 優先，狂試 RAG 12-13 次，bash/read_file 只是最後逃生。
2. 以為 eval 卡在等待使用者核准 bash。實際上使用者已按完核准，後面卡在 reasoning model 下一回合，因為 ChatOpenAI 沒 timeout 且 max retries 靜默重試。
3. 沙箱探測 Ollama 與 dotfiles 造成環境假警報，因此 debug 必須以使用者終端機真實結果為準。

讀 graph/history 程式後，找到兩層獨立裁切：

- Layer 1 turn 級：`ChatSession.recent_turns` 保留最近 10 個 turn，更舊的 eviction 進 Chroma。這層如設計。
- Layer 2 turn 內 tool result 級：`prepare_messages_for_agent` 每個 agent step 只保留最近 4 個 tool results，舊的裁掉。

同時 graph 有 per-turn tool budget：`tool_count >= agent_max_tool_interactions(4)` 時應改用 unbound model 強制 synthesize。但實測 15-16 次遠超預期，代表預算沒有咬住。

Phase 0 instrumentation (`72ff55f`) 在 C1 runner 裡用 `progress_cb` 記錄每個 graph step emitted tool_calls 數、tool args、result previews。跑完後得到三個決定性證據：

1. 預算沒咬住的第一個原因是 parallel overflow。模型一個 AIMessage 可以塞 2 個 tool calls，預算在 step 前檢查，count=3 時模型一次發 2 個就衝破。
2. 失憶確實存在。embedding case 的 query 序列中，第 5 次幾乎重做第 2 次，第 6 次又 explore；tool result preview 顯示第 5 = 第 3、第 8 = 第 2，是重複命中。
3. 更關鍵的是：檢索回來的全是無關資料，因為答案不在語料庫。embedding module 位在被 ingest 排除的 workspace 中，RAG 索引裡沒有它，所以檢索只能回 `JmolWidgetset cache.html`、`3dmol.d.ts`、`poetry.toml`、`SaveToLink.java` 這類垃圾結果。

因此 root cause 被重新定性成四層：

1. 主因：語料缺口 / 不可答查詢。目標內容不在索引裡。
2. 次因：agent 沒有 give-up discipline。搜尋多次無關時不知道該誠實說 KB 沒有。
3. 共犯：Layer 2 context amnesia。超過 4 筆後早期 tool results 被裁掉，模型看不到自己查過。
4. 放大器：parallel tool-call budget overflow。

`fix_plan.md` 重新排序後的修復計畫是：

- Phase 0 instrumentation：已完成。
- Phase 1 give-up discipline：最高槓桿。連續 N 次檢索不相關或 empty，就停止搜尋並說 KB 裡沒有此內容。
- Phase 2 修評測題：embedding case 原本期待 `rag_get_context`，但資料不在索引裡，應改成「優雅放棄」測試或換成語料中真有的主題。
- Phase 3 對齊上限與視窗：讓真正 call 次數硬上限等於 prompt 裡看得到的 tool result 數量，避免裁掉仍可能被使用的結果。
- Phase 4 實驗決定 N。
- Phase 5 timeout、eval progress、eval bash auto deny。

後來 `63a09de` 修掉 parallel overflow 並保留同 turn 內全部 tool results，但 5/31 又發現 Turn 2 還能跑到 6 次工具。進一步 instrument 後，推翻了 `_tool_interaction_count` 沒對齊的假設。真正尾巴是 exhausted 分支呼叫 unbound/raw model 後，raw response 仍可能帶 parsed `tool_calls`，舊程式沒有清掉，`route_after_agent` 會繼續送進 tools。`4333fa4` 的修法是 exhausted 分支也套 `_cap_tool_calls(..., 0)`。修後 deterministic mock 只剩 `raw_counts [4]`。

6/15 這條線又落地幾件事：

- `6424539` 在 base tool workflow prompt 加 graceful give-up rule。
- `bea1d6e` 把 C1 embedding case 重新分類成 graceful give-up。
- `468126a` 在 C1 routing 加 give-up answer scoring 與 progress。
- `050e07e` 測 multi-round parallel tool calls 不突破 cap。
- `ad72c34` 記錄 `agent_max_tool_interactions=4` 的資料依據。
- `48cd5b2` 同步 legacy BehaviorEvaluator embedding case。
- `5070cc3` 加 C1/behavior spec drift guard。

6/15 eval report 顯示這個問題已有明顯改善：C1 8 題全評，7 題通過。唯一失敗仍是 embedding graceful-give-up 題。它已經不再碰 bash，也沒有對無關結果叫 `rag_get_context`，但實際叫了 4 次工具，rubric 要 1-3 次；最後回答語意接近找不到，但沒有命中 expected not-found regex。也就是說，問題已從 runaway / wrong tool 降級成停止條件與答案格式問題。

## 2026-05-31：Extended Thinking 的 scope 與 temporal leakage

5/31 的 `note/20260531/20260531-extended-thinking-scope-debug.md` 記錄另一個真實 extended thinking case。使用者在 `/thinking extended` + `academic-paper-writing` 下問：

「三月 15 號之前的研究內容，假設要單獨發一篇 paper，符合 ICLR 格式與規範，abstract 應該安排哪些重點？」

使用者補充資料在 `Research_notes`，不要直接寫 abstract，先用中文討論內容安排。原始 CLI trace 顯示，agent 第一輪沒有直接亂寫，而是先追問到 3/15 為止的研究重點摘要、是否已有草稿或全文、輸出語言與字數限制；使用者明確說資料在 `Research_notes`、目前沒有內容、不是要寫 abstract 而是問應放哪些重點後，agent 才開始查資料。這一點很重要，因為它說明 extended mode 的 intake 行為不是完全錯誤：在資料位置不明時它會先問，資料位置明確後會進工具查詢。

agent 後來給出結構化建議，品質本身不差：它先判斷 ICLR 是否適合，指出 ICLR 期待 learning/representation 創新、benchmark、ablation 與 prior work 比較，而使用者到 3/15 的成果更像 legacy bioinformatics tool 的系統性逆向工程與 methodology parity work；接著盤點 PiDNA 原始方法論、PiDNA1/PiDNA2 架構差異、Pref(r) 錯誤、beta 參數、PFM/PWM 流程偏差、PFM flexibility criterion、SelectRatio 50% cap、using3XRange fallback、`ufire.txt.gz` ground truth、12 項 parity gaps 等材料；再把 abstract 分成問題背景、研究目標、方法概述、關鍵發現、貢獻/意義、限制六個板塊。它也明確說如果硬投 ICLR，還缺修正前後定量預測表現、與 DeepPBS 等 modern methods 比較、以及方法可推廣性的額外案例；較適合的投稿方向反而是 Bioinformatics、BMC Bioinformatics 或 JOSS。

這份回答之所以值得記錄，是因為它同時展示了 extended thinking 的價值與問題：它沒有憑空亂寫，輸出結構也符合 academic-writing skill 的守門方向；但 trace 暴露 workflow 層級的缺陷。

第一個問題是工具預算作用域。trace 中第二回合共 12 次工具，實際順序是 `rag_explore + read_file`、兩次 `rag_search`、`recall_history + rag_explore`、兩次 `rag_search`、一次 `rag_explore`、三次 `rag_search`。這個 flat trace 看起來像 5/30 那種工具爆走，但 debug note 判定它不是同一類 bug，因為每個 graph run 的 4 次工具上限其實有生效。

- writer graph run：最多 4 次工具。
- reviewer。
- reviser graph run round 1：最多 4 次工具。
- reviewer。
- reviser graph run round 2：最多 4 次工具。
- final validation 如需要還可能再 4 次。

因此 12 次不是 5/30 那種 per-graph-run cap 失效，而是 extended workflow 的總量自然疊加。使用者直覺期待「整個 extended turn 最多 4 次工具」，但系統實際保證的是「每個 graph run 最多 4 次」。這暴露缺少 extended global per-user-turn tool budget。

第二個問題是 stage handoff 不完整。同一個 `_run_graph_turn()` 內 agent 看得到完整 tool messages，但 writer 到 reviewer/reviser 之間不是同一 graph state 延續。Reviewer 看 evidence trace summary，不看完整 ToolMessage；Reviser 看 raw input、rewritten prompt、previous draft、reviewer JSON 與 instruction，不直接看 Writer 完整 evidence。這使 Reviser 有誘因重新查資料，工具量也會疊加。

第三個問題最隱蔽：temporal scope leakage。使用者明確要求「3/15 之前」，但回答中納入了 `ufire.txt.gz`、109,552 行、SelectRatio 50% cap、using3XRange fallback、new_findings、merged notes、3/26 之後的 parity reassessment 等材料。這些不是 hallucination，而是真實資料，但來自使用者指定時間窗之外。因為後來的 3/26 筆記語意上高度相關，RAG 把它們交給 Writer；Reviewer 只檢查 claim 是否有 evidence，沒有檢查 evidence path/date 是否超界。

這份 debug note 的待辦是：

- Extended mode 增加 global per-user-turn tool budget。
- Reviewer/Reviser handoff 提供更完整但 bounded 的 evidence ledger，至少包含 source path/date/query。
- Rewriter 遇到日期上限時，把它轉成硬性 retrieval instruction，例如只讀 `Research_notes/YYYYMMDD <= 20260315`。
- Reviewer 增加 temporal scope check。
- CLI trace 標示階段，如 `[Writer] calling rag_search`、`[Reviser 1] calling rag_search`。

這些在 6/15 前沒有完全落地，應視為仍待處理的 extended mode 設計債。

## 2026-06-14 至 2026-06-15：複雜度健檢後的結構整理

6/14 起 repo 把 `to_be_solved/` 整理成問題卡與 archive，並把 `agent.md` 改名成 `AGENTS.md`。同時保存了一份 complexity audit：`to_be_solved/archive/deep-research-report.md`。這份報告指出 repo 複雜度不是全面失控，而是有幾個地方出現「功能是真的，但結構比必要更繞」。

報告列出幾個 quick wins：

1. Base tool inventory 有多個 source of truth。`graph.build_graph()`、`ChatSession._all_tool_refs()`、evaluation `tool_inventory()`、RAG adapters 都各自維護工具清單，容易 drift。
2. Active skill state 被序列化兩次。`graph.py` 與 `session.py` 都把 SkillRuntime 轉成幾乎一樣的 dict。
3. LLM access contract 並存兩套：core runtime 走 LangChain chat model factories，但 `BaseLLM` / `OpenRouterLLM` / `OllamaLLM` 又形成另一套 prompt-to-text wrapper。
4. Skill metadata frontmatter parser 手刻，但 repo 已依賴 PyYAML。
5. OpenRouter retry 手寫 backoff，但官方 client 已有 retry。

6/14 到 6/15 的工作在 `note/20260615/report.md` 中被整理成三條主線：第一，整理 agent 的基礎能力邊界，讓工具清單、工具可用性、skill 狀態與 LLM 存取方式都有單一來源；第二，把 C1 embedding 失敗案例重新定義成「找不到資料時要優雅放棄」，並補上可評分的答案規格；第三，建立可以一次跑完 C1-C4 的 dev 評測流程，並記錄當天全量結果。這一階段也把 `to_be_solved/` 重新整理成仍待解問題與 archive，舊 `agent.md` 改名成 `AGENTS.md`，並關閉或移除一批已完成問題卡：`frontmatter-parser-pyyaml`、`base-tool-inventory-single-source`、`agent-history-tool-availability`、`agent-history-recall-user-facing-failure`、`openrouter-retry-cleanup`、`agent-tool-call-runaway-followups`、`llm-access-contract`。這讓 6 月中旬的 repo 狀態從「很多調查散落在 note 和 task card」變成「已修、未修、已歸檔」比較清楚。

6/14 至 6/15 的 commit 幾乎逐項處理：

- `c6f277c` 用 PyYAML parse skill frontmatter，取代手寫 parser。
- `6318493` single-source base tool inventory，新增 `agent/tools/inventory.py`。
- `62fac43` tool availability fallback 從 base inventory derive。
- `ca26bcf` 把 archived history-recall failure scenario 固定成 regression test。
- `b0e675f` centralize skill state serialization，新增 `skill_runtime_to_agent_state()` 之類的單一 helper。
- `e61c990` 把 OpenRouter retries 交給 client。
- `e43baaa` 把 `llm_max_retries` forward 到 thinking role models。
- `f213ada` standardize chat model access，刪除 `agent/llm/base.py`，提供 OpenRouter/Ollama chat model factories 與 `invoke_text()`。
- `e36ef78` 加 full eval runner。
- `ca8d08f` 讓 full runner 只寫 ledger，不產生旁路輸出。
- `5ae47a6` 記錄 full dev eval run。
- `e23d36f` / `fa1cd8d` 寫 6/15 eval report。

`note/20260615/report.md` 詳細說明這批整理的結果。

Base tool inventory 的意義是把 `rag_explore`、`rag_search`、`rag_get_context`、`recall_history`、`read_file`、`bash` 的靜態 metadata、tool instance 建立、tool name list、system prompt tool descriptions、tool workflow policy、evaluation tool taxonomy 收斂到同一處。這直接降低了 prompt 說有工具但 runtime 沒綁、或 eval 用另一套工具 universe 的風險。

Tool availability fallback 也被明確修正。extended thinking 的 writer、rewriter、reviewer 如果沒有傳入 `base_tool_names`，原本可能 render 出空的 available tools，導致角色以為 base tools 都不可用；現在 `render_tool_availability_block()` 在參數是 `None` 時 fallback 到 base inventory，但明確傳入空 list 仍代表沒有工具。同時 `capability_map.yaml` 補上 `rag.search` 與 `history.search` 的差異：`recall_history` 查的是持久化聊天歷史，不是 indexed knowledge base。這正是 5/27 spec 想避免的混淆。

History recall 舊失敗情境也被固定成 deterministic regression test。測試重點包括：

- `recall_history` 可用但 writer 沒先查時，reviewer 要標 `retrieval_not_attempted` 並導向 reviser。
- `recall_history` 查不到資料時，答案要誠實說明歷史不足，並區分 chat history、plan logs、indexed KB。
- active skill 禁用 history tool 時，使用者看到的是 tool policy/settings 說明，不是內部 reviewer 指令或 intake checklist。

LLM access contract 的整理則讓主 agent loop、eval、thinking roles 都走更一致的 LangChain chat model factory。這減少了「同樣是 LLM 呼叫但走不同抽象」的維護成本。

OpenRouter retry 也被收斂成單一 config：新增 `AgentConfig.llm_max_retries`，`get_chat_model()` 把它傳給 `ChatOpenAI`，prompt-to-text 呼叫改依賴 client retry，不再維護 `_call_with_retry`、`MAX_RETRIES`、`INITIAL_DELAY`。Thinking role models 也補上同一個 retry 設定，避免 extended roles 還停在硬編碼 retry 值。

C1 graceful-give-up 的新規格也在這裡定型。原本 `rag_context_embedding_followup` 被視為應使用 `rag_get_context` 的 case；但既然 indexed KB 沒有 embedding module 資料，正確行為應改成 bounded search 後誠實說資料不足。新 gold 允許第一個工具是 `rag_search` 或 `rag_explore`，必須至少用 `rag_search`，禁止 `rag_get_context`、history、web、file、bash，工具次數限制 1-3，最終答案必須同時提到 KB/indexed knowledge base 以及 not-found/insufficient evidence 訊號。Legacy `BehaviorEvaluator` 也同步成這套規格，並加 semantic parity test，避免同一 case id 在兩套 eval 中語意漂移。

Full eval runner `scripts/run_full_eval.sh` 讓 C1-C4 可以一次跑完。它預設跑 dev split，跑前拒絕 dirty worktree，可用 `ALLOW_SKIPS=1` 控 C1 skip，可用 `NO_MCP=1` 關 MCP，結果只寫既有 ledger。

當天的 ledger 寫入明確記錄在 `eval/runs/c1.jsonl`、`eval/runs/c2.jsonl`、`eval/runs/c3.jsonl`、`eval/runs/c4.jsonl`，逐題明細是 `eval/runs/details/c1-20260615T064950Z-6d27a65d.json`、`c2-20260615T064955Z-ef938234.json`、`c3-20260615T065007Z-c039bf54.json`、`c4-20260615T065042Z-49804861.json`。這補上了舊 eval 最大缺口之一：結果不是覆蓋式 JSON，而是 append-only ledger 加 per-run details。

6/15 的 dev eval 結果是：

- C1：8 題全評，7/8 通過，`routing_accuracy=0.875`。唯一失敗是 graceful-give-up embedding 題，已從 runaway 降級成 count 與 answer regex 問題。
- C2：recall@5 = 1.0、MRR = 0.833、nDCG@5 = 0.877，和 5/31 相同。正解都找得到，但 `pidna2/README.md` 被 `web/backend/pyproject.toml` 擠到第 2，排序仍可改善。
- C3：validator 與 session 滿分，reviewer 仍有 `c3b-clean-draft-pass` false block，把乾淨稿誤判成 user_input_missing。
- C4：task_success_rate = 0.5，answer_requirements_accuracy 從 0.5 升到 1.0，但 history codename 題仍失敗，因為沒有呼叫 required tool `recall_history`。

C4 的失敗被報告指出是 rubric conflict：dataset 先在 visible context 裡說「Remember that the deployment codename is Blue Lantern.」，下一輪立刻問 codename。現行工具政策寫明內容已在目前對話可見時不要呼叫 `recall_history`，但 C4 rubric 又要求必叫 `recall_history`。因此這題要先決定測的是 persisted history 還是 visible context；若是前者，dataset 應把記憶放到不可見 history；若是後者，required tool 應放寬。

6/15 的結論是：工具系統、history recall regression、LLM access、retry、eval runner 都比之前乾淨，但仍有幾個明確待辦：C1 graceful give-up 還沒完全達標，C2 sorting 要診斷，C3 reviewer false block 要修，C4 history rubric 要釐清，extended global budget 與 temporal scope check 仍未完成。

## 截至 2026-06-15 的 repo 狀態

到掃描基準 `fa1cd8d` 時，這個 repo 已經是一個 agent host，而不是單純 RAG demo。它的主要組件是：

- `agent/`：LangGraph agent runtime、session、state、history、MCP、skills、tools、evaluation、LLM access。
- `../rag`：外部 sibling dependency，提供 RAG core indexing/retrieval/storage。
- `skills/`：內建 skill，例如 `academic-paper-writing` 與 `_prompt-master`。
- `eval/`：C1-C4 datasets 與 append-only run ledger。
- `note/`、`to_be_solved/`：重要調查、報告、問題卡與 archived plans。

主要能力包括：

- RAG tools：explore/search/get_context。
- History RAG：`recall_history` 查 persistent chat history。
- Local file tool：`read_file`，受 skill scope 與 sensitive file policy 保護。
- Bash tool：需使用者批准，且可被 skill policy 禁用。
- MCP tools：Web Search、GitHub，透過 async stdio loader 載入。
- Skill runtime：manifest、capability broker、tool policy、references、task modes、output validation。
- Extended thinking：prompt rewrite、writer graph、reviewer、reviser、repair model、DRAFT/REBUTTAL 分離、final skill validation。
- Evaluation：C1 routing、C2 retrieval、C3 validator/reviewer/session、C4 checklist、ledger、repro metadata。

主要已解問題：

- 初期 KMS 無 graph runtime：已轉 LangGraph。
- RAG 與 agent 邊界混雜：已拆 `rag` 與 `agent`。
- MCP noisy stdout/stderr：已做 stdio filtering 與 failure-tolerant loading。
- 長期記憶只靠 prompt/compaction：已改成 history vector store 與 `recall_history`。
- active skill 下 history tool 被遮蔽：已補 `history.search` capability、tool availability context、review routing。
- evaluation 舊數字不可信：已重建成 C1-C4 claims、datasets、ledger、fingerprints。
- tool-call runaway：已修 parallel overflow、raw response leakage、context amnesia，並加入 give-up discipline。
- base tool inventory 多 source of truth：已 single-source。
- skill state serializer 重複：已集中。
- LLM retry 與 access contract 分裂：已整理。

主要未解或半解問題：

- C1 graceful give-up 還需讓 agent 在 1-3 次無關搜尋後穩定停下，且回答穩定包含 KB/insufficient evidence 訊號。
- C2 排序仍可改善，尤其語意相關但非正解文件會擠到第 1。
- C3 reviewer 對乾淨稿過度保守，會 false block。
- C4 history codename 題的 rubric 與工具政策衝突，需要重設測試目標。
- Extended mode 沒有 global per-user-turn tool budget。
- Extended writer/reviser handoff 只有壓縮 trace，可能導致 reviser 重查。
- Temporal scope leakage 尚未用 retrieval filters 或 reviewer provenance checks 系統性解決。
- Public benchmark spike 只是初步 BEIR adapter，還不是正式外部對標體系。

## 素材對應表

本編年史使用了以下保存材料：

- `record/project-turning-points.md`：作為時間骨架。
- `record/relevant-commits.md`：用來定位 refactor、debug、major feature、plan/spec/report commit。
- `record/all-commits.tsv`：用來確認完整 commit 順序。
- `record/source-documents/20260411-evaluation-run-report.md`：4/11 evaluation failure 與 next steps。
- `record/source-documents/20260413-agent-state-cleanup.md`：state cleanup 決策與 `agent_max_tool_interactions=4` 的早期背景。
- `record/source-documents/20260418-repo-split-plan.md`：RAG/core 與 agent/app 拆分計畫。
- `record/source-documents/20260420-mcp-setup.md`：MCP 兩層配置、Web Search/GitHub MCP、failure behavior。
- `record/source-documents/20260524-thinking-extended-plan-v3-4.md`：extended thinking v3.4 設計。
- `record/source-documents/20260524-problem.md`：history recall 在 academic skill 下失敗的原始情境。
- `record/source-documents/20260527-evaluator-plan.md`：C1-C4 evaluator 重建計畫。
- `record/source-documents/20260527-history-tool-availability-spec.md`：history/tool availability 修復 spec。
- `record/source-documents/20260527-p0-5-history-query-diagnostic.md`：本機 chat_history 查詢診斷。
- `record/source-documents/20260530-c1-routing-findings.md`：C1 首跑發現。
- `record/source-documents/20260530-tool-call-runaway-debug.md`：runaway 偵錯細節。
- `record/source-documents/20260530-tool-call-runaway-fix-plan.md`：runaway fix plan 與後續 Phase 3 證據。
- `record/source-documents/20260531-eval-claim-run.md`：C1-C4 dev run 與 dataset inventory。
- `record/source-documents/20260531-extended-thinking-example.md`：extended thinking 實際 CLI trace。
- `record/source-documents/20260531-extended-thinking-scope-debug.md`：extended budget 與 temporal leakage 分析。
- `record/source-documents/20260614-complexity-audit.md`：複雜度健檢與 quick wins。
- `record/source-documents/20260615-skill-state-single-serializer.md`：skill state serializer task card。
- `record/source-documents/20260615-recent-changes-eval-report.md`：6/14-6/15 整理與 dev eval 結果。

這些素材合在一起呈現出一條很清楚的演進線：專案每次大改幾乎都不是為了抽象而抽象，而是在回答一個具體失敗。RAG 查不準，就加入 metadata 與 context tools；自寫 loop 撐不住，就換 LangGraph；state 長太大，就壓縮與裁切；裁切造成失憶，就對齊 tool budget 與 visible results；history tool 被 skill policy 遮掉，就建立 tool availability context；eval 不可信，就重建 C1-C4；工具清單 drift，就 single-source inventory。這就是此 repo 到 2026-06-15 為止的主要歷史。
