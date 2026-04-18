# `langgraph` branch — 架構說明

目前 repo root 是一個 workspace，底下有兩個 peer package：

- `rag/`：建索引、儲存、檢索、public retrieval API
- `agent/`：LangGraph agent、tool adapters、chat/eval CLI

整個系統分成兩條獨立的 pipeline：**Ingest**（建索引）和 **Chat**（查詢），它們唯一的交會點是磁碟上的 store。

---

## Pipeline 1：Ingest（建索引）

![Ingest Pipeline](ingest_pipeline.svg)

這條 pipeline 只跑一次（或資料更新時重跑），目的是把 repo 裡的檔案變成可搜尋的 chunks。流程四步：

1. `_collect_folders()` 掃描 repo，按 `TEXT_EXTENSIONS` 過濾、按 `SKIP_DIRS` 排除，把檔案按**所在目錄**分組。

2. `LLMTagger.tag()` 對每個目錄呼叫一次 LLM，輸入是目錄路徑 + 檔名 + 檔案預覽（前幾行），輸出是 2-4 個 tags 和一段 summary。這些存進 `folder_meta.json`。

3. `TokenChunker` 把每個檔案切成 1200 token 的 chunk。**關鍵改動在這裡**：切完之後，每個 chunk 的 metadata 會從 `folder_meta.json` 繼承 `category`（第一個 tag）和 `tags`（全部 tags 的 JSON string）。這就是 spec 裡說的「tags 下沉」。

4. 所有 chunk 寫入**同一個** ChromaDB collection `"knowledge"`（舊版是按頂層目錄分成 6 個 collection），同時備份到 `raw.json`。

最終磁碟上只有三個東西：`chroma.sqlite3`（向量索引）、`raw.json`（全文備份）、`folder_meta.json`（目錄摘要）。

---

## Pipeline 2：Chat（Agent 查詢）

![Agent Loop](agent_loop.svg)

整個 `agent/graph.py` 只定義了兩個 node 和一條 conditional edge，LangGraph 會自動處理循環：

**Agent node**：把目前所有 messages（system prompt + 對話歷史 + tool 結果）丟給 LLM。LLM 回傳的東西只有兩種可能——純文字回答，或者 tool call 請求。

**tools_condition**：LangGraph 內建的 router。它看 LLM 的回應裡有沒有 tool calls。有 → 走 ToolNode；沒有 → 結束這輪，回傳文字給使用者。

**ToolNode**：LangGraph 內建的執行器。它收到 LLM 的 tool call，根據 function name dispatch 到對應的 tool，執行完把結果塞回 messages，然後**自動回到 Agent node**。這就形成了循環。

三個 tool 各自讀不同的資料：

- `explore` 讀 `folder_meta.json`，不碰向量資料庫。它回傳的是 categories 清單、tags 清單、date 範圍、每個目錄的 summary。Agent 用這個來了解「知識庫裡有什麼」。

- `search` 打 ChromaDB 做語義搜尋，支援 `category`、`file_type`、`date_from`/`date_to` 四種 metadata filter。回傳的每筆結果帶有 `pid` 和 `chunk_id`。

- `get_context` 讀 `raw.json`，用 `pid` 找到同一個檔案的所有 chunks，然後回傳目標 chunk 前後 N 個 chunk。這解決的是「搜到一段相關的，但想看更多前後文」的需求。

**MemorySaver** 是 LangGraph 的 checkpointer。它把每一輪對話的完整 messages list 存在記憶體裡，所以下一輪對話 LLM 能看到之前聊過什麼。但 process 關掉就沒了。

---

## 舊版 vs 新版的關鍵差異

舊版的 `chat.py` 有 30 行在手動做 LangGraph 現在自動處理的事：parse tool calls → 組 assistant message → 逐一執行 tool → 組 tool result message → append 回 messages → 重新呼叫 LLM。新版把這些全交給 `StateGraph` + `ToolNode` + `tools_condition`，`agent/session.py` 的 `turn()` 只剩一行 `self.graph.invoke()`。

舊版 LLM 模組需要 `ChatResponse`、`ToolCall` dataclass 和 `BaseLLM.chat()` 方法來手動解析 OpenAI 的 tool call 格式。新版用 LangChain 的 `ChatOpenAI`，它原生支援 `.bind_tools()`，所以這些全部可以刪掉。`OpenRouterLLM` 只留 `invoke()` 給 tagger 用。
