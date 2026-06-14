# Minervamuses app 複雜度健檢報告

## Repo 總覽

這個 repo 是 **public**，主語言為 **Python 100%**，目前 GitHub 顯示有 **249 commits**；repo 根目錄可見 `agent/`、`env/`、`eval/`、`note/`、`skills/`、`tests/`，以及 `pyproject.toml`、`EVALUATOR_PLAN.md`、`SKILLS_GUIDE.md` 等檔案。從 `pyproject.toml` 看，專案使用 **Poetry** 管理相依，Python 版本範圍是 `>=3.12,<3.14`，核心依賴包含 `langgraph`、`langchain-core`、`langchain-openai`、`langchain-mcp-adapters`、`mcp`、`openai`、`ollama`、`pydantic`、`python-dotenv`，另外還依賴一個 sibling package `rag = { path = "../rag", develop = true }`。citeturn34view0turn37view0

從 `agent/` 的目錄結構看，核心模組分成 `adapters/`、`evaluation/`、`history_rag/`、`llm/`、`skills/`、`tools/`，這表示此 repo 不是單純聊天 CLI，而是帶有 **LangGraph agent runtime、skills policy、history retrieval、MCP tool loading、evaluation harness** 的完整 agent host。`tests/` 也覆蓋了 skill loader、history、mcp、evaluation、chat CLI 等多個面向，顯示作者確實把這些能力視為正式功能，而不是試驗性腳本。citeturn5view0turn12view2turn13view0turn13view1turn13view2turn14view0turn6view2

我優先深讀的區域是：`agent/session.py`、`agent/graph.py`、`agent/thinking.py`、`agent/skills/*`、`agent/mcp.py`、`agent/tools/*`、`agent/history_rag/*`、`agent/llm/*`。其中 `agent/session.py` 已達 **936 lines / 35.3 KB**，`agent/graph.py` 為 **232 lines / 8.99 KB**；這兩個檔案正好也是功能交界最多、最容易出現多重 abstraction 與 source of truth 漂移的地方。citeturn33view0turn7view0

整體判斷上，這個 repo 的複雜度不是全面失控，而是呈現出一種很典型的 staff-level agent host 演進軌跡：**多數複雜度是有功能需求支撐的，但有幾個地方已經出現「功能是真的，結構卻比必要更繞」的跡象**。以下我只列我認為證據最強、而且改了確實有收益的項目。citeturn10view0turn26view0turn27view1turn19view0

## 面向一：不必要的複雜度

### 基礎工具清單有多個 source of truth

**位置**：`agent/session.py:334-347` `ChatSession._all_tool_refs()`；`agent/graph.py:82-112` `build_graph()`；`agent/evaluation/base.py` `tool_inventory()`；`agent/adapters/langchain/rag_tools.py` `DEFAULT_RAG_TOOL_NAMES` / `create_rag_tools()`。citeturn26view0turn10view0turn20view0turn27view3

**現況**：`graph.build_graph()` 會真正建立 base tools：RAG tools、`recall_history`、`read_file`、`bash`；但 `ChatSession._all_tool_refs()` 又另外硬編一份 `["rag_explore", "rag_search", "rag_get_context", "recall_history", "read_file", "bash"]`；同時 `evaluation.base.tool_inventory()` 也再組一次工具清單，還要依賴 `DEFAULT_RAG_TOOL_NAMES`。citeturn10view0turn26view0turn20view0turn27view3

**問題**：這是典型的 **configuration / inventory drift** 風險。工具集合其實是 runtime 的核心事實，但現在至少在 **graph runtime、session skill activation、evaluation** 三個地方各自維護。只要未來新增、改名、或改變某個 base tool 的綁定來源，這三處就可能不同步。這種複雜度沒有換來額外的正確性或效能，純粹是多個 source of truth 帶來的維護成本。這裡的判斷是架構推論，但它直接建立在三處重複定義的實際程式碼上。citeturn10view0turn26view0turn20view0turn27view3

**建議**：把「base tool inventory」收斂成單一 helper。最簡單的做法是抽出一個像 `build_base_tools(config, history_store)` 與 `base_tool_names(config, history_store)` 的共用函式，讓 `graph`、`session`、`evaluation` 都從同一處拿工具與名稱。`_all_tool_refs()` 最好也不要再自己硬編字串。citeturn10view0turn26view0turn20view0turn27view3

**效益與取捨**：改動成本低，但能明顯降低未來 drift bug；尤其這個 repo 已有 skills policy 與 tool availability block，只要名稱不同步，症狀通常不會立刻爆炸，而是變成很難追的「prompt 說有 / graph 沒綁，或 evaluation 算錯 inventory」類問題。這類問題很浪費高階工程師時間，我認為值得改。citeturn27view1turn26view0turn10view0

**Severity**：Med ｜ **Effort**：Low ｜ **Confidence**：High

### Active skill state 被序列化了兩次，而且內容幾乎完全重複

**位置**：`agent/graph.py:18-33` `_skill_runtime_state()`；`agent/session.py:349-365` `ChatSession._active_skill_state()`。citeturn10view0turn26view0

**現況**：這兩個函式都把 active skill runtime 轉成幾乎同一份 dict：`active_skill`、`skill_root`、`skill_instructions`、`loaded_references`、`task_mode`、`allowed_tools`、`denied_tools`、`tool_policy_active`，以及 validation-related flags。citeturn10view0turn26view0

**問題**：這不是功能抽象，而是 **representation duplication**。現在 skill state 的 schema 雖然靠 `AgentState` 暗示，但真正的組裝邏輯分散在兩處；之後只要有人新增一個 state key，很容易只改到其中一處。因為這份 state 又直接牽動 graph routing 與 tool policy，重複定義的代價不小。更簡單的做法完全可以達成相同功能。citeturn10view0turn16view3turn26view0

**建議**：把這段統一成單一 serializer，例如 `SkillRuntime.to_agent_state()`，或至少在 `agent/state.py` 放一個唯一 helper，`graph` 與 `session` 都走那個 helper。citeturn16view3turn10view0turn26view0

**效益與取捨**：這屬於小改動、穩定收益。它不會立刻改善效能，但可以降低 future change risk，也讓 skill/state 的責任邊界更清楚。因為改法很局部，我傾向列為 quick win。citeturn10view0turn26view0

**Severity**：Low ｜ **Effort**：Low ｜ **Confidence**：High

### 核心 runtime 與 LLM adapter 採用了兩套不同的模型存取契約

**位置**：`agent/graph.py:101` `build_graph()` 使用 `get_chat_model(config)`；`agent/session.py:454-460` `ChatSession._get_thinking_role_model()` 使用 `get_chat_model_for_role()`；另一方面 `agent/llm/base.py` 定義 `BaseLLM`，`agent/llm/openrouter.py` 定義 `OpenRouterLLM(BaseLLM)`，`agent/llm/ollama.py` 定義 `OllamaLLM(BaseLLM)`。測試 `tests/test_openrouter_model.py` 目前也是測 `get_chat_model()`，不是 `OpenRouterLLM`。citeturn10view0turn26view0turn27view4turn28view0turn29view0turn30view0

**現況**：core agent runtime 走的是 `ChatOpenAI` factory 路線，extended-thinking role model 也是 `ChatOpenAI` factory；但 `llm/base.py` / `OpenRouterLLM` / `OllamaLLM` 又另外形成一套 `prompt -> text` 的抽象層。至少在我實際讀到的核心執行路徑裡，這兩套路徑是並存而非統一的。citeturn10view0turn26view0turn27view4turn28view0turn29view0turn30view0

**問題**：這讓讀 code 的人必須先搞懂「哪裡用 LangChain chat model、哪裡用 BaseLLM wrapper」，但程式本身沒有提供明確的 architectural boundary。這不是嚴重 bug，但它會增加 API surface 與認知負擔，而且目前看起來 core path 更偏向 factory / LangChain 介面，wrapper hierarchy 反而像旁支。[需確認] `cli/` 或未完整檢視的 evaluation code 是否仍重度依賴 `BaseLLM`；如果有，問題會從「可刪除」降級成「應隔離」。citeturn10view0turn26view0turn27view4turn28view0turn29view0

**建議**：做一次明確決策：要嘛統一用 LangChain chat model factories，將 `BaseLLM` wrappers 移到 legacy/eval-only 區域；要嘛反過來讓 core path 也走同一個 adapter contract。現在最不好的狀態就是兩套契約長期並存。citeturn27view4turn28view0turn29view0turn10view0turn26view0

**效益與取捨**：這個改動的好處是長期可維護性，而不是短期功能增益，所以值不值得做，要看你們是否還打算持續擴展 provider / model 層。如果這個 repo 會繼續長大，我認為值得；如果目前功能已穩定，則可先做命名與邊界整理，不急著大刪。citeturn27view4turn28view0turn29view0

**Severity**：Med ｜ **Effort**：Med ｜ **Confidence**：Med

## 面向二：重造輪子

### 手刻 YAML frontmatter parser，但同一個 package 其實已經在用 PyYAML

**位置**：`agent/skills/metadata.py:56-130` `_read_skill_metadata()` / `_parse_frontmatter()`；對照 `agent/skills/runtime.py:244-253` `load_skill_manifest()` 與 `agent/skills/broker.py:28-38` `load_capability_map()` 都直接使用 `yaml.safe_load()`。citeturn27view0turn27view1turn27view2

**現況**：`_parse_frontmatter()` 自己用字串掃描、`:` split、continuation lines、quote stripping 去解析 `SKILL.md` 開頭的 frontmatter，而且註解明白寫著它只支援「minimal subset」。但同一個 skills package 裡，`runtime.py` 與 `broker.py` 已經直接用 `yaml.safe_load()` 解析 YAML。citeturn27view0turn27view1turn27view2

**問題**：這同時滿足「有成熟替代方案」與「手刻版沒有明顯優勢」兩個條件。替代方案就是 repo 內已經存在、也已經被依賴的 **PyYAML `yaml.safe_load()`**；而手刻 parser 的唯一明示特色是只支援子集，不是更快、更安全、也不是更完整。對這種 frontmatter 場景，手刻 parser 只會增加 edge case 與維護成本。citeturn27view0turn27view1turn27view2

**建議**：把 frontmatter block 先切出來，再丟給 `yaml.safe_load()`；如果你要的是 markdown-aware 行為，也可以改用成熟 frontmatter library，但在目前 repo 已經有 PyYAML 的前提下，先用 `yaml.safe_load()` 就足夠。citeturn27view0turn27view1turn27view2

**效益與取捨**：這是非常典型的 quick win。改完後 code 會更短、語意更清楚，也比較不容易在多行值、特殊字元、合法 YAML 但超出手刻子集的輸入上翻車。除非你們有刻意禁止完整 YAML 語法的理由，否則很值得改。citeturn27view0turn27view1turn27view2

**Severity**：Med ｜ **Effort**：Low ｜ **Confidence**：High

### `OpenRouterLLM` 自己手寫 retry/backoff，但官方 client 已經提供 retry 機制

**位置**：`agent/llm/openrouter.py:57-70` `_call_with_retry()`，並由 `OpenRouterLLM.invoke()` 使用。citeturn28view0

**現況**：`OpenRouterLLM` 以 `for` 迴圈配上 `time.sleep()` 手刻 exponential backoff，專門處理 `RateLimitError`，最後再由 `invoke()` 走 `_call_with_retry()`。同檔案裡另一條路徑 `get_chat_model()` 則直接把 `max_retries=10` 交給 `ChatOpenAI`。citeturn28view0

**問題**：官方 `openai-python` 已經說明 client 內建 automatic retries，並可透過 `OpenAI(max_retries=...)` 或 `client.with_options(max_retries=...)` 調整；因此這裡的手刻 retry 屬於明顯的 wheel-reinvention。更重要的是，檔案本身並沒有展示這個手刻版比官方機制多出什麼 domain-specific 優勢，例如特殊錯誤分類、observability、或 provider-specific backoff policy。citeturn28view0turn32search0

**建議**：優先改成官方 client 的 `max_retries` 設定；如果你們真的需要比官方更長的 backoff，再只包一層最小增量，而不是整段自己維護。citeturn32search0turn28view0

**效益與取捨**：這個修改的直接好處是減少自維護 HTTP retry 邏輯，也讓 repo 的 retry 行為更貼近 upstream SDK 的預設與更新。唯一取捨是如果你們刻意想把 backoff 拉得比官方更長，可能要再測一次實際吞吐與 rate-limit 行為，但整體仍然偏值得。citeturn32search0turn28view0

**Severity**：Low ｜ **Effort**：Low ｜ **Confidence**：High

## 優先處理清單

### Quick wins

- **先拔掉手刻 frontmatter parser**：把 `agent/skills/metadata.py::_parse_frontmatter()` 改成 frontmatter block + `yaml.safe_load()`；這是最低成本、最直接縮小 maintenance surface 的修改。citeturn27view0turn27view1turn27view2
- **把 active skill state serializer 收斂成單一 helper**：消除 `agent/graph.py::_skill_runtime_state()` 與 `agent/session.py::ChatSession._active_skill_state()` 的重複。citeturn10view0turn26view0
- **把 base tool inventory 收斂成單一 source of truth**：不要再讓 `session`、`graph`、`evaluation` 各自拼一次工具清單。citeturn10view0turn20view0turn26view0turn27view3
- **把 `OpenRouterLLM` 改用官方 retry 設定**：此項風險低，回報也清楚。citeturn28view0turn32search0

### 較大型 refactor

- **統一 LLM access contract**：決定 core runtime 與非 core path 到底要走 `ChatOpenAI` factories，還是走 `BaseLLM` wrapper contract；如果 wrapper 只剩少數 legacy/eval 用途，就應顯式隔離或縮減。[需確認] 這項需要先補看 `agent/cli/` 與剩餘 evaluation path 的實際引用情況，再決定是刪、搬、還是統一。citeturn10view0turn26view0turn27view4turn28view0turn29view0

## 看似複雜但其實合理

`skills` 的 capability resolution、tool policy、以及 `PolicyToolNode` 這一整串，看起來機制不少，但我判定 **合理且有需求支撐**。原因是 repo 裡真的有 skill manifest：例如 `academic-paper-writing/manifest.yaml` 會宣告 `required` 與 `optional` capabilities，並且明確 `disallow: bash`；`broker.resolve_capabilities()` 會做 capability 到 tool 的解析；`PolicyToolNode` 會在實際執行時拒絕超出 policy 的 tool call；`graph.build_graph()` 也會依 state 動態選綁工具。這不是為 pattern 而 pattern，而是為了讓「prompt 看到的工具限制」與「runtime 真正允許的工具」保持一致。citeturn35view1turn27view2turn19view1turn10view0

`agent/mcp.py` 裡把 stdio MCP server 的 stdout 用 shell pipeline 過濾成只保留 `{` 開頭 JSON 行，初看很 hack；但這段其實有清楚的 operational rationale：作者直接在註解中說明有些 upstream MCP server 會把 banner/notice 打到 stdout，導致 stdio transport JSON parse 失敗與 `ExceptionGroup[BrokenResourceError]` 類問題。因此這裡不是 over-design，而是對上游不穩定行為的防禦性修補。citeturn19view0

`read_file` 的 skill-root confinement、sensitive-path denylist、1 MB guardrail 也屬於合理複雜度。因為這個 agent 真的提供本機讀檔能力，若沒有 `Path.resolve()`、skill root escape 檢查、以及 `.env` / `.ssh` / `id_rsa` 類敏感檔阻擋，風險會比現在高得多。這段邏輯有明確安全需求支撐，不建議為了「簡單」而拿掉。citeturn17view3

`extended thinking` 流程很長，但我也不建議直接把它當成 over-design。repo 裡除了 `ChatSession` 的 `normal/extended` 模式切換外，`AgentConfig` 還有獨立的 reviewer / rewrite / repair model slots，並且真的存在 `_prompt-master` skill，代表這不是半成品分支，而是被當成功能面經營的能力。在這個前提下，我會把它視為「昂貴但有產品目的」的複雜度，而不是無謂複雜度。citeturn26view0turn16view0turn35view2

## 開放問題與限制

這份報告是建立在我**直接讀到的 public repo 實際程式碼**上，但我沒有完整展開所有 `agent/cli/` 檔案，也沒有逐一抓取每個 test file；因此凡是涉及「某 abstraction 是否完全未被其他模組使用」的判斷，我都只給到 **Med confidence**，並在文中標記了 [需確認]。citeturn34view0turn6view2

另外，這個 repo 依賴一個 sibling package `rag = { path = "../rag" }`。某些看起來像重複或保守設計的地方，可能其實是在配合 `rag` 的 API 邊界或相容性需求；由於這次沒有同時審 `../rag`，我避免把這類推論寫成強斷言。citeturn37view0