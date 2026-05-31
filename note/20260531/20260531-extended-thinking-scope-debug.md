# Extended thinking scope debug

日期:2026-05-31
相關檔案:`note/20260531/example.md`

## 一句話

今天用 `example.md` 回看一次 `/thinking extended` + `academic-paper-writing` 的實際行為後,確認這次回覆品質本身不差,但暴露兩個 workflow 層級的問題:extended mode 的工具預算是每個 graph run 各自計算,不是整個 user turn 共用;另外 RAG 檢索沒有把「3/15 之前」轉成硬性 retrieval scope,導致後來的 3/26 筆記混入答案。

## 原始情境

使用者在 CLI 中:

- 切換 `/thinking extended`
- 啟用 `academic-paper-writing`
- 問「三月 15 號之前的研究內容,如果要單獨發一篇 paper,符合 ICLR 格式與規範,abstract 應該安排哪些重點」
- 補充資料在 `Research_notes`,不要直接寫 abstract,先用中文討論內容安排

結果 agent 先正確追問資料位置/草稿/語言限制;補充後實際查了資料並給出結構化建議。輸出包含 ICLR fit 判斷、abstract 板塊安排、缺口與替代期刊建議。

## 觀察到的工具 trace

`example.md` 中第二回合工具呼叫共 12 次:

```text
rag_explore + read_file
rag_search + rag_search
recall_history + rag_explore
rag_search + rag_search
rag_explore
rag_search + rag_search + rag_search
```

這看起來像「工具呼叫又爆走」,但和 2026-05-30 的 runaway bug 不完全同類。

## 診斷 1:4 次工具上限有生效,但作用域是單次 graph run

目前設定:

- `agent_max_tool_interactions = 4`
- graph 內用 `ToolMessage` 數計算已用工具結果
- parallel tool calls 會依剩餘額度裁切
- exhausted/raw path 的 tool calls 也會裁成 0
- 同一 graph run 內全部 tool results 會保留,不再製造 turn 內失憶

但是 extended mode 不是單一 graph run。它的實際流程是:

```text
rewriter
writer graph run      # 最多 4 次工具
reviewer
reviser graph run     # 最多 4 次工具
reviewer
reviser graph run     # 最多 4 次工具
final skill validation retry, if needed  # 可能再最多 4 次工具
```

因此 `example.md` 的 12 次工具呼叫很可能是:

```text
writer 4 + reviser round 1 4 + reviser round 2 4 = 12
```

結論:這次不像是 per-graph-run 預算失效,而是 extended workflow 的總預算天然會疊加。

## 診斷 2:跨 writer/reviser 階段沒有完整共享工具上下文

同一個 `_run_graph_turn()` 內,agent 看得到自己前面叫過的完整 `ToolMessage`。這是已修好的部分。

但 extended 的階段交接不是同一個 graph state 延續:

| 階段 | 看得見 | 看不見 |
|------|--------|--------|
| writer | prompt history、active skill、tool availability、rewritten prompt、自己同 run 內完整工具結果 | 後續 reviewer/reviser 尚未發生 |
| reviewer | raw input、rewritten prompt、draft、active skill context、tool availability、工具 trace 摘要 | 完整 ToolMessage |
| reviser | raw input、rewritten prompt、previous draft、reviewer JSON、reviser instruction | writer 的完整 ToolMessage 與完整 evidence trace |

所以先前「同一 turn 工具結果被裁掉導致失憶」的 bug 已經修掉;但 extended 仍有另一種設計限制:writer 到 reviser 之間只用 draft/reviewer feedback/壓縮 trace 交接,不是完整工具記憶共享。這會讓 reviser 有誘因重新查資料。

## 診斷 3:日期違規不是 hallucination,而是 temporal scope leakage

回答最大問題不是憑空捏造,而是用了真資料但超出使用者指定時間窗。

使用者明確說「三月 15 號之前」。但回覆中納入了:

- `ufire.txt.gz`
- `109,552 行`
- `SelectRatio 50% cap`
- `using3XRange fallback`
- `new_findings`
- `merged notes`
- 3/26 之後的 parity reassessment/revision 結論

檢索確認:這些主要出現在 `Research_notes/20260326` 之後,不是 3/15 前已知成果。

可能錯誤路徑:

1. rewriter 沒把「3/15 之前」改寫成硬性檔案/date filter。
2. writer 對整個 `Research_notes` 做語意搜尋。
3. 3/26 筆記大量回顧 0315 問題,語意上高度相關。
4. RAG 把後來整理得更完整的資料交給 writer。
5. reviewer 只檢查 claim 是否有 evidence,沒有檢查 evidence 是否落在指定日期範圍內。

這種錯誤可以定義為:

> Temporal scope leakage through semantically relevant later notes.

它比 hallucination 隱蔽,因為 reviewer 的 claim-evidence alignment 會通過。

## 這次不是什麼問題

- 不是單純模型品質差:輸出結構、venue fit、不要直接寫 abstract、placeholder 防捏造都做得不錯。
- 不是 2026-05-30 那個同 graph run 內 ToolMessage 被裁掉的 bug 復發。
- 不是 per-graph-run 工具上限沒咬住;12 次剛好符合 extended 多 graph run 疊加。

## 真正問題

1. **Extended global budget 缺失**
   - 目前 `agent_max_tool_interactions=4` 是每個 `_run_graph_turn()` 的上限。
   - extended 一個 user turn 可能開多個 `_run_graph_turn()`。
   - 使用者直覺期待的「整個 extended turn 最多 4 次工具」目前不存在。

2. **Extended stage handoff 不完整**
   - reviewer 只看壓縮工具摘要。
   - reviser 不直接看 writer 的完整 ToolMessage。
   - 這會讓 reviser 重新查資料,造成工具量疊加。

3. **Temporal retrieval policy 缺失**
   - 指定日期上限時,RAG 沒有硬性限制路徑或 metadata date。
   - reviewer rubric 沒檢查 evidence provenance date。

## 待辦建議

- 為 extended mode 增加 global per-user-turn tool budget,讓 writer/reviser/final validation 共用同一個工具額度。
- 在 reviewer/reviser handoff 中提供更完整但 bounded 的 evidence ledger,至少包含來源 path/date/query,避免 reviser 盲目重查。
- rewriter 遇到日期上限時,必須把它轉成 retrieval instruction,例如「只讀 `Research_notes/YYYYMMDD <= 20260315`」。
- reviewer 增加 temporal scope check:若 user 指定時間窗,任何 evidence path/date 超界都要標成 major/blocker。
- CLI trace 應標示階段,例如 `[Writer] calling rag_search`、`[Reviser 1] calling rag_search`,避免 flat trace 看起來像同一 agent 爆走。

